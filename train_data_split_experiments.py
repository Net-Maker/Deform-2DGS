import os
import sys
import argparse
from argparse import ArgumentParser
import json
import copy
import torch
from train_demo import training, prepare_output_and_logger
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from scene import Scene, GaussianModel, DeformModel2D
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer_2d import render
from random import randint
from tqdm import tqdm
from utils.image_utils import psnr
from utils.general_utils import get_linear_noise_func

def training_with_data_split(dataset, opt, pipe, testing_iterations, saving_iterations, data_percentage):
    """支持数据分割的训练函数"""
    
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel2D(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    # 创建Scene时传入数据百分比
    scene = Scene(dataset, gaussians, data_percentage=data_percentage)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    knn_idx = None
    knn_weights = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc=f"Training progress ({data_percentage*100:.0f}% data)")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    # 导入必要的函数
    from train_demo import compute_scale_consistency_loss, training_report

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        # 从训练集的相机集合随机选取一个
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg_re["rend_dist"]
        rend_normal = render_pkg_re['rend_normal']
        surf_normal = render_pkg_re['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # mask_loss
        gt_mask = viewpoint_cam.gt_alpha_mask.cuda()
        
        # loss
        total_loss = loss + dist_loss + normal_loss

        # 在densify_until_iter之后添加scale consistency loss
        if iteration == opt.densify_until_iter:
            from pytorch3d.ops import knn_points
                
            # 确保使用contiguous的张量来构建KNN
            xyz = gaussians.get_xyz.detach().contiguous()
            knn_result = knn_points(xyz.unsqueeze(0), xyz.unsqueeze(0), K=20+1)
            knn_idx = knn_result.idx.squeeze(0)[:, 1:].contiguous()  # [N, K]
            knn_dists = knn_result.dists.squeeze(0)[:, 1:].sqrt().contiguous()  # [N, K]
            knn_weights = torch.exp(-2000 * knn_dists**2).contiguous()

        if iteration > opt.densify_until_iter:
            lambda_scale = 0.1
            scale_consistency_loss = compute_scale_consistency_loss(gaussians.get_xyz, gaussians.get_scaling, knn_idx, knn_weights)
            total_loss = total_loss + lambda_scale * scale_consistency_loss
        
        total_loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print(f"使用 {data_percentage*100:.1f}% 数据训练完成，最佳PSNR: {best_psnr}")
    return best_psnr


def run_data_split_experiments(args):
    """运行10个数据分割实验，从10%到100%"""
    
    results = {}
    
    for percentage in range(10, 101, 10):  # 10%, 20%, ..., 100%
        print(f"\n{'='*50}")
        print(f"开始运行 {percentage}% 数据的实验")
        print(f"{'='*50}")
        
        # 复制基础参数
        current_args = copy.deepcopy(args)
        
        # 修改输出路径，包含数据百分比信息
        base_model_path = current_args.model_path
        current_args.model_path = os.path.join(base_model_path, f"data_{percentage}percent")
        
        # 设置数据百分比参数
        data_percentage = percentage / 100.0
        
        # 运行训练
        try:
            # 为每个实验创建新的parser，避免参数冲突
            exp_parser = ArgumentParser(description="Training script parameters")
            lp = ModelParams(exp_parser)
            op = OptimizationParams(exp_parser)
            pp = PipelineParams(exp_parser)
            
            dataset = lp.extract(current_args)
            opt = op.extract(current_args)
            pipe = pp.extract(current_args)
            
            best_psnr = training_with_data_split(
                dataset, opt, pipe,
                current_args.test_iterations, 
                current_args.save_iterations,
                data_percentage
            )
            results[f"{percentage}%"] = best_psnr
            print(f"{percentage}% 数据实验完成，最佳PSNR: {best_psnr}")
        except Exception as e:
            print(f"{percentage}% 数据实验失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results[f"{percentage}%"] = None
    
    # 保存实验结果
    results_path = os.path.join(args.model_path, "split_experiments_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("所有实验完成！结果汇总：")
    for percentage, psnr in results.items():
        print(f"{percentage}: PSNR = {psnr}")
    print(f"详细结果已保存到: {results_path}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = ArgumentParser(description="Data split experiments")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, 
                        default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # 确保输出目录存在
    if not args.model_path:
        args.model_path = "./output/data_split_experiments"
    
    print("Optimizing " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    run_data_split_experiments(args) 