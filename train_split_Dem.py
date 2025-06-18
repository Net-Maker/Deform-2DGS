import copy,json
import numpy as np
from skimage.morphology import skeletonize_3d as skeletonize
from skimage.morphology import remove_small_holes
from scipy.sparse.csgraph import shortest_path
from scipy.special import softmax
from skimage import filters
from cc3d import largest_k

from seaborn import color_palette
import open3d as o3d
import trimesh

import os
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel, DeformModel2D
from scene.deform_model import PureLBSModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.ssdr_utils import init_bone_label
from utils.mesh_utils import GaussianExtractor,post_process_mesh,simplify_mesh,create_more_camera,trimesh2o3d, o3d2trimesh
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.ops import knn_points

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch

from utils.ssdr_utils import reconstruct_single_pose_vertices, reconstruct_single_pose_rotations
from utils.CoverageAxis_utils import get_CoverageAxis_solver, convert_to_standard_bone_transform
from utils.skeletonizer import visualise_skeletonizer
from utils.DemBones_utils import DemBonesCalculator, generate_bone_position_sequence, generate_bone_position_sequence_from_ssdr

def get_point_sequence(viewpoint_cam_stack, gaussians, deform):
    gaussian_point_cloud = []

    for idx, viewpoint_cam in tqdm(enumerate(viewpoint_cam_stack)):

        fid = viewpoint_cam.fid
        # frame_id = int(fid * 200 + 0.5)
        # print(int(fid * 200 + 0.5))
        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)

        # Query the gaussians
        
        d_xyz, d_rotation, d_scaling = deform.step(
            gaussians.get_xyz.detach(), time_input
        )
        gaussian_point_cloud.append(np.array(gaussians.get_xyz.detach().cpu() + d_xyz.detach().cpu()))

    return np.array(gaussian_point_cloud)

def get_t_gs(viewpoint_cam_stack, gaussians, deform, t):
    fid = viewpoint_cam_stack[t].fid
    N = gaussians.get_xyz.shape[0]
    time_input = fid.unsqueeze(0).expand(N, -1)
    d_xyz, d_rotation, d_scaling = deform.step(
        gaussians.get_xyz.detach(), time_input
    )
    return np.array(gaussians.get_xyz.detach().cpu() + d_xyz.detach().cpu())




def training_with_data_split(dataset, opt, pipe, testing_iterations, saving_iterations, data_percentage):
    """支持数据分割的训练函数"""
    
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

    # SSDR for Init
    viewpoint_stack_static = scene.getTrainCameras().copy()
    fid_for_mesh = viewpoint_stack_static[0].fid
    view_point_num = len(viewpoint_stack_static) - 1 #  减去1,因为这个只是用来索引数组的，需要防止溢出
    poses = get_point_sequence(viewpoint_stack_static, gaussians, deform)
    
    
    # generate mesh for the first frame
    print("NOTE:======  generate mesh at the first frame ======")
    mesh_path = os.path.join(scene.model_path, "mesh")
    os.makedirs(mesh_path, exist_ok=True)
    
    mesh_file = os.path.join(mesh_path, "mesh.ply")
    mesh_post_file = os.path.join(mesh_path, "mesh_post.ply")
    simplified_mesh_file = os.path.join(mesh_path, "mesh_post_simplified.ply")  # 假设这是简化后的文件名
    
    if os.path.exists(simplified_mesh_file):
        print("NOTE:======  simplified mesh already exists, skipping mesh generation and simplification ======")
        mesh_post_simplified = o3d.io.read_triangle_mesh(simplified_mesh_file)
    elif os.path.exists(mesh_file) and os.path.exists(mesh_post_file):
        print("NOTE:======  mesh files exist, only performing mesh simplification ======")
        mesh_post = o3d.io.read_triangle_mesh(mesh_post_file)
        print("NOTE:======  simplify the mesh at the first frame using Quadric Error Metrics  ======")
        mesh_post_simplified, output_path = simplify_mesh(mesh_post_file)
    else:
        # 原有的mesh生成逻辑
        gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
        # 可视化相机位置
        cameras = create_more_camera(scene.getAllCameras(), gaussians)
        gaussExtractor.reconstruction(
            cameras,
            pipe,
            background,
            deform,
            state="mesh",
            depth_filtering=True, 
            time_input=torch.tensor(fid_for_mesh,device="cuda").expand(gaussians.get_xyz.shape[0], 1)
        )
        gaussExtractor.export_image(mesh_path)
        mesh = gaussExtractor.extract_mesh_bounded()
        mesh_post = post_process_mesh(mesh, cluster_to_keep=1000, fill_holes=False)

        o3d.io.write_triangle_mesh(mesh_file, mesh)
        o3d.io.write_triangle_mesh(mesh_post_file, mesh_post)
        print("NOTE:======  mesh at the first frame generated in {} ======".format(mesh_path))

        print("NOTE:======  simplify the mesh using Quadric Error Metrics  ======")
        mesh_post_simplified, output_path = simplify_mesh(mesh_post_file)
        mesh_post_simplified = trimesh2o3d(mesh_post_simplified)

    # # get_CoverageAxis_result
    # print("NOTE:======  Using CoverageAxis to generate the messy and redundant candidate points as the input of SSDR  ======")
    # coarse_bone_dilation = 0.05
    # solver = get_CoverageAxis_solver(mesh_post_simplified, dilation=coarse_bone_dilation, output_path=os.path.join(scene.model_path, "CoverageAxis"))

    # solver.get_moving_mask(poses)
    # solver.vertice_assignments(poses[0], poses, visualize=True, output_path=os.path.join(scene.model_path, "CoverageAxis"))
    
    # # 初始化图结构并分析运动
    # adjacency_matrix, moving_mask = solver.init_graph(poses, mesh_post_simplified, dilation=coarse_bone_dilation, visualize=False)

    # # 基于运动分析生成骨架
    # moving_joints, bones, joints_index, root_indx = solver.skeleton_with_motion(poses, visualize=False)
    # standard_bone_transforms = convert_to_standard_bone_transform(solver.bone_transforms, moving_joints)

    calculator = DemBonesCalculator()
    # 有了顶点分配信息，开始Dem-Bones流程
    if os.path.exists(os.path.join(scene.model_path, "ssdr_result.npy")):
        result_can_be_visualize = np.load(os.path.join(scene.model_path, "ssdr_result.npy"), allow_pickle=True).item()
        weights = result_can_be_visualize["weights"]
        transformations = result_can_be_visualize["transformations"]
        reconstruction = result_can_be_visualize["reconstruction"]
    else:  
        weights, transformations, rmse, reconstruction = calculator.compute_pure_ssdr(
                poses, poses[0], 30, max_iters=10)
        result_can_be_visualize = {
            "rest_pose": poses[0],
            "weights": weights, # (B, N)
            "transformations": transformations, # (T, B, 4, 4)
            "reconstruction": reconstruction, # (T, N, 3)
        }
        np.save(os.path.join(scene.model_path, "ssdr_result.npy"), result_can_be_visualize)
    bone_positions_sequence = generate_bone_position_sequence_from_ssdr(transformations, weights, poses[0])
    #print("moving_joints", moving_joints.max(), moving_joints.min(), moving_joints.shape)
    #print("poses[0].max()", poses[0].max(), "poses[0].min()", poses[0].min(), "poses[0].shape", poses[0].shape, "bone_positions_sequence[0].max()", bone_positions_sequence[0].max(), "bone_positions_sequence[0].min()", bone_positions_sequence[0].min(), "bone_positions_sequence[0].shape", bone_positions_sequence[0].shape)
    calculator.visualize_4d_skinning(reconstruction, weights.T, bone_positions_sequence=bone_positions_sequence, fps=30, bone_size=0.05)
   


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