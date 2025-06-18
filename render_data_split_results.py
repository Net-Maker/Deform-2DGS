#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel2D
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_2d import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args, get_combined_args_with_model_path
from scene import GaussianModel
import imageio
import numpy as np
import time
import json
import sys


def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    """复刻render_2d.py的render_set函数"""
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        depth = results["surf_depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        
        if load2gpu_on_the_fly:
            view.load2device('cpu')


def interpolate_time(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    """复刻render_2d.py的interpolate_time函数"""
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = len(views)
    idx = torch.randint(0, len(views), (1,)).item()
    print("view idx:",idx)
    view = views[idx]
    renderings = []
    depths = []
    
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["surf_depth"]
        depth = depth / (depth.max() + 1e-5)
        depths.append(to8b(depth.cpu().numpy()))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    depths = np.stack(depths, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    imageio.mimwrite(os.path.join(depth_path, 'depth.mp4'), depths, fps=30, quality=8)


def interpolate_view(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer):
    """复刻render_2d.py的interpolate_view函数"""
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = len(views)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]

    render_poses = torch.stack(render_wander_path(view), 0)

    renderings = []
    depths = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["surf_depth"]
        depth = depth / (depth.max() + 1e-5)
        depths.append(to8b(depth.cpu().numpy()))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    depths = np.stack(depths, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    imageio.mimwrite(os.path.join(depth_path, 'depth.mp4'), depths, fps=30, quality=8)


def interpolate_poses(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer):
    """复刻render_2d.py的interpolate_poses函数"""
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    
    frame = len(views)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    depths = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        depths.append(to8b(depth.cpu().numpy()))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    depths = np.stack(depths, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    imageio.mimwrite(os.path.join(depth_path, 'depth.mp4'), depths, fps=30, quality=8)


def interpolate_view_original(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer):
    """复刻render_2d.py的interpolate_view_original函数"""
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = len(views)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    depths = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        depths.append(to8b(depth.cpu().numpy()))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    depths = np.stack(depths, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)
    imageio.mimwrite(os.path.join(depth_path, 'depth.mp4'), depths, fps=60, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    """复刻render_2d.py的interpolate_all函数"""
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    
    frame = len(views)
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]

    renderings = []
    depths = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["surf_depth"]
        depth = depth / (depth.max() + 1e-5)
        depths.append(to8b(depth.cpu().numpy()))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    depths = np.stack(depths, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    imageio.mimwrite(os.path.join(depth_path, 'depth.mp4'), depths, fps=30, quality=8)


def render_sets_for_percentage(percentage: int, base_model_path: str, iteration: int, skip_train: bool, skip_test: bool, mode: str):
    """为特定百分比的数据渲染结果，严格遵循render_2d.py的逻辑"""
    
    # 构建该百分比的模型路径
    model_path = os.path.join(base_model_path, f"data_{percentage}percent")
    
    # 检查模型是否存在
    if iteration == -1:
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        if os.path.exists(point_cloud_dir):
            iterations = [int(d.split('_')[1]) for d in os.listdir(point_cloud_dir) 
                         if d.startswith('iteration_') and os.path.isdir(os.path.join(point_cloud_dir, d))]
            if iterations:
                current_iteration = max(iterations)
            else:
                current_iteration = 40000
        else:
            current_iteration = 40000
    else:
        current_iteration = iteration
    
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{current_iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"模型文件不存在: {ply_path}")
        return
    
    print(f"渲染 {percentage}% 数据的结果 (iteration {current_iteration})")
    
    # 为每个百分比创建独立的参数解析器，完全模拟render_2d.py的逻辑
    temp_parser = ArgumentParser(description="Testing script parameters")
    temp_model = ModelParams(temp_parser, sentinel=True)
    temp_pipeline = PipelineParams(temp_parser)
    temp_parser.add_argument("--iteration", default=-1, type=int)
    temp_parser.add_argument("--skip_train", action="store_true")
    temp_parser.add_argument("--skip_test", action="store_true")
    temp_parser.add_argument("--quiet", action="store_true")
    temp_parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    

    # 构造该百分比的命令行参数
    cmdline_args = [
        '--iteration', str(current_iteration),
        '--mode', mode
    ]
    if skip_train:
        cmdline_args.append('--skip_train')
    if skip_test:
        cmdline_args.append('--skip_test')
        
    # 使用新的get_combined_args_with_model_path获取完整的参数
    args = get_combined_args_with_model_path(temp_parser, model_path, cmdline_args)
    print(f"成功加载 {percentage}% 数据的参数: " + args.model_path)
        
    # 提取参数对象
    dataset = temp_model.extract(args)
    pipeline = temp_pipeline.extract(args)
        
    # 调用render_sets函数，完全按照render_2d.py的方式
    render_sets_impl(dataset, current_iteration, pipeline, skip_train, skip_test, mode, percentage/100)



def render_sets_impl(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, mode: str, percentage: int):
    """完全复刻render_2d.py的render_sets函数"""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, data_percentage=percentage)
        deform = DeformModel2D(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform)


def render_data_split_results(base_model_path: str, iteration: int, skip_train: bool, skip_test: bool, mode: str):
    """渲染所有数据分割实验的结果"""
    
    # 读取实验结果
    results_path = os.path.join(base_model_path, "split_experiments_results.json")
    
    if not os.path.exists(results_path):
        print("未找到实验结果文件，请先运行数据分割实验")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    successful_percentages = []
    for key, value in results.items():
        if value is not None:
            percentage = int(key.replace('%', ''))
            successful_percentages.append(percentage)
    
    successful_percentages.sort()
    print(f"找到 {len(successful_percentages)} 个成功的实验: {successful_percentages}")
    
    if not successful_percentages:
        print("没有成功的实验结果")
        return
    
    # 为每个百分比渲染结果
    for percentage in successful_percentages:
        print(f"\n{'='*50}")
        print(f"处理 {percentage}% 数据")
        print(f"{'='*50}")
        
        try:
            render_sets_for_percentage(percentage, base_model_path, iteration, skip_train, skip_test, mode)
        except Exception as e:
            print(f"处理 {percentage}% 数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*50}")
    print("所有渲染任务完成！")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Render data split experiment results")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    parser.add_argument("--base_model_path", required=True, type=str, help="Base path containing data split experiments")
    
    # 直接解析命令行参数，不使用get_combined_args
    args = parser.parse_args()
    print("渲染数据分割实验结果: " + args.base_model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_data_split_results(args.base_model_path, args.iteration, args.skip_train, args.skip_test, args.mode) 