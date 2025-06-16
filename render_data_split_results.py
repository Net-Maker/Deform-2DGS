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
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel
import imageio
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns


def render_single_percentage(model_path, percentage, iteration, views, gaussians, pipeline, background, deform, is_6dof, load2gpu_on_the_fly):
    """渲染单个数据百分比的结果"""
    
    render_path = os.path.join(model_path, f"data_{percentage}percent", "renders")
    gts_path = os.path.join(model_path, f"data_{percentage}percent", "gt")
    depth_path = os.path.join(model_path, f"data_{percentage}percent", "depth")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    
    print(f"渲染 {percentage}% 数据的结果...")
    
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {percentage}% data")):
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
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(depth, os.path.join(depth_path, f'{idx:05d}.png'))
        
        if load2gpu_on_the_fly:
            view.load2device('cpu')


def create_comparison_video(base_model_path, percentages, views, iteration):
    """创建不同数据百分比的对比视频"""
    
    comparison_path = os.path.join(base_model_path, "comparison_videos")
    makedirs(comparison_path, exist_ok=True)
    
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    
    # 选择几个关键视角进行对比
    key_views = [0, len(views)//4, len(views)//2, 3*len(views)//4, len(views)-1]
    
    for view_idx in key_views:
        view_comparison = []
        
        for percentage in percentages:
            render_path = os.path.join(base_model_path, f"data_{percentage}percent", "renders")
            if os.path.exists(render_path):
                img_path = os.path.join(render_path, f'{view_idx:05d}.png')
                if os.path.exists(img_path):
                    img = plt.imread(img_path)
                    view_comparison.append(img)
        
        if view_comparison:
            # 创建对比图
            fig, axes = plt.subplots(1, len(view_comparison), figsize=(len(view_comparison)*3, 3))
            if len(view_comparison) == 1:
                axes = [axes]
                
            for i, (img, percentage) in enumerate(zip(view_comparison, percentages)):
                axes[i].imshow(img)
                axes[i].set_title(f'{percentage}% Data')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_path, f'view_{view_idx:05d}_comparison.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()


def create_time_interpolation_comparison(base_model_path, percentages, view_idx, frame_count=150):
    """创建时间插值的对比视频"""
    
    comparison_path = os.path.join(base_model_path, "time_interpolation_comparison")
    makedirs(comparison_path, exist_ok=True)
    
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    
    all_frames = []
    
    for percentage in percentages:
        model_path = os.path.join(base_model_path, f"data_{percentage}percent")
        
        if not os.path.exists(os.path.join(model_path, "point_cloud", f"iteration_{40000}", "point_cloud.ply")):
            print(f"模型文件不存在: {percentage}%")
            continue
            
        # 加载模型
        gaussians = GaussianModel(3)  # sh_degree = 3
        gaussians.load_ply(os.path.join(model_path, "point_cloud", f"iteration_{40000}", "point_cloud.ply"))
        
        deform = DeformModel2D(is_blender=False, is_6dof=False)
        deform.load_weights(model_path)
        
        # 创建虚拟视角
        dummy_scene = Scene(ModelParams(), gaussians, shuffle=False)
        views = dummy_scene.getTestCameras()
        if view_idx >= len(views):
            view_idx = 0
            
        view = views[view_idx]
        
        bg_color = [0, 0, 0]  # 黑色背景
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipeline = PipelineParams()
        
        percentage_frames = []
        
        for t in range(frame_count):
            fid = torch.Tensor([t / (frame_count - 1)]).cuda()
            xyz = gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
            
            results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
            rendering = results["render"]
            percentage_frames.append(to8b(rendering.cpu().numpy()))
        
        all_frames.append(percentage_frames)
    
    # 创建对比视频
    if all_frames:
        combined_frames = []
        
        for frame_idx in range(frame_count):
            frame_row = []
            for percentage_frames in all_frames:
                if frame_idx < len(percentage_frames):
                    frame_row.append(percentage_frames[frame_idx])
            
            if frame_row:
                # 水平拼接帧
                combined_frame = np.concatenate(frame_row, axis=2)  # 沿宽度拼接
                combined_frames.append(combined_frame)
        
        if combined_frames:
            combined_frames = np.stack(combined_frames, 0).transpose(0, 2, 3, 1)
            imageio.mimwrite(os.path.join(comparison_path, 'time_comparison.mp4'), 
                           combined_frames, fps=30, quality=8)


def plot_performance_curves(base_model_path):
    """绘制性能曲线图"""
    
    results_path = os.path.join(base_model_path, "split_experiments_results.json")
    
    if not os.path.exists(results_path):
        print("未找到实验结果文件")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # 提取数据
    percentages = []
    psnr_values = []
    
    for key, value in results.items():
        if value is not None:
            percentage = int(key.replace('%', ''))
            percentages.append(percentage)
            psnr_values.append(value)
    
    # 排序
    sorted_data = sorted(zip(percentages, psnr_values))
    percentages, psnr_values = zip(*sorted_data)
    
    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, psnr_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('数据百分比 (%)')
    plt.ylabel('PSNR (dB)')
    plt.title('数据量与模型性能的关系')
    plt.grid(True, alpha=0.3)
    plt.xticks(percentages)
    
    # 添加数值标签
    for x, y in zip(percentages, psnr_values):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_model_path, 'performance_curve.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建详细的统计表
    stats_text = "数据分割实验结果统计:\n"
    stats_text += "=" * 40 + "\n"
    for percentage, psnr in zip(percentages, psnr_values):
        stats_text += f"{percentage:3d}% 数据: PSNR = {psnr:.3f} dB\n"
    
    # 计算改进程度
    if len(psnr_values) > 1:
        stats_text += "\n改进分析:\n"
        stats_text += "-" * 20 + "\n"
        for i in range(1, len(psnr_values)):
            improvement = psnr_values[i] - psnr_values[0]
            stats_text += f"{percentages[0]}% -> {percentages[i]}%: +{improvement:.3f} dB\n"
    
    with open(os.path.join(base_model_path, 'performance_analysis.txt'), 'w') as f:
        f.write(stats_text)
    
    print("性能分析完成，结果已保存到:")
    print(f"- 性能曲线图: {os.path.join(base_model_path, 'performance_curve.png')}")
    print(f"- 详细分析: {os.path.join(base_model_path, 'performance_analysis.txt')}")


def render_data_split_results(base_model_path, iteration=-1, mode='render'):
    """渲染数据分割实验的所有结果"""
    
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
    
    # 渲染每个百分比的结果
    if mode in ['render', 'all']:
        for percentage in successful_percentages:
            model_path = os.path.join(base_model_path, f"data_{percentage}percent")
            
            if iteration == -1:
                # 寻找最新的iteration
                point_cloud_dir = os.path.join(model_path, "point_cloud")
                if os.path.exists(point_cloud_dir):
                    iterations = [int(d.split('_')[1]) for d in os.listdir(point_cloud_dir) 
                                if d.startswith('iteration_')]
                    if iterations:
                        iteration = max(iterations)
                    else:
                        iteration = 40000
            
            ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
            
            if not os.path.exists(ply_path):
                print(f"模型文件不存在: {ply_path}")
                continue
            
            # 加载模型和场景
            gaussians = GaussianModel(3)  # sh_degree = 3
            gaussians.load_ply(ply_path)
            
            deform = DeformModel2D(is_blender=False, is_6dof=False)
            deform.load_weights(model_path)
            
            # 创建临时场景来获取相机
            temp_parser = ArgumentParser()
            temp_lp = ModelParams(temp_parser)
            temp_args = temp_parser.parse_args(['-s', 'dummy'])  # 占位符
            temp_dataset = temp_lp.extract(temp_args)
            temp_dataset.source_path = os.path.dirname(base_model_path)  # 假设数据在上级目录
            
            try:
                scene = Scene(temp_dataset, gaussians, shuffle=False)
                
                bg_color = [0, 0, 0]  # 黑色背景
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                
                temp_pp = PipelineParams(temp_parser)
                pipeline = temp_pp.extract(temp_args)
                
                # 渲染测试视图
                test_views = scene.getTestCameras()
                render_single_percentage(base_model_path, percentage, iteration, 
                                       test_views, gaussians, pipeline, background, 
                                       deform, False, False)
                
            except Exception as e:
                print(f"渲染 {percentage}% 数据时出错: {str(e)}")
                continue
    
    # 创建对比图和视频
    if mode in ['comparison', 'all']:
        print("创建对比图...")
        # 这里需要根据实际的数据路径调整
        # create_comparison_video(base_model_path, successful_percentages, test_views, iteration)
    
    # 绘制性能曲线
    if mode in ['analysis', 'all']:
        print("分析性能曲线...")
        plot_performance_curves(base_model_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render data split experiment results")
    parser.add_argument("--model_path", "-m", required=True, type=str, 
                       help="Path to the base model directory containing split experiments")
    parser.add_argument("--iteration", default=-1, type=int,
                       help="Iteration to render (-1 for latest)")
    parser.add_argument("--mode", default='all', 
                       choices=['render', 'comparison', 'analysis', 'all'],
                       help="Rendering mode")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    print(f"渲染数据分割实验结果: {args.model_path}")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_data_split_results(args.model_path, args.iteration, args.mode)
    
    print("渲染完成!") 