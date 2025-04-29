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
from scene import Scene, GaussianModel, DeformModel2D
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel2D(dataset.is_blender, dataset.is_6dof)
    deform.load_weights(dataset.model_path, iteration=40000)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians, load_iteration=40000, shuffle=False) # 读取之前训练完成的结果, 不shuffle
    #ssdr_path = "/home/wjx/research/code/GaussianAnimator/DG-Mesh/outputs/ssdr/ssdr_result_jumping2.npy"
    #gaussians.get_attribute_from_SSDR(ssdr_path)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
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
        mesh_post = post_process_mesh(mesh, cluster_to_keep=1000)

        o3d.io.write_triangle_mesh(mesh_file, mesh)
        o3d.io.write_triangle_mesh(mesh_post_file, mesh_post)
        print("NOTE:======  mesh at the first frame generated in {} ======".format(mesh_path))

        print("NOTE:======  simplify the mesh using Quadric Error Metrics  ======")
        mesh_post_simplified, output_path = simplify_mesh(mesh_post_file)
        mesh_post_simplified = trimesh2o3d(mesh_post_simplified)


if __name__ == "__main__":
    # 获取Gaussianposes
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1000, 3000, 5000, 6000, 7000] + list(range(10000, 40001, 2000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 3_000, 5_000, 6_000,7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")