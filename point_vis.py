import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer_2d import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel2D
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from pc_viewer import visualize,visualize_with_weights,visualize_with_high_contrast,visualize_with_enhanced_contrast

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



def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel2D(dataset.is_blender, dataset.is_6dof)
    deform.load_weights(dataset.model_path, iteration=10000)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians, load_iteration=10000, shuffle=False) # 读取之前训练完成的结果, 不shuffle
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
    view_point_num = len(viewpoint_stack_static) - 1 #  减去1,因为这个只是用来索引数组的，需要防止溢出
    poses = get_point_sequence(viewpoint_stack_static, gaussians, deform)
    weights, knn_idx, knn_weights = visualize_with_enhanced_contrast(poses)
    # visualize(points=poses, weights=weights)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
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