#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
import json
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh
from gaussian_renderer_2d import render
import cv2
from utils.mesh_simple_utils import Mesh
from scene.cameras import Camera
import scipy
def trimesh2o3d(mesh):
    """
    将 trimesh 对象转换为 open3d mesh 对象
    
    参数:
        mesh: trimesh.Trimesh 对象
        
    返回:
        o3d.geometry.TriangleMesh: 转换后的Open3D mesh对象
    """
    # 检查输入类型
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("输入必须是trimesh.Trimesh对象, mesh type: ", type(mesh))
    
    # 创建Open3D mesh对象
    o3d_mesh = o3d.geometry.TriangleMesh()
    
    # 转换顶点
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=np.float32))
    
    # 转换面片
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces, dtype=np.int32))
    
    # 如果有顶点法向量，也进行转换
    if mesh.vertex_normals is not None:
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(
            np.array(mesh.vertex_normals, dtype=np.float32))
    else:
        # 如果没有法向量，计算法向量
        o3d_mesh.compute_vertex_normals()
    
    # 如果有顶点颜色，进行转换
    if mesh.visual.vertex_colors is not None:
        # trimesh的颜色是RGBA格式，需要转换为RGB并归一化到[0,1]
        vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # 如果有UV坐标和纹理，也可以转换（这里只是示例，实际使用时可能需要根据具体情况调整）
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(mesh.visual.uv)
    
    return o3d_mesh

def o3d2trimesh(mesh):
    """
    将 open3d mesh 对象转换为 trimesh 对象
    
    参数:
        mesh: o3d.geometry.TriangleMesh 对象
        
    返回:
        trimesh.Trimesh: 转换后的trimesh对象
    """
    # 检查输入类型
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError("输入必须是open3d.geometry.TriangleMesh对象")
    
    # 转换顶点和面片
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # 准备其他属性
    vertex_normals = None
    vertex_colors = None
    
    # 转换顶点法向量（如果存在）
    if mesh.has_vertex_normals():
        vertex_normals = np.asarray(mesh.vertex_normals)
    
    # 转换顶点颜色（如果存在）
    if mesh.has_vertex_colors():
        # Open3D的颜色是RGB格式[0,1]，需要转换为RGBA格式[0,255]
        vertex_colors = np.asarray(mesh.vertex_colors)
        vertex_colors = np.concatenate([
            (vertex_colors * 255).astype(np.uint8),
            np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
        ], axis=1)
    
    # 创建trimesh对象
    tri_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        process=False  # 禁用自动处理以保持原始数据
    )
    
    # 如果有顶点颜色，设置颜色
    if vertex_colors is not None:
        tri_mesh.visual.vertex_colors = vertex_colors
    
    # 如果有UV坐标，也可以转换（这里只是示例）
    if hasattr(mesh, 'triangle_uvs') and mesh.triangle_uvs is not None:
        tri_mesh.visual.uv = np.asarray(mesh.triangle_uvs)
    
    return tri_mesh

def simplify_mesh(input_path, target_vertices=5000, simplification_rate=0.2, valence_aware=False, isotropic=False):
    """
    简化3D网格模型的函数
    
    参数:
        input_path (str): 输入网格文件的路径
        target_vertices (int, optional): 目标顶点数量
        simplification_rate (float): 简化率 (0-1之间), 仅当target_vertices为None时使用
        valence_aware (bool): 是否使用考虑顶点度的简化
        isotropic (bool): 是否使用各向同性简化
        
    返回:
        simplified_mesh: 简化后的网格对象
        output_path (str): 保存的输出文件路径
    """
    # 加载网格
    mesh = Mesh(input_path)
    mesh_name = os.path.basename(input_path).split(".")[-2]
    
    # 计算目标顶点数
    if target_vertices is None:
        target_vertices = int(len(mesh.vs) * simplification_rate)
    
    # 验证目标顶点数是否有效
    if target_vertices >= mesh.vs.shape[0]:
        raise ValueError(f"目标顶点数必须小于原始顶点数 {mesh.vs.shape[0]}")
    
    # 执行网格简化
    if isotropic:
        simplified_mesh = mesh.edge_based_simplification(target_v=target_vertices, valence_aware=valence_aware)
    else:
        simplified_mesh = mesh.simplification(target_v=target_vertices, valence_aware=valence_aware)
    
    # 保存结果到原文件夹
    output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{mesh_name}_simplified.ply")
    simplified_mesh.save(output_path)

    result_mesh = trimesh.Trimesh(simplified_mesh.vs, simplified_mesh.faces)
    
    return result_mesh, output_path


def post_process_mesh(mesh, cluster_to_keep=1000, fill_holes=True, max_hole_size=10000):
    """
    Post-process a mesh to filter out floaters and disconnected parts.
    If total clusters are less than cluster_to_keep, keep all clusters.
    
    Args:
        mesh: Input mesh
        cluster_to_keep: Number of largest clusters to keep
        fill_holes: Whether to fill holes in the mesh
        max_hole_size: Maximum hole size to fill (in triangle count)
    """
    import copy
    print("post processing the mesh...")
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    
    # 获取实际的簇数量
    total_clusters = len(np.unique(triangle_clusters))
    # 如果实际簇数量小于要保留的数量，则使用最小的簇大小作为阈值
    if total_clusters <= cluster_to_keep:
        n_cluster = np.min(cluster_n_triangles)
    else:
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    
    n_cluster = max(n_cluster, 100)  # filter meshes smaller than 100
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    print(f"Total clusters: {total_clusters}, Clusters kept: {len(np.unique(triangle_clusters[~triangles_to_remove]))}")

    if fill_holes:
        print("Filling holes... (using pymeshlab)")
        import pymeshlab

        # open3d mesh_0 转 numpy
        vertices = np.asarray(mesh_0.vertices)
        faces = np.asarray(mesh_0.triangles)

        # 用pymeshlab处理
        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(vertices, faces)
        ms.add_mesh(m, "mesh_to_fill")
        # 执行补洞操作
        ms.apply_filter('meshing_repair_non_manifold_vertices')
        ms.apply_filter('meshing_remove_duplicate_vertices')
        ms.apply_filter('meshing_remove_duplicate_faces')
        ms.apply_filter('meshing_repair_non_manifold_edges')
        ms.apply_filter('meshing_close_holes', maxholesize=max_hole_size)
        ms.apply_filter('generate_surface_reconstruction_screened_poisson')

        # 取回修复后的mesh
        filled_mesh = ms.current_mesh()
        filled_vertices = filled_mesh.vertex_matrix()
        filled_faces = filled_mesh.face_matrix()

        # 转回open3d
        mesh_0 = o3d.geometry.TriangleMesh()
        mesh_0.vertices = o3d.utility.Vector3dVector(filled_vertices)
        mesh_0.triangles = o3d.utility.Vector3iVector(filled_faces)
        mesh_0.remove_unreferenced_vertices()
        mesh_0.remove_degenerate_triangles()
        print("Final vertex count: {}".format(len(mesh_0.vertices)))

        # 只保留最大连通分量
        print("Keeping only the largest connected component...")
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        # 找到最大连通分量的ID
        largest_cluster_idx = np.argmax(cluster_n_triangles)
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        mesh_0.remove_unreferenced_vertices()
        mesh_0.remove_degenerate_triangles()

        is_watertight = mesh_0.is_watertight()
        print(f"Watertight (open3d): {is_watertight}")


    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        intrinsic=o3d.camera.PinholeCameraIntrinsic(width=viewpoint_cam.image_width, 
                    height=viewpoint_cam.image_height, 
                    cx = viewpoint_cam.image_width/2,
                    cy = viewpoint_cam.image_height/2,
                    fx = viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx / 2.)),
                    fy = viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy / 2.)))

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.points = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack,pipeline,background, deform,state,depth_filtering, time_input):
        """
        reconstruct radiance field given cameras
        将重建的结果存在内存里
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            xyz = self.gaussians.get_xyz
            
            # 直接调用输入中的time_input
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
            results = render(viewpoint_cam, 
                             self.gaussians, 
                             pipeline, 
                             background, 
                             d_xyz, 
                             d_rotation, 
                             d_scaling)
            
            rgb = results["render"]
            depth = results["surf_depth"]
            # # 深度平滑处理
            # depth_np = depth.cpu().numpy()
            # # 只对有效深度区域做高斯滤波（可选）
            # mask = (depth_np > 0)
            # depth_np_smooth = scipy.ndimage.gaussian_filter(depth_np, sigma=3.0)
            # # 保持无效区域为0
            # depth_np_smooth = depth_np_smooth * mask
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())

        self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        self.depthmaps = torch.stack(self.depthmaps, dim=0)

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.03, sdf_trunc=0.1, depth_trunc=12, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            # print("depth.shape",depth.shape,"depth.min",depth.min(),"depth.max",depth.max())
            
            # if we have mask provided, use it
            # if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
            #     depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            max_depth = np.max(self.depthmaps[idx][0].cpu().numpy())
            cv2.imwrite(os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".png"), self.depthmaps[idx][0].cpu().numpy()/max_depth*255)
            #save_img_u8(self.depthmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".png"))

def generate_camera_positions(n_cameras, radius=1.0, center=np.array([0, 0, 0])):
    """
    使用斐波那契点集方法生成球面上均匀分布的相机位置
    
    参数:
    n_cameras: 相机数量
    radius: 球面半径
    center: 球心坐标
    
    返回:
    positions: 形状为(n_cameras, 3)的数组，包含所有相机位置
    """
    positions = np.zeros((n_cameras, 3))
    
    # 黄金角 ~137.5°
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    for i in range(n_cameras):
        # y坐标均匀分布在[-1,1]之间，避免在极点聚集
        y = 1 - (i / float(n_cameras - 1)) * 2
        
        # 当前y下的圆半径
        radius_at_y = np.sqrt(1 - y * y)
        
        # 使用黄金角递增θ角度
        theta = golden_angle * i
        
        # 计算x和z坐标
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        
        # 缩放到指定半径并加上中心点
        positions[i] = np.array([x, y, z]) * radius + center
    
    return positions

def create_more_camera(existing_cameras, gaussians, json_path=None, save_path=None, radius_factor=1, n_cameras=300, visualize=False, visualize_dir=None):
    # 计算现有相机的中心点和平均距离
    camera_positions = np.array([cam.camera_center.cpu().numpy() for cam in existing_cameras])
    center = np.array([0, 0, 0])

    distances = np.linalg.norm(camera_positions - center, axis=1)
    avg_distance = np.mean(distances)
    radius = avg_distance * radius_factor
    
    # 获取参考相机的内参
    ref_cam = existing_cameras[0]

    
    
    # 如果提供了JSON文件路径，读取现有相机信息以获取最大UID
    max_id = 0
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                json_cameras = json.load(f)
            
            # 找出最大的ID
            for cam in json_cameras:
                max_id = max(max_id, cam['id'])
            
            print(f"从JSON文件中读取到最大ID: {max_id}")
        except Exception as e:
            print(f"读取JSON文件时出错: {e}")
    
    new_positions = generate_camera_positions(n_cameras, radius, center)
    
    # 创建新相机列表
    new_cameras = []
    
    for i in range(n_cameras):
        # 计算相机位置
        position = new_positions[i]
        center = np.array([0, 0, 0])  # 目标点

        # 计算z轴（指向原点）
        z_axis = (center - position)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # 选择up向量
        up = np.array([0, -1, 0])
        if np.abs(np.dot(z_axis, up)) > 0.9:
            up = np.array([0, 0, 1])

        # 计算x/y轴
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # 构建旋转矩阵（列向量为x/y/z轴，世界到相机）
        R = np.stack([x_axis, y_axis, z_axis], axis=1)

        # 如果你的系统需要“世界到相机”变换，R直接用即可；如果需要“相机到世界”，用R.T
        # 这里假设后续用的是“世界到相机”变换

        # 平移向量
        T = -R.T @ position  # 世界到相机坐标系的平移

        
        # 创建新相机ID（续接现有ID）
        new_id = max_id + i + 1
        
        # 创建新相机
        fid = float(i) / max(1, n_cameras - 1)  # 归一化的帧ID
        new_cam = Camera(
            colmap_id=new_id,
            R=R,
            T=T,
            FoVx=ref_cam.FoVx,
            FoVy=ref_cam.FoVy,
            image=existing_cameras[0].original_image,  # 没有图像, 只是用来初始化
            gt_alpha_mask=None,
            image_name=f"spherical_{new_id:03d}",
            uid=new_id,
            fid=fid
        )
        
        new_cameras.append(new_cam)
    
    # 可视化相机分布
    print(f"生成了 {len(new_cameras)} 个球形分布的相机")
    
    # 保存相机到JSON文件（如果提供了保存路径）
    if save_path:
        # 将相机转换为JSON格式
        json_cams = []
        for cam in new_cameras:
            # 构建与现有JSON格式一致的相机数据
            cam_data = {
                "id": int(cam.colmap_id),
                "img_name": cam.image_name,
                "width": int(ref_cam.image_width),
                "height": int(ref_cam.image_height),
                "position": cam.T.tolist(),
                "rotation": cam.R.tolist(),
                "fy": float(ref_cam.fy),
                "fx": float(ref_cam.fx)
            }
            json_cams.append(cam_data)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as file:
            json.dump(json_cams, file, indent=4)
        print(f"相机已保存至: {save_path}")
    
    # 如果需要可视化相机分布
    if visualize:
        simple_visualize_cameras(new_cameras + existing_cameras, points=gaussians.get_xyz().detach().cpu().numpy(), scale=0.1, custom_color=None)
    
    return new_cameras + existing_cameras
def simple_visualize_cameras(cameras, points=None, scale=0.1, custom_color=None):
    """
    简化版相机与点云可视化函数，固定使用Blender标准坐标系（相机沿+Z轴观察）
    
    参数:
        cameras: 相机列表
        points: 点云数据，可以是numpy数组或torch.Tensor，形状为[N, 3]或None
        scale: 相机锥体的缩放比例
        custom_color: 自定义相机颜色字典，格式为{"original": [r,g,b], "generated": [r,g,b]}
    
    返回:
        None，但会打开一个交互式可视化窗口
    """
    print("开始可视化相机与点云...")
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="相机与点云可视化", width=1280, height=720)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    opt.point_size = 2.0  # 点云大小
    
    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord_frame)
    
    # 相机类型到颜色的映射
    if custom_color is None:
        # 默认颜色映射
        camera_colors = {
            "original": [0.0, 0.8, 0.0],  # 原始相机为绿色
            "generated": [0.8, 0.0, 0.0]  # 生成的相机为红色
        }
    else:
        camera_colors = custom_color
    
    # 添加点云（如果提供）
    if points is not None:
        # 将points转换为numpy数组（如果是torch.Tensor）
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        # 创建点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        
        # 给点云着色（白色）
        point_cloud.paint_uniform_color([1.0, 1.0, 1.0])
        
        # 添加到可视化器
        vis.add_geometry(point_cloud)
    
    # 相机ID阈值（假设原始相机ID较小，生成的相机ID较大）
    threshold_id = 100
    
    # 添加相机
    for i, camera in enumerate(cameras):
        try:
            # 从相机获取位置、旋转和视场角
            position = camera.camera_center.cpu().numpy() if hasattr(camera, 'camera_center') else camera.T
            
            # 获取相机旋转矩阵
            if hasattr(camera, 'R'):
                rotation = camera.R.T if torch.is_tensor(camera.R) else camera.R.T
            elif hasattr(camera, 'R_mat'):
                rotation = camera.R_mat.T.cpu().numpy() if torch.is_tensor(camera.R_mat) else camera.R_mat.T
            else:
                print(f"警告: 无法获取相机 {i} 的旋转矩阵，跳过")
                continue
            
            # 获取相机视场角
            fov_x = getattr(camera, 'FoVx', 0.8)  # 默认值约45度
            fov_y = getattr(camera, 'FoVy', 0.6)
            
            # 判断相机类型
            camera_type = "original" if hasattr(camera, 'colmap_id') and camera.colmap_id < threshold_id else "generated"
            color = camera_colors.get(camera_type, [0.5, 0.5, 0.5])  # 如果未指定颜色，使用灰色
            
            # 创建相机视锥体
            # 计算视锥体的近平面和远平面尺寸
            near = 0.1 * scale
            far = 0.5 * scale
            
            # 计算近平面和远平面的尺寸
            near_height = 2 * near * np.tan(fov_y / 2)
            near_width = 2 * near * np.tan(fov_x / 2)
            far_height = 2 * far * np.tan(fov_y / 2)
            far_width = 2 * far * np.tan(fov_x / 2)
            
            # 创建视锥体的点（固定使用blender_neg_z坐标系）
            points = [
                # 相机中心点
                np.array([0, 0, 0]),
                
                # 近平面的四个角点
                np.array([near_width/2, near_height/2, near]),
                np.array([-near_width/2, near_height/2, near]),
                np.array([-near_width/2, -near_height/2, near]),
                np.array([near_width/2, -near_height/2, near]),
                
                # 远平面的四个角点
                np.array([far_width/2, far_height/2, far]),
                np.array([-far_width/2, far_height/2, far]),
                np.array([-far_width/2, -far_height/2, far]),
                np.array([far_width/2, -far_height/2, far])
            ]
            
            # 相机观察方向
            view_dir = np.array([0, 0, 1]) * scale * 0.6
            
            # 定义线段连接关系
            lines = [
                # 从相机中心到近平面的连线
                [0, 1], [0, 2], [0, 3], [0, 4],
                
                # 近平面的边
                [1, 2], [2, 3], [3, 4], [4, 1],
                
                # 远平面的边
                [5, 6], [6, 7], [7, 8], [8, 5],
                
                # 近平面到远平面的连线
                [1, 5], [2, 6], [3, 7], [4, 8]
            ]
            
            # 创建线条集合
            line_set = o3d.geometry.LineSet()
            
            # 转换点到世界坐标系
            transformed_points = []
            for point in points:
                # 先旋转后平移
                transformed_point = rotation @ point + position
                transformed_points.append(transformed_point)
            
            # 设置点和线
            line_set.points = o3d.utility.Vector3dVector(transformed_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            
            # 设置线条颜色
            colors = [color for _ in range(len(lines))]
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            # 添加指示相机观察方向的线段
            direction_line = o3d.geometry.LineSet()
            
            # 观察方向点 - 从相机位置沿观察方向延伸
            view_dir_world = rotation @ view_dir + position
            
            direction_points = [position, view_dir_world]
            direction_lines = [[0, 1]]
            
            direction_line.points = o3d.utility.Vector3dVector(direction_points)
            direction_line.lines = o3d.utility.Vector2iVector(direction_lines)
            
            # 使用蓝色表示观察方向
            direction_colors = [[0.0, 0.0, 1.0] for _ in range(len(direction_lines))]
            direction_line.colors = o3d.utility.Vector3dVector(direction_colors)
            
            # 添加相机锥体和方向线到可视化器
            vis.add_geometry(line_set)
            vis.add_geometry(direction_line)
            
            # 添加相机坐标系
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale*0.5)
            # 设置坐标系变换
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = position
            cam_frame.transform(transform)
            # 添加到可视化器
            vis.add_geometry(cam_frame)
            
        except Exception as e:
            print(f"处理相机 {i} 时出错: {e}")
    
    # 显示帮助信息
    print("\n相机与点云可视化窗口已打开")
    print("控制说明:")
    print("  - 鼠标左键: 旋转场景")
    print("  - 鼠标右键: 平移场景")
    print("  - 鼠标滚轮: 缩放场景")
    
    # 运行可视化循环
    vis.run()
    vis.destroy_window()

def simple_visualize_scene(gaussians, cameras=None, scale=0.1, custom_color=None):
    """
    简化版场景可视化函数，用于可视化高斯点云和相机
    
    参数:
        gaussians: Gaussians对象，用于提取点云
        cameras: 相机列表（可选）
        scale: 相机锥体的缩放比例
        custom_color: 自定义相机颜色字典
    """
    try:
        # 从gaussians获取点云数据
        points = get_points_from_gaussians(gaussians)
        
        # 使用简化版相机可视化函数
        simple_visualize_cameras(cameras if cameras else [], points, scale, custom_color)
    except Exception as e:
        print(f"可视化场景时出错: {e}")
        import traceback
        traceback.print_exc()