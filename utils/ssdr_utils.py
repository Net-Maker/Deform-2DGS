# SSDR Implementation in Python
# Dalton Omens

# import maya.api.OpenMaya as om
# import pymel.core as pm
import numpy as np
from scipy.optimize import lsq_linear
from scipy.cluster.vq import vq, kmeans, whiten
import time
import os
import trimesh
import mayavi as mlab
import re
import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from tqdm import tqdm
import torch

# def reconstruct_single_pose_vertices(rest_pose, rest_bones_t, bone_transforms, W):
#     """
#     通过单一骨骼变换和权重计算重建顶点位置。

#     参数:
#     rest_pose (torch.Tensor): 初始静止姿态，形状为 (num_verts, 3)。
#     rest_bones_t (torch.Tensor): 骨骼的静态位置，形状为 (num_bones, 3)。
#     bone_transforms (torch.Tensor): 单一骨骼变换矩阵，形状为 (num_bones, 4, 3)。
#     W (torch.Tensor): 顶点权重，形状为 (num_verts, num_bones)。

#     返回:
#     torch.Tensor: 重建后的顶点位置，形状为 (num_verts, 3)。
#     """
    
#     # 1. 计算 p_corrected = rest_pose[np.newaxis, :, :] - rest_bones_t[:, np.newaxis, :]
#     # 形状变为 (num_bones, num_verts, 3)
#     p_corrected = rest_pose.unsqueeze(0) - rest_bones_t.unsqueeze(1)

#     # 2. 计算变换后的顶点位置
#     # constructions = np.einsum('bij,blk->bil', bone_transforms[:, :3, :], p_corrected)
#     # 形状 由 bone_trans(B,4,3) 和 p_corrected(B,V,3) 得到 constructions(B,V,3)
#     # print(p_corrected.shape, bone_transforms.shape)
#     constructions = torch.einsum('bij,blk->blk', bone_transforms[:, :3, :], p_corrected)

#     # 3. 将平移部分加到 constructions 中
#     # constructions += bone_transforms[:, np.newaxis, 3, :]
#     # print(constructions.shape, bone_transforms.shape)
#     # print(constructions.shape, bone_transforms[:, 3, :].shape)
#     constructions += bone_transforms[:, None, 3, :]

#     # 4. 对每个骨骼的贡献乘以权重
#     # constructions *= (W.T)[:, np.newaxis, :, np.newaxis]
#     constructions *= W.T[:, :, None]

#     # 5. 对所有骨骼的贡献求和，得到最终的顶点位置
#     # vertex_positions = np.sum(constructions, axis=0)
#     vertex_positions = torch.sum(constructions, dim=0)

#     # 返回形状为 (num_verts, 3) 的重建顶点位置
#     return vertex_positions / 10

def reconstruct_single_pose_vertices(rest_pose, rest_bones_t, bone_transforms, W):
    """
    通过单一骨骼变换和权重计算重建顶点位置。

    参数:
    rest_pose (torch.Tensor): 初始静止姿态，形状为 (num_verts, 3)。
    rest_bones_t (torch.Tensor): 骨骼的静态位置，形状为 (num_bones, 3)。
    bone_transforms (torch.Tensor): 单一骨骼变换矩阵，形状为 (num_bones, 4, 3)。
    W (torch.Tensor): 顶点权重，形状为 (num_verts, num_bones)。

    返回:
    torch.Tensor: 重建后的顶点位置，形状为 (num_verts, 3)。
    """
    trans = bone_transforms.unsqueeze(1)
    result = reconstruct_vertices(rest_pose, rest_bones_t, trans, W)
    # result 形状为 (1, num_verts, 3)
    return result.squeeze(0)

def reconstruct_vertices(rest_pose, rest_bones_t, bone_transforms, W):
    """
    通过骨骼变换和权重计算重建顶点位置。

    参数:
    rest_pose (torch.Tensor): 初始静止姿态，形状为 (num_verts, 3)。
    rest_bones_t (torch.Tensor): 骨骼的静态位置，形状为 (num_bones, 3)。
    bone_transforms (torch.Tensor): 骨骼变换矩阵，形状为 (num_bones, num_poses, 4, 3)。
    W (torch.Tensor): 顶点权重，形状为 (num_verts, num_bones)。

    返回:
    torch.Tensor: 重建后的顶点位置，形状为 (num_poses, num_verts, 3)。
    """
    
    # 确保输入的张量为浮点数类型
    # rest_pose = rest_pose.float()
    # rest_bones_t = rest_bones_t.float()
    # bone_transforms = bone_transforms.float()
    # W = W.float()

    # 1. 计算 p_corrected = rest_pose[np.newaxis, :, :] - rest_bones_t[:, np.newaxis, :]
    # 形状变为 (num_bones, num_verts, 3)
    p_corrected = rest_pose.unsqueeze(0) - rest_bones_t.unsqueeze(1)

    # 2. 计算变换后的顶点位置
    # constructions = np.einsum('bijk,blk->bilj', bone_transforms[:, :, :3, :], p_corrected)
    # 形状变为 (num_bones, num_poses, num_verts, 3)
    constructions = torch.einsum('bijk,blk->bilj', bone_transforms[:, :, :3, :], p_corrected)

    # 3. 将平移部分加到 constructions 中
    # constructions += bone_transforms[:, :, np.newaxis, 3, :]
    constructions += bone_transforms[:, :, None, 3, :]

    # 4. 对每个骨骼的贡献乘以权重
    # constructions *= (W.T)[:, np.newaxis, :, np.newaxis]
    constructions *= W.T[:, None, :, None]

    # 5. 对所有骨骼的贡献求和，得到最终的顶点位置
    # vertex_positions = np.sum(constructions, axis=0)
    vertex_positions = torch.sum(constructions, dim=0)

    # 返回形状为 (num_poses, num_verts, 3) 的重建顶点位置
    return vertex_positions


def reconstruct_rotations(rest_bones_t, bone_transforms, W):
    """
    通过骨骼变换和权重计算重建顶点旋转。

    参数:
    rest_bones_t (torch.Tensor): 骨骼的静态位置，形状为 (num_bones, 3)。
    bone_transforms (torch.Tensor): 骨骼变换矩阵，形状为 (num_bones, num_poses, 4, 4)。
    W (torch.Tensor): 顶点权重，形状为 (num_verts, num_bones)。

    返回:
    torch.Tensor: 重建后的顶点旋转，形状为 (num_poses, num_verts, 3, 3)。
    """
    
    # 提取旋转部分
    rotations = bone_transforms[:, :, :3, :3]  # 形状为 (num_bones, num_poses, 3, 3)

    # 对每个骨骼的旋转乘以权重
    weighted_rotations = rotations.unsqueeze(2) * W.T[:, None, :, None, None]  # 形状为 (num_bones, num_poses, num_verts, 3, 3)

    # 对所有骨骼的旋转求和，得到最终的顶点旋转
    vertex_rotations = torch.sum(weighted_rotations, dim=0)  # 形状为 (num_poses, num_verts, 3, 3)

    return vertex_rotations

def reconstruct_single_pose_rotations(rest_bones_t, bone_transforms, W):
    """
    通过骨骼变换和权重计算重建单个顶点旋转。

    参数:
    rest_bones_t (torch.Tensor): 骨骼的静态位置，形状为 (num_bones, 3)。
    bone_transforms (torch.Tensor): 骨骼变换矩阵，形状为 (num_bones, 4, 4)。
    W (torch.Tensor): 顶点权重，形状为 (num_verts, num_bones)。

    返回:
    torch.Tensor: 重建后的单个顶点旋转，形状为 (num_verts, 3, 3)。
    """
    
    # 提取旋转部分
    rotations = bone_transforms[:, :3, :3]  # 形状为 (num_bones, 3, 3)

    # 对每个骨骼的旋转乘以权重
    weighted_rotations = rotations.unsqueeze(1) * W.T[:, :, None, None]  # 形状为 (num_bones, num_verts, 3, 3)

    # 对所有骨骼的旋转求和，得到最终的顶点旋转
    vertex_rotations = torch.sum(weighted_rotations, dim=0)  # 形状为 (num_verts, 3, 3)

    return vertex_rotations


def create_bone_point_cloud(bone_positions, color=[1, 0, 0], size=0.05):
    """
    创建骨骼点云
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bone_positions)
    pcd.paint_uniform_color(color)
    
    # 创建球体表示骨骼点
    spheres = []
    for i, center in enumerate(bone_positions):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(color)
        mesh_sphere.translate(center)
        spheres.append(mesh_sphere)
    
    return spheres

def create_vertex_point_cloud(vertices, color=[0, 0.7, 0.7]):
    """
    创建顶点点云
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.paint_uniform_color(color)
    return pcd

def visualize_bones_and_vertices(rest_pose, bone_positions, window_name="Bones and Vertices"):
    """
    可视化骨骼和顶点
    
    inputs: rest_pose      |num_verts| x 3 顶点坐标
            bone_positions |num_bones| x 3 骨骼坐标
            window_name   窗口标题
    """
    # 初始化GUI应用
    app = gui.Application.instance
    if app is None:
        app = gui.Application.instance
    app.initialize()
    
    # 创建可视化窗口
    vis = o3d.visualization.O3DVisualizer(window_name, 1024, 768)
    vis.show_settings = True
    
    # 添加顶点点云
    vertex_pcd = create_vertex_point_cloud(rest_pose)
    vis.add_geometry("vertices", vertex_pcd)
    
    # 添加骨骼点
    bone_spheres = create_bone_point_cloud(bone_positions)
    for i, sphere in enumerate(bone_spheres):
        vis.add_geometry(f"bone_{i}", sphere)
    
    # 添加骨骼索引标签
    for i, pos in enumerate(bone_positions):
        vis.add_3d_label(pos, str(i))
    
    # 重置相机视角
    vis.reset_camera_to_default()
    
    # 运行可视化
    app.add_window(vis)
    app.run()

def visualize_bone_assignments(rest_pose, bone_positions, vert_assignments, window_name="Bone Assignments"):
    """
    可视化骨骼分配结果，每个骨骼的顶点使用不同颜色
    
    inputs: rest_pose       |num_verts| x 3 顶点坐标
            bone_positions  |num_bones| x 3 骨骼坐标
            vert_assignments |num_verts| 顶点到骨骼的分配
            window_name    窗口标题
    """
    # 初始化GUI应用
    app = gui.Application.instance
    if app is None:
        app = gui.Application.instance
        app.initialize()
    
    # 创建可视化窗口
    vis = o3d.visualization.O3DVisualizer(window_name, 1024, 768)
    vis.show_settings = True
    
    # 为每个骨骼生成随机颜色
    num_bones = len(bone_positions)
    bone_colors = np.random.rand(num_bones, 3)
    
    # 创建顶点点云并根据骨骼分配设置颜色
    vertex_pcd = o3d.geometry.PointCloud()
    vertex_pcd.points = o3d.utility.Vector3dVector(rest_pose)
    vertex_colors = bone_colors[vert_assignments]
    vertex_pcd.colors = o3d.utility.Vector3dVector(vertex_colors)
    vis.add_geometry("vertices", vertex_pcd)
    
    # 添加骨骼点
    bone_spheres = create_bone_point_cloud(bone_positions, color=[1, 0, 0])
    for i, sphere in enumerate(bone_spheres):
        vis.add_geometry(f"bone_{i}", sphere)
    
    # 添加骨骼索引标签
    for i, pos in enumerate(bone_positions):
        vis.add_3d_label(pos, str(i))
    
    # 重置相机视角
    vis.reset_camera_to_default()
    
    # 运行可视化
    app.add_window(vis)
    app.run()

# # 使用示例
# if __name__ == "__main__":
#     # 加载数据
#     data = np.load("result.npy", allow_pickle=True).item()
#     rest_pose = data['rest_pose']
#     bone_positions = data['rest_bones_t']
    
#     # 基础可视化
#     visualize_bones_and_vertices(rest_pose, bone_positions)
    
#     # 如果有顶点分配信息，可视化分配结果
#     if 'vert_assignments' in data:
#         vert_assignments = data['vert_assignments']
#         visualize_bone_assignments(rest_pose, bone_positions, vert_assignments)

def visualize_SSDR_result_pc(iter, W, rest_pose, rest_bone, bone_trans):
    # results = np.load("/home/wjx/research/code/DG-Mesh/outputs/ssdr/ssdr_result.npy", allow_pickle=True).item()
    # rest_pose = results["rest_pose"]
    # W = results["W"]
    # rest_bone = results["rest_bone"]
    # bone_trans = results["Bone_Trans"]
    # print("W shape:", W.shape)
    # print("Bone Transforms shape:", bone_trans.shape)
    # print("Rest Bones shape:", rest_bone.shape)
    # print("Rest Pose shape:", rest_pose.shape)

    # for i in range(bone_trans.shape[0]):
    #     bone_index = i  # 选择一个骨骼的索引
    #     weights = W[:, bone_index]  # 获取对应骨骼的weights

    #     mlab.figure(bgcolor=(1, 1, 1))
    #     points = rest_pose
    #     # 通过颜色显示skin weights
    #     mlab.points3d(points[:, 0], points[:, 1], points[:, 2], weights, scale_factor=0.01, colormap='viridis')
    #     mlab.title(f"Skinning Weights for Bone {bone_index}")
    #     mlab.show()
    # 计算变换后的点云
    transformed_points, transformed_bones = apply_bone_transforms(rest_pose, W, bone_trans)

    # 配置绘制参数
    colors = [(1, 0, 0), (0, 0, 1)]  # 每个点云的颜色 (红色和绿色)
    opacity = [0.5, 1]
    sizes = [0.01, 0.1]  # 每个点云的大小
    camera_params = {
        'azimuth': -90,
        'elevation': 75,
        'distance': 6,
        'focalpoint': (0, 0, 0)
    }
    image_size = (1024, 768)  # 渲染图像大小
    output_path = "outputs/ssdr/results_pc"

    for i, (frame_points, frame_bones) in enumerate(zip(transformed_points, transformed_bones)):
        mlab.figure(bgcolor=(1, 1, 1), size=image_size)
        mlab.clf()

        mlab.points3d(frame_points[:, 0], frame_points[:, 1], frame_points[:, 2], color=colors[0],
                      scale_factor=sizes[0], opacity=opacity[0])
        frame_bones = np.array(frame_bones)
        print(frame_bones.shape)
        mlab.points3d(frame_bones[:, 0], frame_bones[:, 1], frame_bones[:, 2], color=colors[1], scale_factor=sizes[1],
                      opacity=opacity[1])
        mlab.view(azimuth=camera_params.get('azimuth', 90),
                  elevation=camera_params.get('elevation', 90),
                  distance=camera_params.get('distance', 10),
                  focalpoint=camera_params.get('focalpoint', (0, 0, 0)))

        mlab.savefig(output_path + f"/frame_{i}.png")
        mlab.close()

def kabsch(P, Q):
    """
    Computes the optimal translation and rotation matrices that minimize the 
    RMS deviation between two sets of points P and Q using Kabsch's algorithm.
    More here: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Inspiration: https://github.com/charnley/rmsd
    
    inputs: P  N x 3 numpy matrix representing the coordinates of the points in P
            Q  N x 3 numpy matrix representing the coordinates of the points in Q
            
    return: A 4 x 3 matrix where the first 3 rows are the rotation and the last is translation
    """
    if (P.size == 0 or Q.size == 0):
        raise ValueError("Empty matrices sent to kabsch")
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P                       # Center both matrices on centroid
    Q_centered = Q - centroid_Q
    H = P_centered.T.dot(Q_centered)                  # covariance matrix
    U, S, V = np.linalg.svd(H)                        # SVD
    R = U.dot(V).T                                    # calculate optimal rotation
    if np.linalg.det(R) < 0:                          # correct rotation matrix for             
        V[2,:] *= -1                                  #  right-hand coordinate system
        R = U.dot(V).T                          
    t = centroid_Q - R.dot(centroid_P)                # translation vector
    return np.vstack((R, t))


def remove_unused_bones(vert_assignments, num_bones, threshold=30):
    """
    移除分配顶点数量少于阈值的骨骼
    
    inputs: vert_assignments |num_verts| array of bone indices
            num_bones       Current number of bones
            threshold      Minimum number of vertices per bone
            
    return: new_assignments, new_num_bones
    """
    # 统计每个骨骼的顶点数量
    bone_counts = np.bincount(vert_assignments, minlength=num_bones)
    
    # 找出顶点数量小于阈值的骨骼
    unused_bones = np.where(bone_counts < threshold)[0]
    
    if len(unused_bones) == 0:
        return vert_assignments, num_bones
        
    # 创建新的骨骼索引映射
    valid_bones = np.where(bone_counts >= threshold)[0]
    new_num_bones = len(valid_bones)
    bone_map = np.full(num_bones, -1)
    for i, bone in enumerate(valid_bones):
        bone_map[bone] = i
        
    # 创建新的顶点分配数组
    new_assignments = np.full_like(vert_assignments, -1)
    
    # 首先处理有效骨骼的顶点
    valid_verts = np.where(np.isin(vert_assignments, valid_bones))[0]
    new_assignments[valid_verts] = bone_map[vert_assignments[valid_verts]]
    
    # 处理无效骨骼的顶点
    invalid_verts = np.where(np.isin(vert_assignments, unused_bones))[0]
    
    # 将这些顶点分配给最近的有效骨骼
    for vert in invalid_verts:
        min_dist = float('inf')
        best_bone = 0
        for new_bone, old_bone in enumerate(valid_bones):
            # 计算到该骨骼所有顶的平均位置的距离
            bone_verts = np.where(vert_assignments == old_bone)[0]
            if len(bone_verts) > 0:
                bone_center = np.mean(rest_pose[bone_verts], axis=0)
                dist = np.linalg.norm(rest_pose[vert] - bone_center)
                if dist < min_dist:
                    min_dist = dist
                    best_bone = new_bone
        new_assignments[vert] = best_bone
    
    return new_assignments, new_num_bones


def compute_bone_positions(rest_pose, vert_assignments, num_bones, surface_constraint=False):
    """
    计算骨骼的最优位置
    
    inputs: rest_pose          顶点位置
            vert_assignments   顶点到骨骼的分配
            num_bones         骨骼数量
            surface_constraint 是否强制骨骼在表面上
    """
    bone_positions = np.zeros((num_bones, 3))
    
    for bone in range(num_bones):
        # 获取该骨骼对应的顶点
        bone_verts = np.where(vert_assignments == bone)[0]
        vert_positions = rest_pose[bone_verts]
        
        if len(bone_verts) == 0:
            continue
            
        # 计算质心作为初始位置
        centroid = np.mean(vert_positions, axis=0)
        
        if surface_constraint:
            # 找到最近的表面点
            distances = np.linalg.norm(vert_positions - centroid, axis=1)
            nearest_surface_idx = np.argmin(distances)
            bone_positions[bone] = vert_positions[nearest_surface_idx]
        else:
            bone_positions[bone] = centroid
            
    return bone_positions


def optimize_bone_transforms(poses, rest_pose, bone_positions, vert_assignments):
    """
    优化骨骼变换，保持局部结构
    """
    num_bones = len(bone_positions)
    num_poses = poses.shape[0]
    bone_transforms = np.empty((num_bones, num_poses, 4, 3))
    
    
    for bone in range(num_bones):
        # 计算骨骼中心点
        bone_verts = np.where(vert_assignments == bone)[0]
        if len(bone_verts) == 0:
            continue
            
        # 所有点都减去骨骼中心点
        rest_pose_corrected = rest_pose - bone_positions[bone]
        
        for pose in range(num_poses):
            # 计算最优刚体变换
            transform = kabsch(
                rest_pose_corrected[bone_verts],  # 只取该骨骼对应的点
                poses[pose, bone_verts]  # 直接使用目标pose的坐标
            )
            
            bone_transforms[bone, pose] = transform
    

            
    return bone_transforms



def optimize_bone_structure(W, bone_transforms, rest_bones_t, rest_pose, threshold=0.1, 
                          merge_distance_threshold=0.25, merge_transform_threshold=0.4):
    """
    优化骨骼结构：
    1. 删除影响较小的骨骼
    2. 合并距离相近且运动相似的骨骼
    
    参数:
    - W: shape (num_vertices, num_bones) 权重矩阵
    - bone_transforms: shape (num_bones, num_frames, 4, 3) 骨骼变换矩阵
    - rest_bones_t: shape (num_bones, 3) 初始骨骼位置
    - rest_pose: shape (num_vertices, 3) 初始点云位置
    - threshold: float, 骨骼影响阈值
    - merge_distance_threshold: float, 骨骼合并的距离阈值
    - merge_transform_threshold: float, 骨骼合并的变换相似度阈值
    """
    num_vertices = W.shape[0]
    num_bones = W.shape[1]
    
    # 1. 评估每个骨骼的影响力
    bone_influence = np.zeros(num_bones)
    for i in range(num_bones):
        # 计算骨骼的最大权重和平均权重
        max_weight = np.max(W[:, i])
        mean_weight = np.mean(W[:, i])
        # 综合评分 = 最大权重 * 0.4 + 平均权重 * 0.3 
        bone_influence[i] = max_weight * 0.4 + mean_weight * 0.3
    
    # 2. 确定要删除的骨骼
    removed_bones = np.where(bone_influence < threshold)[0]
    kept_bones = np.where(bone_influence >= threshold)[0]
    
    print(f"删除的骨骼数量: {len(removed_bones)}")
    print(f"保留的骨骼数量: {len(kept_bones)}")
    
    # 3. 重新分配权重
    new_W = np.zeros((num_vertices, len(kept_bones)))
    
    for vertex_idx in range(num_vertices):
        # 对于每个顶点
        old_weights = W[vertex_idx]
        removed_weight_sum = np.sum(old_weights[removed_bones])
        
        if removed_weight_sum > 0:
            # 找到最近的3个保留的骨骼
            vertex_pos = rest_pose[vertex_idx]
            kept_bones_pos = rest_bones_t[kept_bones]
            distances = np.linalg.norm(kept_bones_pos - vertex_pos, axis=1)
            closest_bones_idx = np.argsort(distances)[:3]
            
            # 基于距离的反比重新分配权重
            inv_distances = 1 / (distances[closest_bones_idx] + 1e-6)
            weight_factors = inv_distances / np.sum(inv_distances)
            
            # 设置新的权重
            for i, bone_idx in enumerate(closest_bones_idx):
                new_W[vertex_idx, bone_idx] = (old_weights[kept_bones[bone_idx]] + 
                                             removed_weight_sum * weight_factors[i])
        else:
            # 如果顶点不受被删除骨骼的影响，直接复制原始权重
            new_W[vertex_idx] = old_weights[kept_bones]
    
    # 4. 归一化权重
    row_sums = new_W.sum(axis=1)
    new_W = new_W / row_sums[:, np.newaxis]
    
    # 5. 更新骨骼变换矩阵和初始骨骼位置
    new_bone_transforms = bone_transforms[kept_bones]
    new_rest_bones_t = rest_bones_t[kept_bones]

    # 6. 合并距离非常近且运动非常相似的骨骼
    def compute_transform_similarity(trans1, trans2):
        """计算两个变换矩阵序列的相似度"""
        # 计算旋转部分的差异
        rotation_diff = np.mean(np.linalg.norm(trans1[:, :3, :] - trans2[:, :3, :], axis=(1,2)))
        # 计算平移部分的差异
        translation_diff = np.mean(np.linalg.norm(trans1[:, 3, :] - trans2[:, 3, :], axis=1))
        return rotation_diff + translation_diff
    
    def merge_bones(bone1_idx, bone2_idx):
        """合并两个骨骼
        
        合并策略：
        1. 位置：基于权重的加权平均
        2. 变换：基于权重插值变换矩阵
        3. 权重：直接相加
        """
        # 计算两个骨骼的总影响力
        w1 = np.sum(new_W[:, bone1_idx])
        w2 = np.sum(new_W[:, bone2_idx])
        total_weight = w1 + w2
        w1_ratio = w1 / total_weight
        w2_ratio = w2 / total_weight
        
        # 1. 计算新的骨骼位置（基于权重的加权平均）
        new_position = (new_rest_bones_t[bone1_idx] * w1_ratio + 
                       new_rest_bones_t[bone2_idx] * w2_ratio)
        
        # 2. 插值变换矩阵
        # 对每一帧进行插值
        num_frames = new_bone_transforms.shape[1]
        new_transform = np.zeros_like(new_bone_transforms[bone1_idx])
        
        for frame in range(num_frames):
            # 分别处理旋转和平移
            # 2.1 处理旋转部分 (前3x3)
            rot1 = new_bone_transforms[bone1_idx, frame, :3, :]
            rot2 = new_bone_transforms[bone2_idx, frame, :3, :]
            
            # 将旋转矩阵转换为四元数进行插值
            from scipy.spatial.transform import Rotation
            r1 = Rotation.from_matrix(rot1)
            r2 = Rotation.from_matrix(rot2)
            # 使用slerp进行球面线性插值
            # 正确的slerp插值方法
            key_rots = Rotation.from_matrix([rot1, rot2])
            times = [0, 1]
            slerp = Rotation.from_rotvec(
                key_rots.as_rotvec()[0] * (1 - w2_ratio) + 
                key_rots.as_rotvec()[1] * w2_ratio
            )
            new_transform[frame, :3, :] = slerp.as_matrix()
            
            # 2.2 处理平移部分 (第4行)
            trans1 = new_bone_transforms[bone1_idx, frame, 3, :]
            trans2 = new_bone_transforms[bone2_idx, frame, 3, :]
            # 线性插值平移向量
            new_transform[frame, 3, :] = trans1 * w1_ratio + trans2 * w2_ratio
        
        # 3. 合并权重
        merged_weights = new_W[:, bone1_idx] + new_W[:, bone2_idx]
        
        return new_position, new_transform, merged_weights
    
    # 迭代直到没有可以合并的骨骼对
    while True:
        num_current_bones = len(new_rest_bones_t)
        bones_to_merge = []
        
        # 寻找可以合并的骨骼对
        for i in range(num_current_bones):
            for j in range(i+1, num_current_bones):
                # 计算空间距离
                distance = np.linalg.norm(new_rest_bones_t[i] - new_rest_bones_t[j])
                
                if distance < merge_distance_threshold:
                    # 计算变换相似度
                    transform_diff = compute_transform_similarity(
                        new_bone_transforms[i], new_bone_transforms[j])
                    print(f"bone{i} and bone{j} transform_diff: {transform_diff}")
                    if transform_diff < merge_transform_threshold:
                        bones_to_merge.append((i, j))
        
        if not bones_to_merge:
            break
            
        # 执行合并
        # 从后向前合并，避免索引变化的影响
        bones_to_merge.sort(key=lambda x: (-x[0], -x[1]))
        
        # 创建一个集合来跟踪已经被合并的骨骼
        merged_bones = set()
        
        for bone1_idx, bone2_idx in bones_to_merge:
            # 检查这些骨骼是否已经被合并
            if bone1_idx in merged_bones or bone2_idx in merged_bones:
                continue
                
            # 合并骨骼
            new_position, new_transform, merged_weights = merge_bones(bone1_idx, bone2_idx)
            
            # 更新数据
            # 1. 更新位置
            new_rest_bones_t[bone1_idx] = new_position
            # 2. 更新变换矩阵
            new_bone_transforms[bone1_idx] = new_transform
            # 3. 更新权重
            new_W[:, bone1_idx] = merged_weights
            
            # 删除被合并的骨骼
            new_rest_bones_t = np.delete(new_rest_bones_t, bone2_idx, axis=0)
            new_bone_transforms = np.delete(new_bone_transforms, bone2_idx, axis=0)
            new_W = np.delete(new_W, bone2_idx, axis=1)
            
            # 将这两个骨骼标记为已合并
            merged_bones.add(bone2_idx)
            
            # 更新所有待合并对中的索引，因为删除操作会改变后续索引
            bones_to_merge = [(b1, b2 if b2 < bone2_idx else b2 - 1) 
                            for b1, b2 in bones_to_merge 
                            if b1 != bone1_idx and b2 != bone2_idx]
        
        print(f"合并了 {len(merged_bones)} 个骨骼")
    
    # 最后再次归一化权重
    row_sums = new_W.sum(axis=1)
    new_W = new_W / row_sums[:, np.newaxis]
    
    return new_W, new_bone_transforms, new_rest_bones_t


def initialize_with_structure(poses, rest_pose, num_bones, faces=None, iterations=5):
    """
    带有结构保持约束的骨骼初始化
    """
    # 1. 初始化顶点分配
    whitened = whiten(rest_pose)
    codebook, _ = kmeans(whitened, num_bones)
    vert_assignments, _ = vq(whitened, codebook)
    
    # 2. 移除无用骨骼
    vert_assignments, num_bones = remove_unused_bones(
        vert_assignments, num_bones, threshold=300
    )
    print("actual num_bones:", num_bones)
    
    # 3. 计算最优骨骼位置
    bone_positions = compute_bone_positions(
        rest_pose, vert_assignments, num_bones, 
        surface_constraint=True
    )
    
    # 4. 优化骨骼变换
    bone_transforms = optimize_bone_transforms(
        poses, rest_pose, bone_positions, 
        vert_assignments
    )
    
    # 5. 迭代优化
    num_poses = poses.shape[0]
    for it in range(iterations):
        # 更新顶点分配
        constructed = np.empty((num_bones, num_poses, rest_pose.shape[0], 3))
        for bone in range(num_bones):
            # 应用骨骼变换
            R = bone_transforms[bone, :, :3, :]
            t = bone_transforms[bone, :, 3, :]
            local_coords = rest_pose - bone_positions[bone]
            
            for pose in range(num_poses):
                constructed[bone, pose] = (
                    R[pose].dot(local_coords.T).T + t[pose]
                )
        
        # 计算重建误差
        errors = np.linalg.norm(
            constructed - poses[np.newaxis, :, :, :], 
            axis=3
        )
        
        # 更新顶点分配
        vert_assignments = np.argmin(errors.mean(axis=1), axis=0)
        
        # 更新骨骼位置和变换
        bone_positions = compute_bone_positions(
            rest_pose, vert_assignments, num_bones
        )
        bone_transforms = optimize_bone_transforms(
            poses, rest_pose, bone_positions, 
            vert_assignments
        )
    
    # visualize_bones_and_vertices(rest_pose, bone_positions)
    return bone_transforms, bone_positions


def initialize(poses, rest_pose, num_bones, iterations=5,initial_bone=None):
    """
    Uses the k-means algorithm to initialize bone transformations.

    inputs: poses       |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose   |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones   Number of bones to initialize
            iterations  Number of iterations to run the k-means algorithm

    return: A |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked Rotation and Translation
              for each pose, for each bone.
            A |num_bones| x 3 matrix representing the translations of the rest bones.
    """
    num_verts = rest_pose.shape[0]
    num_poses = poses.shape[0]

    # Use k-means to assign bones to vertices
    whitened = whiten(rest_pose)
    codebook, _ = kmeans(whitened, num_bones)
    # codebook = np.array(initial_bone)
    vert_assignments, _ = vq(whitened, codebook) # Bone assignment for each vertex (|num_verts| x 1) 将k-means分配的簇中的点的label分配
    print("org_vert_assignments.shape:", vert_assignments.shape)
    # 我希望在这里加一个去除无用骨骼的步骤
    vert_assignments, num_bones = remove_unused_bones(vert_assignments, num_bones, threshold=300)
    print("actual num_bones:", num_bones)
    print("actual vert_assignments:", vert_assignments.shape)
    
    bone_counts = np.bincount(vert_assignments, minlength=num_bones)
    print("bone_counts:", bone_counts)

    bone_transforms = np.empty((num_bones, num_poses, 4, 3))   # [(R, T) for for each pose] for each bone
                                                               # 3rd dim has 3 rows for R and 1 row for T            
    rest_bones_t = np.empty((num_bones, 3))                    # Translations for bones at rest pose
    rest_pose_corrected = np.empty((num_bones, num_verts, 3))  # Rest pose - mean of vertices attached to each bone

    # Compute initial random bone transformations
    for bone in range(num_bones):
        rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0) #骨骼中心点
        rest_pose_corrected[bone] = rest_pose - np.mean(rest_pose[vert_assignments == bone], axis=0)
        for pose in range(num_poses):
            bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone]) # 算出每个骨骼的刚性变换
    
    for it in range(iterations):
        # Re-assign bones to vertices using smallest reconstruction error from all poses
        constructed = np.empty((num_bones, num_poses, num_verts, 3)) # |num_bones| x |num_poses| x |num_verts| x 3
        for bone in range(num_bones):
            Rp = bone_transforms[bone,:,:3,:].dot((rest_pose - rest_bones_t[bone]).T).transpose((0, 2, 1)) # |num_poses| x |num_verts| x 3
            # R * p + T
            constructed[bone] = Rp + bone_transforms[bone, :, np.newaxis, 3, :]
        errs = np.linalg.norm(constructed - poses, axis=(1, 3)) # 对每个pose，计算每个骨骼带来的变换的误差的和，每个骨骼会对应一个RT变换，由此来计算误差的和，下次分配的时候，找出该点对应的误差最小的骨骼，进行二次分配
        vert_assignments = np.argmin(errs, axis=0)    
        
        ## Visualization of vertex assignments for bone 0 over iterations
        ## Make 5 copies of an example pose mesh and call them test0, test1...
        #for i in range(num_verts):
        #    if vert_assignments[i] == 0:
        #        pm.select('test{0}.vtx[{1}]'.format(it, i), add=True)
        #print(vert_assignments)

        # For each bone, for each pose, compute new transform using kabsch
        for bone in range(num_bones):
            rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0)
            rest_pose_corrected[bone] = rest_pose - np.mean(rest_pose[vert_assignments == bone], axis=0)
            for pose in range(num_poses):
                bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone])

    return bone_transforms, rest_bones_t

def Cluster(bones,points):
    whiten_pose = whiten(points)
    bones = np.array(bones)
    vert_assignments,_ = vq(whiten_pose,bones)

    return vert_assignments





def initialize_w_Bone(poses, rest_pose, num_bones, iterations=5,initial_bone=None):
    """
    Uses the k-means algorithm to initialize bone transformations.

    inputs: poses       |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose   |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones   Number of bones to initialize
            iterations  Number of iterations to run the k-means algorithm

    return: A |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked Rotation and Translation
              for each pose, for each bone.
            A |num_bones| x 3 matrix representing the translations of the rest bones.
    """
    num_verts = rest_pose.shape[0]
    num_poses = poses.shape[0]
    bone_transforms = np.empty((num_bones, num_poses, 4, 3))   # [(R, T) for for each pose] for each bone
                                                               # 3rd dim has 3 rows for R and 1 row for T            
    rest_bones_t = np.empty((num_bones, 3))                    # Translations for bones at rest pose
    rest_pose_corrected = np.empty((num_bones, num_verts, 3))  # Rest pose - mean of vertices attached to each bone

    # Use k-means to assign bones to vertices
    # whitened = whiten(rest_pose)
    # # codebook, _ = kmeans(whitened, num_bones)
    # codebook = np.array(initial_bone)
    # vert_assignments, _ = vq(whitened, codebook) # Bone assignment for each vertex (|num_verts| x 1) 将k-means分配的簇中的点的label分配
    
    vert_assignments = Cluster(initial_bone, rest_pose)

    # Compute initial random bone transformations
    for bone in range(num_bones):
        rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0) #骨骼中心点
        rest_pose_corrected[bone] = rest_pose - np.mean(rest_pose[vert_assignments == bone], axis=0)
        for pose in range(num_poses):
            bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone]) # 算出每个骨骼的刚性变换
    
    for it in range(iterations):
        # Re-assign bones to vertices using smallest reconstruction error from all poses
        constructed = np.empty((num_bones, num_poses, num_verts, 3)) # |num_bones| x |num_poses| x |num_verts| x 3
        for bone in range(num_bones):
            Rp = bone_transforms[bone,:,:3,:].dot((rest_pose - rest_bones_t[bone]).T).transpose((0, 2, 1)) # |num_poses| x |num_verts| x 3
            # R * p + T
            constructed[bone] = Rp + bone_transforms[bone, :, np.newaxis, 3, :]
        errs = np.linalg.norm(constructed - poses, axis=(1, 3)) # 对每个pose，计算每个骨骼带来的变换的误差的和，每个骨骼会对应一个RT变换，由此来计算误差的和，下次分配的时候，找出该点对应的误差最小的骨骼，进行二次分配
        vert_assignments = np.argmin(errs, axis=0)    
        
        ## Visualization of vertex assignments for bone 0 over iterations
        ## Make 5 copies of an example pose mesh and call them test0, test1...
        #for i in range(num_verts):
        #    if vert_assignments[i] == 0:
        #        pm.select('test{0}.vtx[{1}]'.format(it, i), add=True)
        #print(vert_assignments)

        # For each bone, for each pose, compute new transform using kabsch
        for bone in range(num_bones):
            rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0)
            rest_pose_corrected[bone] = rest_pose - np.mean(rest_pose[vert_assignments == bone], axis=0)
            for pose in range(num_poses):
                bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone])

    return bone_transforms, rest_bones_t


def update_weight_map(bone_transforms, rest_bones_t, poses, rest_pose, sparseness):
    """
    Update the bone-vertex weight map W by fixing bone transformations and using a least squares
    solver subject to non-negativity constraint, affinity constraint, and sparseness constraint.

    inputs: bone_transforms |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
            rest_bones_t    |num_bones| x 3 matrix representing the translations of the rest bones
            poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            sparseness      Maximum number of bones allowed to influence a particular vertex

    return: A |num_verts| x |num_bones| weight map representing the influence of the jth bone on the ith vertex
    """
    num_verts = rest_pose.shape[0]
    num_poses = poses.shape[0]
    num_bones = bone_transforms.shape[0]

    W = np.empty((num_verts, num_bones))

    for v in range(num_verts):
        # For every vertex, solve a least squares problem
        Rp = np.empty((num_bones, num_poses, 3))
        for bone in range(num_bones):
            Rp[bone] = bone_transforms[bone,:,:3,:].dot(rest_pose[v] - rest_bones_t[bone]) # |num_bones| x |num_poses| x 3
        # R * p + T
        Rp_T = Rp + bone_transforms[:, :, 3, :] # |num_bones| x |num_poses| x 3
        A = Rp_T.transpose((1, 2, 0)).reshape((3 * num_poses, num_bones)) # 3 * |num_poses| x |num_bones|
        b = poses[:, v, :].reshape(3 * num_poses) # 3 * |num_poses| x 1
        # A和b是最小二乘问题的两个变量

        # Bounds ensure non-negativity constraint and kind of affinity constraint
        # 直接调用函数解决有约束的最小二乘问题，并归一化权重
        w = lsq_linear(A, b, bounds=(0, 1), method='bvls').x  # |num_bones| x 1
        w /= np.sum(w) # Ensure that w sums to 1 (affinity constraint)

        # Remove |B| - |K| bone weights with the least "effect"·
        # 只获取前K个Bone weights
        effect = np.linalg.norm((A * w).reshape(num_poses, 3, num_bones), axis=1) # |num_poses| x |num_bones| # 这一行是算误差
        effect = np.sum(effect, axis=0) # |num_bones| x 1
        num_discarded = max(num_bones - sparseness, 0)
        effective = np.argpartition(effect, num_discarded)[num_discarded:] # |sparseness| x 1

        # Run least squares again, but only use the most effective bones
        # 再用最有用的bone再跑一次这个问题
        A_reduced = A[:, effective] # 3 * |num_oses| x |sparseness|
        w_reduced = lsq_linear(A_reduced, b, bounds=(0, 1), method='bvls').x # |sparseness| x 1
        w_reduced /= np.sum(w_reduced) # Ensure that w sums to 1 (affinity constraint)

        w_sparse = np.zeros(num_bones)
        w_sparse[effective] = w_reduced
        w_sparse /= np.sum(w_sparse) # Ensure that w_sparse sums to 1 (affinity constraint)

        W[v] = w_sparse

    return W


def update_bone_transforms(W, bone_transforms, rest_bones_t, poses, rest_pose):
    """
    Updates the bone transformations by fixing the bone-vertex weight map and minimizing an
    objective function individually for each pose and each bone.
    
    inputs: W               |num_verts| x |num_bones| matrix: bone-vertex weight map. Rows sum to 1, sparse.
            bone_transforms |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
            rest_bones_t    |num_bones| x 3 matrix representing the translations of the rest bones
            poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            
    return: |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
    """
    num_bones = W.shape[1]
    num_poses = poses.shape[0]
    num_verts = W.shape[0]
    
    for pose in range(num_poses):
        for bone in range(num_bones):
            # Represents the points in rest pose without this rest bone's translation
            p_corrected = rest_pose - rest_bones_t[bone] # |num_verts| x 3

            # Calculate q_i for all vertices by equation (6)
            constructed = np.empty((num_bones, num_verts, 3)) # |num_bones| x |num_verts| x 3
            for bone2 in range(num_bones):
                # can't use p_corrected before because we want to correct for every bone2 distinctly
                Rp = bone_transforms[bone2,pose,:3,:].dot((rest_pose - rest_bones_t[bone2]).T).T # |num_verts| x 3
                # R * p + T
                constructed[bone2] = Rp + bone_transforms[bone2, pose, 3, :]
            # w * (R * p + T)
            constructed = constructed.transpose((1, 0, 2)) * W[:, :, np.newaxis] # |num_verts| x |num_bones| x 3
            constructed = np.delete(constructed, bone, axis=1) # |num_verts| x |num_bones-1| x 3
            q = poses[pose] - np.sum(constructed, axis=1) # |num_verts| x 3

            # Calculate p_star, q_star, p_bar, and q_bar for all verts by equation (8)
            p_star = np.sum(np.square(W[:, bone, np.newaxis]) * p_corrected, axis=0) # |num_verts| x 3 => 3 x 1
            p_star /= np.sum(np.square(W[:, bone])) # 3 x 1
            
            q_star = np.sum(W[:, bone, np.newaxis] * q, axis=0) # |num_verts| x 3 => 3 x 1
            q_star /= np.sum(np.square(W[:, bone])) # 3 x 1
            p_bar = p_corrected - p_star # |num_verts| x 3
            q_bar = q - W[:, bone, np.newaxis] * q_star # |num_verts| x 3
            
            # Perform SVD by equation (9)
            P = (p_bar * W[:, bone, np.newaxis]).T # 3 x |num_verts|
            Q = q_bar.T # 3 x |num_verts|
            
            U, S, V = np.linalg.svd(np.matmul(P, Q.T))

            # Calculate rotation R and translation t by equation (10)
            R = U.dot(V).T # 3 x 3
            t = q_star - R.dot(p_star) # 3 x 1
            
            bone_transforms[bone, pose, :3, :] = R
            bone_transforms[bone, pose, 3, :] = t
    
    return bone_transforms


def SSDR(poses, rest_pose, num_bones, sparseness=4, max_iterations=20,initial_Bone=None):
    """
    Computes the Smooth Skinning Decomposition with Rigid bones
    
    inputs: poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones       number of bones to create
            sparseness      max number of bones influencing a single vertex
            
    return: An i x j matrix of bone-vertex weights, where i = # vertices and j = # bones
            A length-B list of (length-t lists of bone transformations [R_j | T_j] ), one list for each bone
            A list of bone translations for the bones at rest
    """
    start_time = time.time()
    print("SSDR: start initialize")
    bone_transforms, rest_bones_t = initialize_with_structure(poses, rest_pose, num_bones,initial_Bone)
    print("SSDR: finish initialize, start reconstruction")
    for iter in range(max_iterations):
        W = update_weight_map(bone_transforms, rest_bones_t, poses, rest_pose, sparseness)
        bone_transforms = update_bone_transforms(W, bone_transforms, rest_bones_t, poses, rest_pose)
        errors = reconstruct(rest_pose, bone_transforms, rest_bones_t, W) - poses
        print("iter:",iter,"Reconstruction error:", np.mean(np.linalg.norm(errors, axis=2)))

        if iter == max_iterations - 5:
            W, bone_transforms, rest_bones_t = optimize_bone_structure(W, bone_transforms, rest_bones_t, rest_pose, threshold=0.1, 
                          merge_distance_threshold=0.25, merge_transform_threshold=0.4)
    
    end_time = time.time()
    print("Done. Calculation took {0} seconds".format(end_time - start_time))
    errors = reconstruct(rest_pose, bone_transforms, rest_bones_t, W) - poses
    print("Avg reconstruction error:", np.mean(np.linalg.norm(errors, axis=2)))

    return W, bone_transforms, rest_bones_t


def SSDR_w_InitBone(poses, rest_pose, num_bones, sparseness=4, max_iterations=20,initial_Bone=None):
    """
    Computes the Smooth Skinning Decomposition with Rigid bones
    
    inputs: poses           |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            num_bones       number of bones to create
            sparseness      max number of bones influencing a single vertex
            
    return: An i x j matrix of bone-vertex weights, where i = # vertices and j = # bones
            A length-B list of (length-t lists of bone transformations [R_j | T_j] ), one list for each bone
            A list of bone translations for the bones at rest
    """
    start_time = time.time()
    print("SSDR: start initialize")
    if initial_Bone is not None:
        num_bones = initial_Bone.shape[0]
    bone_transforms, rest_bones_t = initialize_w_Bone(poses, rest_pose, num_bones,initial_Bone)
    print("SSDR: finish initialize, start reconstruction")
    for iter in range(max_iterations):
        W = update_weight_map(bone_transforms, rest_bones_t, poses, rest_pose, sparseness)
        bone_transforms = update_bone_transforms(W, bone_transforms, rest_bones_t, poses, rest_pose)
        errors = reconstruct(rest_pose, bone_transforms, rest_bones_t, W) - poses
        print("iter:",iter,"Reconstruction error:", np.mean(np.linalg.norm(errors, axis=2)))
    
    end_time = time.time()
    print("Done. Calculation took {0} seconds".format(end_time - start_time))
    errors = reconstruct(rest_pose, bone_transforms, rest_bones_t, W) - poses
    print("Avg reconstruction error:", np.mean(np.linalg.norm(errors, axis=2)))

    return W, bone_transforms, rest_bones_t



def reconstruct(rest_pose, bone_transforms, rest_bones_t, W):
    """
    Computes the skinned vertex positions on some poses given bone transforms and weights.

    inputs : rest_pose       |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
             bone_transforms |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked 
                                Rotation and Translation for each pose, for each bone.
             rest_bones_t    |num_bones| x 3 matrix representing the translations of the rest bones
             W               |num_verts| x |num_bones| matrix: bone-vertex weight map. Rows sum to 1, sparse.

    return: |num_poses| x |num_verts| x 3 Vertex positions for all poses: sum{bones} (w * (R @ p + T)) 
    """
    # Points in rest pose without rest bone translations
    p_corrected = rest_pose[np.newaxis, :, :] - rest_bones_t[:, np.newaxis, :] # |num_bones| x |num_verts| x 3
    constructions = np.einsum('bijk,blk->bilj', bone_transforms[:, :, :3, :], p_corrected) # |num_bones| x |num_poses| x |num_verts| x 3
    constructions += bone_transforms[:, :, np.newaxis, 3, :] # |num_bones| x |num_poses| x |num_verts| x 3
    constructions *= (W.T)[:, np.newaxis, :, np.newaxis] # |num_bones| x |num_poses| x |num_verts| x 3
    return np.sum(constructions, axis=0)

def load_mesh_sequence_from_files(mesh_folder):
    # 获取文件夹中所有.obj文件，并按名称排序
    mesh_files = sorted([f for f in os.listdir(mesh_folder) if f.endswith('.obj')])
    # print("Found mesh files:", mesh_files)

    num_poses = len(mesh_files) - 1
    print("Number of poses:", num_poses)

    rest_pose = None
    poses = []

    # 从文件名中提取数字进行排序
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
        

    from tqdm import tqdm
    for i, mesh_file in tqdm(enumerate(sorted(mesh_files, key=extract_number)), desc="Loading meshes", total=len(mesh_files)):
        mesh_path = os.path.join(mesh_folder, mesh_file)
        # 使用trimesh加载mesh文件
        mesh = trimesh.load(mesh_path)

        # 获取顶点信息并确保类型为float64
        points = np.array(mesh.vertices, dtype=np.float64)
        # print(f"Mesh {i} vertices shape:", points.shape)
        # print(f"Mesh {i} vertices dtype:", points.dtype)

        # 如果是最后一个mesh文件，设为rest_pose
        if i == num_poses:
            rest_pose = points
        else:
            poses.append(points)

    poses = np.array(poses, dtype=np.float64)
    
    # 打印最终数据的信息
    print("\nFinal data shapes:")
    print("Poses shape:", poses.shape)
    print("Poses dtype:", poses.dtype)

    return rest_pose, poses


def remove_empty_bones(vert_assignments, init_bones, existing_empty_bones=None):
    """
    检测并移除没有分配到顶点的骨骼
    
    参数:
        vert_assignments: 顶点分配数组，表示每个顶点分配到的骨骼索引
        init_bones: 骨骼点坐标数组
        existing_empty_bones: 已经被移除的骨骼索引列表（相对于原始骨骼集），默认为None
        
    返回:
        init_bones: 更新后的骨骼点坐标数组（已移除空骨骼）
        current_empty_bones: 当前检测到的空骨骼索引列表（相对于传入的init_bones）
        all_empty_bones: 所有被移除的骨骼索引列表（相对于原始骨骼集）
    """
    num_bones = init_bones.shape[0]
    
    # 初始化空骨骼列表
    current_empty_bones = []  # 相对于当前init_bones的索引
    all_empty_bones = [] if existing_empty_bones is None else existing_empty_bones.copy()  # 相对于原始骨骼集的索引
    
    # 检测没有分配到顶点的骨骼
    print("Checking for empty bones...")
    for bone_idx in range(num_bones):
        count = np.sum(vert_assignments == bone_idx)
        if count == 0:
            current_empty_bones.append(bone_idx)
    
    # 如果没有空骨骼，直接返回
    if not current_empty_bones:
        return init_bones, current_empty_bones, all_empty_bones
    
    # 将当前空骨骼的相对索引映射到绝对索引（相对于原始骨骼集）
    if all_empty_bones:
        # 创建从当前索引到原始索引的映射关系
        # 这个映射考虑了已经移除的骨骼导致的索引偏移
        original_indices = []
        current_idx = 0  # 当前骨骼索引
        original_idx = 0  # 原始骨骼索引
        
        # 遍历所有可能的原始索引
        while original_idx < num_bones + len(all_empty_bones):
            if original_idx in all_empty_bones:
                # 跳过已经移除的骨骼索引
                original_idx += 1
                continue
                
            if current_idx in current_empty_bones:
                # 当前检测到的空骨骼，记录其对应的原始索引
                original_indices.append(original_idx)
            
            current_idx += 1
            original_idx += 1
            
            # 如果已经处理完所有当前骨骼，则退出
            if current_idx >= num_bones:
                break
        
        # 添加新的空骨骼（原始索引）到all_empty_bones
        all_empty_bones.extend(original_indices)
    else:
        # 第一次检测空骨骼，直接添加当前索引到all_empty_bones
        all_empty_bones.extend(current_empty_bones)
    
    # 从骨骼数组中移除空骨骼
    init_bones = np.delete(init_bones, current_empty_bones, axis=0)
    print(f"Removed {len(current_empty_bones)} empty bones (current indices): {current_empty_bones}")
    print(f"All removed bones (original indices): {sorted(all_empty_bones)}")
    
    return init_bones, current_empty_bones, all_empty_bones

def init_bone_label(rest_pose, poses, init_bones, iterations=5):
# def initialize_w_Bone(poses, rest_pose, num_bones, iterations=5,initial_bone=None):
    """
    Uses the k-means algorithm to initialize bone transformations.

    inputs: poses       |num_poses| x |num_verts| x 3 matrix representing coordinates of vertices of each pose
            rest_pose   |num_verts| x 3 numpy matrix representing the coordinates of vertices in rest pose
            init_bones   |num_bones| x 3 numpy matrix representing the coordinates of vertices in init bones
            iterations  Number of iterations to run the k-means algorithm

    return: A |num_bones| x |num_poses| x 4 x 3 matrix representing the stacked Rotation and Translation
              for each pose, for each bone.
            A |num_bones| x 3 matrix representing the translations of the rest bones.
    """
    num_verts = rest_pose.shape[0]
    num_poses = poses.shape[0]
    num_bones = init_bones.shape[0]
    bone_transforms = np.empty((num_bones, num_poses, 4, 3))   # [(R, T) for for each pose] for each bone
                                                               # 3rd dim has 3 rows for R and 1 row for T            
    rest_bones_t = np.empty((num_bones, 3))                    # Translations for bones at rest pose
    rest_pose_corrected = np.empty((num_bones, num_verts, 3))  # Rest pose - mean of vertices attached to each bone

    # Use k-means to assign bones to vertices
    # whitened = whiten(rest_pose)
    # # codebook, _ = kmeans(whitened, num_bones)
    # codebook = np.array(initial_bone)
    # vert_assignments, _ = vq(whitened, codebook) # Bone assignment for each vertex (|num_verts| x 1) 将k-means分配的簇中的点的label分配
    
    empty_bones = []  # 记录所有被移除的空骨骼
    
    while True:
        vert_assignments = Cluster(init_bones, rest_pose)
        print("Vertex assignments statistics:")
        
        # 检测并移除空骨骼
        init_bones, current_empty_bones, empty_bones = remove_empty_bones(vert_assignments, init_bones, empty_bones)
        num_bones = init_bones.shape[0]
        
        if not current_empty_bones:
            break
    
    print(f"removed bones from init: {empty_bones}")

    # Compute initial random bone transformations
    for bone in tqdm(range(num_bones), desc="Computing initial bone transformations"):
        rest_bones_t[bone] = np.mean(rest_pose[vert_assignments == bone], axis=0) #骨骼中心点
        # rest_bones_t[bone] = init_bones[bone]
        rest_pose_corrected[bone] = rest_pose - rest_bones_t[bone]
        for pose in range(num_poses):
            bone_transforms[bone, pose] = kabsch(rest_pose_corrected[bone, vert_assignments == bone], poses[pose, vert_assignments == bone]) # 算出每个骨骼的刚性变换
    
    for it in tqdm(range(iterations), desc="re-assign bones to vertices"):
        # 初始化最小误差数组和顶点分配数组
        min_errors = np.full(num_verts, np.inf)
        vert_assignments = np.zeros(num_verts, dtype=np.int32)
        
        # 逐骨骼计算误差
        for bone in tqdm(range(num_bones), desc=f"Computing errors for iteration {it+1}"):
            # 计算当前骨骼的变换
            Rp = bone_transforms[bone,:,:3,:].dot((rest_pose - rest_bones_t[bone]).T).transpose((0, 2, 1))
            current_transformed = Rp + bone_transforms[bone, :, np.newaxis, 3, :]
            
            # 逐顶点计算误差，避免大型数组操作
            # 计算当前骨骼对每个顶点的误差
            errors = np.mean(np.sum((current_transformed - poses) ** 2, axis=2), axis=0)
            
            # 更新最小误差和对应的骨骼分配
            update_mask = errors < min_errors
            min_errors[update_mask] = errors[update_mask]
            vert_assignments[update_mask] = bone
            
            # 清理临时变量
            del Rp, current_transformed, errors
        
        # 检测并移除空骨骼
        init_bones_before = init_bones.copy()
        init_bones, current_empty_bones, empty_bones = remove_empty_bones(vert_assignments, init_bones, empty_bones)
        
        # 如果有新的空骨骼被移除，需要更新相关数据结构
        if len(current_empty_bones) > 0:
            print(f"发现并移除了新的空骨骼，需要更新数据结构")
            # 需要重建骨骼变换矩阵等数据结构，移除空骨骼对应的数据
            num_bones = init_bones.shape[0]  # 更新后的骨骼数量
            
            # 创建新的数据结构
            new_bone_transforms = np.empty((num_bones, num_poses, 4, 3))
            new_rest_bones_t = np.empty((num_bones, 3))
            
            # 更新索引映射
            old_to_new = np.full(len(init_bones_before), -1)
            new_idx = 0
            for old_idx in range(len(init_bones_before)):
                if old_idx not in current_empty_bones:
                    old_to_new[old_idx] = new_idx
                    # 复制骨骼数据到新数组
                    if new_idx < num_bones:  # 安全检查
                        new_bone_transforms[new_idx] = bone_transforms[old_idx]
                        new_rest_bones_t[new_idx] = rest_bones_t[old_idx]
                    new_idx += 1
            
            # 替换原始数组
            bone_transforms = new_bone_transforms
            rest_bones_t = new_rest_bones_t
            
            # 重新映射顶点分配
            for i in range(len(vert_assignments)):
                old_bone = vert_assignments[i]
                if old_bone in current_empty_bones:
                    # 如果顶点分配给了被移除的骨骼，需要重新分配
                    # 这应该不会发生，因为我们移除的就是没有顶点的骨骼
                    print(f"警告：顶点 {i} 分配给了被移除的骨骼 {old_bone}")
                else:
                    vert_assignments[i] = old_to_new[old_bone]
        
        # 更新骨骼变换
        for bone in range(num_bones):
            bone_verts = vert_assignments == bone
            if not np.any(bone_verts):
                continue
                
            rest_bones_t[bone] = np.mean(rest_pose[bone_verts], axis=0)
            rest_pose_corrected = rest_pose[bone_verts] - rest_bones_t[bone]
            
            for pose in range(num_poses):
                bone_transforms[bone, pose] = kabsch(
                    rest_pose_corrected,
                    poses[pose, bone_verts]
                )

    return bone_transforms, vert_assignments, empty_bones


# Get numpy vertex arrays from selected objects. Rest pose is most recently selected.
# selectionLs = om.MGlobal.getActiveSelectionList()
# num_poses = selectionLs.length() - 1
# rest_pose = np.array(om.MFnMesh(selectionLs.getDagPath(num_poses)).getPoints(om.MSpace.kWorld))[:, :3]
# poses = np.array([om.MFnMesh(selectionLs.getDagPath(i)).getPoints(om.MSpace.kWorld) for i in range(num_poses)])[:, :, :3]

if __name__ == "__main__":
    mesh_folder = "/home/wjx/research/data/dg-mesh/bird/mesh_gt"
    rest_pose, poses = load_mesh_sequence_from_files(mesh_folder)
    W, bone_transforms, rest_bones_t = SSDR(poses, rest_pose, 50)
    # 使用np和字典保存结果
    result = {
        "W": W,
        "bone_transforms": bone_transforms,
        "rest_bones_t": rest_bones_t,
        "rest_pose": rest_pose
    }
    np.save("result.npy", result)


# you can also design a pre-define bone to run this code.