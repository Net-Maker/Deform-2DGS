import numpy as np
from scipy import sparse
import pyssdr
import trimesh
import open3d as o3d
import numpy as np
import matplotlib.colors as mcolors
import os
import pyvista as pv
from pyvista import examples
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

def process_transformations(transformations):
    """
    处理初始化的变换矩阵，确保它们是4x4的齐次矩阵
    
    参数:
        transformations: numpy数组，形状为(B, T, 4, 3)的变换矩阵

    返回:
        transformations: numpy数组，形状为(B, T, 4, 4)的齐次变换矩阵
    """
    if transformations.shape[-1] == 3:
        # 创建最后一列 [0, 0, 0, 1]
        last_column = np.zeros((transformations.shape[0], transformations.shape[1], 4, 1))
        last_column[:, :, 3, 0] = 1  # 设置最后一行的最后一个元素为1
        
        # 将最后一列添加到原始矩阵
        transformations = np.concatenate([transformations, last_column], axis=-1)
    
    return transformations


def stand_transform2Dembones_transform(transform):
    """
    将标准B,T,4,4变换矩阵转换为DemBones的变换矩阵4*T,4*B
    """
    return transform.transpose((1, 2, 0, 3)).reshape((4*transform.shape[1], 4*transform.shape[0]))

def generate_bone_position_sequence(skeleton_points, bone_transforms):
    """
    根据骨骼点和变换矩阵生成骨骼位置序列
    
    参数:
        skeleton_points: numpy数组，形状为(B, 3)，表示B个骨骼点的初始位置
        bone_transforms: numpy数组，形状为(B, T, 4, 4)，表示T个时间步上B个骨骼点的变换矩阵
        
    返回:
        position_sequence: numpy数组，形状为(T, B, 3)，表示T个时间步上B个骨骼点的位置
    """
    if bone_transforms.shape[-1] == 3:
        bone_transforms = process_transformations(bone_transforms)
    
    # 获取维度信息并检查一致性
    B_transforms = bone_transforms.shape[0]  # 变换矩阵中的骨骼数量
    B_points = skeleton_points.shape[0]      # 骨骼点数量
    
    if B_transforms != B_points:
        print(f"警告：骨骼点数量({B_points})与变换矩阵数量({B_transforms})不匹配")
        # 使用较小的数量
        B = min(B_points, B_transforms)
        skeleton_points = skeleton_points[:B]
    else:
        B = B_transforms
        
    T = bone_transforms.shape[1]  # 时间步数
    
    # 初始化结果数组 (T, B, 3)
    position_sequence = np.zeros((T, B, 3))
    
    # 对每个时间步
    for t in range(T):
        # 对每个骨骼点
        for b in range(B):
            # 获取当前变换矩阵
            transform = bone_transforms[b, t]  # (4, 4)
            R = transform[:3, :3]
            T = transform[:3, 3]
            
            skel_point = skeleton_points[b]
            
            # 应用变换矩阵
            transformed_point = np.dot(R, skel_point) + T
            
            # 将结果从齐次坐标转回3D坐标
            position_sequence[t, b] = transformed_point
    
    return position_sequence

def generate_bone_position_sequence_from_ssdr(standard_bone_transforms, weights, rest_pose):
    """
    根据标准变换矩阵和蒙皮权重获取每个骨骼的代表点序列
    
    参数:
        standard_bone_transforms: numpy数组，形状为(T, B, 4, 4)，标准的骨骼变换矩阵
        weights: numpy数组，形状为(B, N) 或 (N, B)，蒙皮权重矩阵
        rest_pose: numpy数组，形状为(N, 3)，静止姿态下的点位置
        
    返回:
        bone_position_sequence: numpy数组，形状为(T, B, 3)，每个时间步骨骼代表点的位置
    """
    # 确保权重矩阵的形状为(N, B)
    if weights.shape[0] < weights.shape[1]:
        weights = weights.T
    
    N, B = weights.shape  # N个点，B个骨骼
    T = standard_bone_transforms.shape[0]  # T个时间步

    # 为每个骨骼找到权重最大的点
    max_weight_indices = np.argmax(weights, axis=0)  # 形状为(B,)
    max_weight_points = rest_pose[max_weight_indices]  # 形状为(1, B, 3)
    max_weight_points = max_weight_points.reshape((B, 3))
    #print("max_weight_points", max_weight_points.shape)
    
    #print(f"每个骨骼的代表点索引: {max_weight_indices}")
    #print(f"每个骨骼的最大权重值: {np.max(weights, axis=0)}")
    
    # 初始化结果数组
    bone_position_sequence = np.zeros((T, B, 3))
    #print("standard_bone_transforms.shape", standard_bone_transforms.shape)
    # 对每个时间步和每个骨骼计算变换后的位置
    for t in range(T):
        for b in range(B):
            # 获取该骨骼在时间t的变换矩阵
            transform_matrix = standard_bone_transforms[t, b]  # 4x4矩阵
            
            # 获取该骨骼的代表点（静止姿态）
            rest_point = max_weight_points[b]  # 3D点
            #print("rest_point", rest_point.shape)
            
            # 将3D点转换为齐次坐标
            rest_point_homo = np.append(rest_point, 1.0)  # 形状为(4,)
            #print("rest_point_homo", rest_point_homo.shape)
            # 应用变换矩阵
            transformed_point_homo = transform_matrix @ rest_point_homo  # 4x4 @ 4x1 = 4x1
            
            # 转换回3D坐标
            bone_position_sequence[t, b] = transformed_point_homo[:3]
    
    return bone_position_sequence

def get_bone_representative_points_and_weights(weights, rest_pose):
    """
    获取每个骨骼的代表点及其权重信息
    
    参数:
        weights: numpy数组，形状为(B, N) 或 (N, B)，蒙皮权重矩阵
        rest_pose: numpy数组，形状为(N, 3)，静止姿态下的点位置
        
    返回:
        representative_points: numpy数组，形状为(B, 3)，每个骨骼的代表点
        max_weights: numpy数组，形状为(B,)，每个骨骼的最大权重值
        point_indices: numpy数组，形状为(B,)，每个骨骼代表点的索引
    """
    # 确保权重矩阵的形状为(N, B)
    if weights.shape[0] < weights.shape[1]:
        weights = weights.T
    
    N, B = weights.shape
    
    # 为每个骨骼找到权重最大的点
    point_indices = np.argmax(weights, axis=0)  # 形状为(B,)
    max_weights = np.max(weights, axis=0)  # 形状为(B,)
    representative_points = rest_pose[point_indices]  # 形状为(B, 3)
    
    return representative_points, max_weights, point_indices

def apply_transforms_to_points(points, transforms):
    """
    将变换矩阵应用到点集上
    
    参数:
        points: numpy数组，形状为(B, 3)，要变换的点
        transforms: numpy数组，形状为(T, B, 4, 4)，变换矩阵序列
        
    返回:
        transformed_points: numpy数组，形状为(T, B, 3)，变换后的点序列
    """
    T, B = transforms.shape[:2]
    transformed_points = np.zeros((T, B, 3))
    
    for t in range(T):
        for b in range(B):
            # 将3D点转换为齐次坐标
            point_homo = np.append(points[b], 1.0)
            
            # 应用变换矩阵
            transformed_homo = transforms[t, b] @ point_homo
            
            # 转换回3D坐标
            transformed_points[t, b] = transformed_homo[:3]
    
    return transformed_points
class DemBonesCalculator:
    def __init__(self):
        """初始化SSDR计算器"""
        self.weights = None
        self.transformations = None
        self.rmse = None
        self.reconstruction = None
        self.skeleton = None
        self.rest_pose = None
        self.poses = None
        

    def get_max_weights_bone_seq(self, weights, points_sequence):
        """获取每个骨骼对应的最大权重点的运动序列"""
        bone_positions_sequence = []
        for frame_points in points_sequence:
            bone_positions = []
            for bone_idx in range(weights.shape[1]):
                max_weight_vertex = np.argmax(weights[:, bone_idx])
                bone_positions.append(frame_points[max_weight_vertex])
        return np.array(bone_positions_sequence)


    def compute_pure_ssdr(self, poses, rest_pose, num_bones, faces=None, bone_transforms=None, tolerance=1e-3, patience=3, max_iters=100, nnz=4, weights_smooth=0):
        self.poses = poses
        self.rest_pose = rest_pose
        self.num_bones = num_bones

        ssdr = pyssdr.MyDemBones()

        # 设置参数
        ssdr.tolerance = tolerance
        ssdr.patience = patience
        ssdr.nIters = max_iters
        ssdr.nnz = nnz
        ssdr.weightsSmooth = weights_smooth
        ssdr.bindUpdate = 2

        # 重塑轨迹数据
        T, N, D = poses.shape
        trajectory = poses.transpose((0, 2, 1))
        trajectory = trajectory.reshape((D*T, N))

        # 转换面片数据
        face_list = [] if faces is None else [list(face) for face in faces]
        
        # 加载数据
        ssdr.load_data(trajectory, face_list)
        ssdr.nB = num_bones
        
        self.weights, self.transformations, self.rmse = ssdr.run_ssdr(num_bones, "here.fbx")
        print("rmse", self.rmse)

        new_num_bones = ssdr.nB
        self.transformations = self.transformations.reshape((T, 4, new_num_bones,4)).transpose((0, 2, 1, 3)) # T,B,4,4

        # 计算重建结果
        reconstruction = ssdr.compute_reconstruction(list(range(N)))
        self.reconstruction = np.zeros((T, N, 3))
        self.reconstruction[:, :, 0] = reconstruction[list(range(0, 3*T, 3))]
        self.reconstruction[:, :, 1] = reconstruction[list(range(1, 3*T, 3))]
        self.reconstruction[:, :, 2] = reconstruction[list(range(2, 3*T, 3))]
        
        return self.weights, self.transformations, self.rmse, self.reconstruction
        


    def compute_ssdr_with_init(self, poses, rest_pose, skeleton, faces=None, vert_assignments=None,bone_transforms=None, num_bones=None, tolerance=1e-3, 
                    patience=3, max_iters=100, nnz=4, weights_smooth=0):
        """
        计算SSDR分解
        input:
            poses: numpy数组，形状为(T, N, 3)，表示T个时间步上N个点的运动序列
            rest_pose: numpy数组，形状为(N, 3)，表示N个点的静止位置
            skeleton: numpy数组，形状为(B, 3)，表示B个骨骼点的位置
            faces: list，表示面片列表
            vert_assignments: numpy数组，形状为(N,)，表示N个点的骨骼分配结果
            bone_transforms: numpy数组，形状为(B, T, 4, 4)，表示T个时间步上B个骨骼点的变换矩阵
        """
        self.poses = poses
        self.rest_pose = rest_pose
        self.skeleton = skeleton
        
        # 初始化SSDR
        ssdr = pyssdr.MyDemBones()
        
        # 设置参数
        ssdr.tolerance = tolerance
        ssdr.patience = patience
        ssdr.nIters = max_iters
        ssdr.nnz = nnz
        ssdr.weightsSmooth = weights_smooth
        ssdr.bindUpdate = 2
        ssdr.m = stand_transform2Dembones_transform(bone_transforms) # 4*T,4*B, 这是DemBones里面的数据存储格式
        
        # 重塑轨迹数据
        T, N, D = poses.shape
        trajectory = poses.transpose((0, 2, 1))
        trajectory = trajectory.reshape((D*T, N))
        
        # 转换面片数据
        face_list = [] if faces is None else [list(face) for face in faces]
        
        # 加载数据
        ssdr.load_data(trajectory, face_list)
        
        # 设置骨骼数量
        if num_bones is None:
            num_bones = skeleton.shape[0]
            
        # 初始化标签
        if skeleton is not None:
            if vert_assignments is not None:
                # 如果提供了骨骼分配结果，直接使用
                ssdr.label = vert_assignments
            else:
                # 否则使用最近邻方法计算标签
                from scipy.spatial import cKDTree
                tree = cKDTree(skeleton)
                _, indices = tree.query(rest_pose)
                ssdr.label = indices
            
            ssdr.nB = num_bones
            ssdr.labelToWeights()
            ssdr.computeTransFromLabel()
        
        # 运行SSDR
        self.weights, self.transformations, self.rmse = ssdr.run_ssdr(num_bones, "here.fbx")

        self.transformations = self.transformations.reshape((T, 4, num_bones,4)).transpose((0, 2, 1, 3)) # T,B,4,4

        
        # 计算重建结果
        reconstruction = ssdr.compute_reconstruction(list(range(N)))
        self.reconstruction = np.zeros((T, N, 3))
        self.reconstruction[:, :, 0] = reconstruction[list(range(0, 3*T, 3))]
        self.reconstruction[:, :, 1] = reconstruction[list(range(1, 3*T, 3))]
        self.reconstruction[:, :, 2] = reconstruction[list(range(2, 3*T, 3))]
        
        return self.weights, self.transformations, self.rmse, self.reconstruction

    def visualize_4d_skinning(self, points_sequence=None, weights=None, bone_positions_sequence=None, 
                            fps=30, output_path=None, loop=True, bone_size=0.05):
        """
        使用Open3D可视化4D蒙皮动画
        
        参数:
            points_sequence: 点云序列，形状为(T, N, 3)
            weights: 权重矩阵，形状为(N, B)或(B, N)
            bone_positions_sequence: 骨骼位置序列，形状为(T, B, 3)
            fps: 帧率
            output_path: 输出路径（如果需要保存视频）
            loop: 是否循环播放
            bone_size: 骨骼球体大小
        """
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D库未安装或导入失败，无法进行可视化。请安装Open3D: pip install open3d")
            return
            
        # 使用类内部数据或传入的数据
        points_sequence = points_sequence if points_sequence is not None else self.poses
        weights = weights if weights is not None else self.weights
        bone_positions_sequence = bone_positions_sequence if bone_positions_sequence is not None else generate_bone_position_sequence(self.skeleton, self.transformations)
        
        # 确保权重是密集矩阵
        if sparse.issparse(weights):
            weights = weights.toarray()
        
        # 检查权重形状并转置如果需要
        if weights.shape[0] == len(bone_positions_sequence[0]) and weights.shape[1] == len(points_sequence[0]):
            weights = weights.T
        
        # 生成骨骼颜色
        num_bones = weights.shape[1]
        colors = self.generate_distinct_colors(num_bones)
        
        # 计算每个点的混合颜色
        blended_colors = np.zeros((len(points_sequence[0]), 3))
        for i in range(num_bones):
            weight = weights[:, i].reshape(-1, 1)
            color = np.array(colors[i]).reshape(1, 3)
            blended_colors += weight * color
        
        # 确保RGB值在有效范围内
        blended_colors = np.clip(blended_colors, 0, 1)
        
        # 创建可视化窗口
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
        except Exception as e:
            print(f"无法创建Open3D可视化窗口: {e}")
            print("请确保您的系统支持Open3D的可视化功能。")
            return
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_sequence[0])
        pcd.colors = o3d.utility.Vector3dVector(blended_colors)
        vis.add_geometry(pcd)
        
        # 创建骨骼球体
        bone_spheres = []
        for i, pos in enumerate(bone_positions_sequence[0]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=bone_size)
            sphere.translate(pos)
            sphere.paint_uniform_color(colors[i])
            sphere.compute_vertex_normals()
            vis.add_geometry(sphere)
            bone_spheres.append(sphere)
        
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        # 如果需要保存视频
        if output_path:
            try:
                import cv2
                import os
                
                width = 1920
                height = 1080
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for i in tqdm(range(len(points_sequence))):
                    # 更新点云位置
                    pcd.points = o3d.utility.Vector3dVector(points_sequence[i])
                    
                    # 更新骨骼位置
                    for j, (sphere, pos) in enumerate(zip(bone_spheres, bone_positions_sequence[i])):
                        # 计算位移向量
                        current_pos = np.asarray(sphere.vertices)[0]
                        translation = pos - current_pos
                        sphere.translate(translation)
                        sphere.compute_vertex_normals()
                    
                    # 更新可视化
                    vis.update_geometry(pcd)
                    for sphere in bone_spheres:
                        vis.update_geometry(sphere)
                    vis.poll_events()
                    vis.update_renderer()
                    
                    # 捕获帧并保存
                    img = np.asarray(vis.capture_screen_float_buffer())
                    img = (img * 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out.write(img)
                
                out.release()
            except ImportError:
                print("未安装OpenCV库，无法保存视频。请安装OpenCV: pip install opencv-python")
                # 回退到交互式显示
                output_path = None
            except Exception as e:
                print(f"保存视频时出错: {e}")
                # 回退到交互式显示
                output_path = None
        
        if not output_path:
            # 交互式显示
            play_state = {'playing': True, 'frame_index': 0}
            last_update_time = time.time()
            
            try:
                while True:
                    current_time = time.time()
                    if current_time - last_update_time >= 1.0/fps and play_state['playing']:
                        i = play_state['frame_index']
                        
                        # 更新点云位置
                        pcd.points = o3d.utility.Vector3dVector(points_sequence[i])
                        
                        # 更新骨骼位置
                        for j, (sphere, pos) in enumerate(zip(bone_spheres, bone_positions_sequence[i])):
                            current_pos = np.asarray(sphere.vertices)[0]
                            translation = pos - current_pos
                            sphere.translate(translation)
                            sphere.compute_vertex_normals()
                        
                        # 更新可视化
                        vis.update_geometry(pcd)
                        for sphere in bone_spheres:
                            vis.update_geometry(sphere)
                        
                        # 更新帧索引
                        play_state['frame_index'] = (i + 1) % len(points_sequence) if loop else min(i + 1, len(points_sequence) - 1)
                        last_update_time = current_time
                    
                    # 处理事件和渲染
                    if not vis.poll_events():
                        break
                    vis.update_renderer()
            except KeyboardInterrupt:
                print("用户中断了可视化。")
            except Exception as e:
                print(f"可视化过程中出错: {e}")
        
        # 关闭可视化窗口
        try:
            vis.destroy_window()
        except:
            pass

    def visualize_skinning_weights(self, points=None, weights=None, bone_positions=None, 
                                 alpha=0.3, bone_size=0.05, save_path=None):
        """可视化蒙皮权重"""
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D库未安装或导入失败，无法进行可视化。请安装Open3D: pip install open3d")
            return
        
        # 使用类内部数据或传入的数据
        points = points if points is not None else self.rest_pose
        weights = weights if weights is not None else self.weights
        bone_positions = bone_positions if bone_positions is not None else self.skeleton
        
        # Convert sparse weights to dense if needed
        if sparse.issparse(weights):
            weights = weights.toarray()
        
        # Check weights shape and transpose if needed
        if weights.shape[0] == len(bone_positions) and weights.shape[1] == len(points):
            # Shape is (num_bones, N), transpose to (N, num_bones)
            weights = weights.T
        
        # Create color map for bones
        num_bones = weights.shape[1]
        colors = self.generate_distinct_colors(num_bones)
        
        # Create point cloud with blended colors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Calculate blended colors for each point
        blended_colors = np.zeros((len(points), 3))  # RGB only
        for i in range(num_bones):
            weight = weights[:, i].reshape(-1, 1)  # Ensure column vector (N, 1)
            color = np.array(colors[i]).reshape(1, 3)  # Ensure row vector (1, 3)
            blended_colors += weight * color  # Broadcasting (N, 1) * (1, 3) -> (N, 3)
        
        # Ensure RGB values are in valid range [0, 1]
        blended_colors = np.clip(blended_colors, 0, 1)
        
        pcd.colors = o3d.utility.Vector3dVector(blended_colors)

        # Create bone spheres
        bone_geoms = []
        for i, pos in enumerate(bone_positions):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=bone_size)
            sphere.translate(pos)
            sphere.paint_uniform_color(colors[i])  # Use RGB for mesh
            sphere.compute_vertex_normals()
            bone_geoms.append(sphere)

        # If save_path is provided, use offscreen rendering
        if save_path:
            try:
                # Use matplotlib to save a simple visualization
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot points
                point_array = np.asarray(pcd.points)
                color_array = np.asarray(pcd.colors)
                ax.scatter(point_array[:, 0], point_array[:, 1], point_array[:, 2], 
                          c=color_array, s=1, alpha=alpha)
                
                # Plot bones
                for i, pos in enumerate(bone_positions):
                    ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                              c=[colors[i]], s=100, alpha=1.0)
                
                # Set equal aspect ratio
                ax.set_box_aspect([1, 1, 1])
                
                # Save figure
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Visualization saved to {save_path}")
                return
            except Exception as e:
                print(f"保存可视化图像时出错: {e}")
                print("尝试使用交互式可视化...")
        
        # Otherwise try interactive visualization
        try:
            # Create visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            
            # Add geometries
            vis.add_geometry(pcd)
            for geom in bone_geoms:
                vis.add_geometry(geom)

            # Set render options
            opt = vis.get_render_option()
            opt.point_size = 3.0
            opt.background_color = np.array([0.1, 0.1, 0.1])
            
            # Run visualization
            vis.run()
            vis.destroy_window()
        except Exception as e:
            print(f"交互式可视化失败: {e}")
            print("请尝试使用save_path参数以保存可视化为图像，或检查Open3D安装。")

    @staticmethod
    def generate_distinct_colors(n):
        """生成不同的颜色"""
        if n <= 10:
            # For small number of bones, use a predefined color palette
            # These colors are chosen to be visually distinct
            base_colors = [
                (0.9, 0.1, 0.1),  # Red
                (0.1, 0.6, 0.9),  # Blue
                (0.1, 0.9, 0.1),  # Green
                (0.9, 0.6, 0.1),  # Orange
                (0.6, 0.1, 0.9),  # Purple
                (0.9, 0.9, 0.1),  # Yellow
                (0.1, 0.9, 0.6),  # Teal
                (0.9, 0.1, 0.6),  # Pink
                (0.5, 0.5, 0.5),  # Gray
                (0.7, 0.3, 0.0),  # Brown
            ]
            return base_colors[:n]
        
        # For larger sets, use the golden ratio to space hues evenly
        colors = []
        golden_ratio_conjugate = 0.618033988749895
        h = 0.1  # Starting hue
        
        for i in range(n):
            # Generate color with varying saturation and value to increase contrast
            s = 0.7 + 0.3 * ((i % 3) / 2.0)  # Vary saturation
            v = 0.8 + 0.2 * ((i % 2) / 1.0)  # Vary value/brightness
            
            rgb = mcolors.hsv_to_rgb((h, s, v))
            colors.append(rgb)
            
            # Use golden ratio to get next hue - this creates maximally distinct colors
            h = (h + golden_ratio_conjugate) % 1.0
        
        return colors

    def visualize_adjacency_matrix(self, adjacency_matrix, skeleton_points=None, title="骨骼邻接矩阵可视化"):
        """
        可视化邻接矩阵和骨骼连接图
        
        Args:
            adjacency_matrix: 邻接矩阵
            skeleton_points: 骨骼点坐标
            title: 图像标题
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # 创建一个图形，包含两个子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            fig.suptitle(title, fontsize=16)
            
            # 1. 绘制邻接矩阵热图
            im = ax1.imshow(adjacency_matrix, cmap='viridis')
            ax1.set_title("邻接矩阵")
            ax1.set_xlabel("骨骼索引")
            ax1.set_ylabel("骨骼索引")
            plt.colorbar(im, ax=ax1)
            
            # 2. 绘制网络图
            G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph)
            
            # 如果提供了骨骼坐标，使用这些作为节点位置
            if skeleton_points is not None:
                # 为了更好地可视化，只使用前两个坐标
                pos = {i: (skeleton_points[i, 0], skeleton_points[i, 1]) for i in range(len(skeleton_points))}
            else:
                # 否则使用spring布局
                pos = nx.spring_layout(G)
            
            # 计算每个节点的度
            degrees = np.sum(adjacency_matrix, axis=1) + np.sum(adjacency_matrix, axis=0)
            
            # 绘制节点，节点大小与度数成比例
            nx.draw_networkx_nodes(G, pos, ax=ax2, 
                                 node_size=[50 + 30 * d for d in degrees],
                                 node_color=list(degrees),
                                 cmap='viridis',
                                 alpha=0.8)
            
            # 绘制边，边宽度与权重成比例
            edge_weights = [adjacency_matrix[i, j] for i, j in G.edges()]
            nx.draw_networkx_edges(G, pos, ax=ax2, 
                                 width=edge_weights,
                                 alpha=0.6,
                                 arrows=True,
                                 arrowstyle='-|>',
                                 arrowsize=10)
            
            # 绘制节点标签
            nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8)
            
            ax2.set_title("骨骼连接图")
            ax2.set_xlabel("X坐标")
            ax2.set_ylabel("Y坐标")
            ax2.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig("skeleton_adjacency.png", dpi=150)
            print(f"已保存骨骼邻接图到 skeleton_adjacency.png")
            plt.close()
            
        except ImportError:
            print("缺少必要的库：matplotlib 或 networkx。请安装它们以启用可视化功能。")
        except Exception as e:
            print(f"可视化骨骼邻接矩阵时出错: {e}")

    def combine_relative_bones(self, adjacency_matrix, transformations, skeleton_points, labels=None):
        """
        将adjacency邻接且相对运动几乎一致的骨骼合并，简化骨骼结构
        
        Args:
            adjacency_matrix: numpy数组，形状为(B, B)，表示骨骼的邻接矩阵
            transformations: numpy数组，形状为(T, B, 4, 4)，表示骨骼的变换矩阵
            skeleton_points: numpy数组，形状为(B, 3)，表示骨骼点的位置
            labels: numpy数组，形状为(N,)，表示每个顶点所属的骨骼标签，可选
            
        Returns:
            new_adjacency: 合并后的邻接矩阵，形状为(new_B, new_B)
            new_transformations: 更新后的变换矩阵，形状为(T, new_B, 4, 4)
            new_bones: 合并后的骨骼位置，形状为(new_B, 3)
            new_labels: 合并后的顶点标签，仅当提供labels参数时返回
        """
        B = adjacency_matrix.shape[0]  # 原始骨骼数量
        T = transformations.shape[0]   # 帧数
        
        if skeleton_points is None or skeleton_points.shape[0] != B:
            raise ValueError(f"skeleton_points必须提供，且形状应为({B}, 3)")
        
        # 可视化原始邻接矩阵
        print("可视化原始骨骼邻接结构...")
        self.visualize_adjacency_matrix(adjacency_matrix, skeleton_points, title="原始骨骼邻接结构")
        
        # 分析邻接节点的相似度
        print("分析邻接节点的相似度...")
        edge_count = 0
        high_similarity_count = 0  # 相似度高于阈值的边数量
        similarity_values = []
        
        for i in range(B):
            for j in range(i+1, B):  # 只处理上三角矩阵，避免重复
                if adjacency_matrix[i, j] > 0 or adjacency_matrix[j, i] > 0:  # 如果两点邻接
                    similarity = self.compute_motion_similarity_external(i, j, transformations)
                    similarity_values.append(similarity)
                    edge_count += 1
                    
                    if similarity > 0.75:  # 使用与get_merge_groups相同的阈值
                        high_similarity_count += 1
                        print(f"高相似度边: 骨骼 {i} - {j}, 相似度: {similarity:.4f}")
        
        if edge_count > 0:
            print(f"邻接节点总数: {edge_count}, 高相似度边数: {high_similarity_count} ({high_similarity_count/edge_count*100:.2f}%)")
            
            if similarity_values:
                similarity_values = np.array(similarity_values)
                print(f"邻接节点相似度统计 - 最小值: {np.min(similarity_values):.4f}, 最大值: {np.max(similarity_values):.4f}")
                print(f"邻接节点相似度统计 - 均值: {np.mean(similarity_values):.4f}, 中位数: {np.median(similarity_values):.4f}")
                print(f"邻接节点相似度分位数 - 25%: {np.percentile(similarity_values, 25):.4f}, 75%: {np.percentile(similarity_values, 75):.4f}")
        
        # 定义子函数：计算节点的度
        def compute_node_degrees(adj_matrix):
            """计算每个节点的度（同时考虑入度和出度）"""
            # 计算出度 (行和)
            out_degrees = np.sum(adj_matrix, axis=1)
            # 计算入度 (列和)
            in_degrees = np.sum(adj_matrix, axis=0)
            # 返回总度数 (入度+出度)
            return out_degrees + in_degrees
        
        # 定义子函数：计算两个骨骼之间的运动相似度
        def compute_motion_similarity(bone_i, bone_j, trans_matrices):
            """
            计算两个骨骼之间的运动相似度
            同时考虑旋转矩阵和平移向量的相似度
            """
            rot_sim = 0  # 旋转相似度
            trans_sim = 0  # 平移相似度
            
            # 提取所有时间步的旋转和平移信息
            for t in range(T):
                # 提取旋转矩阵
                rot_i = trans_matrices[t, bone_i, :3, :3]
                rot_j = trans_matrices[t, bone_j, :3, :3]
                
                # 提取平移向量
                trans_i = trans_matrices[t, bone_i, :3, 3]
                trans_j = trans_matrices[t, bone_j, :3, 3]
                
                # 计算旋转矩阵的相似度 (使用Frobenius内积归一化)
                # 确保分母不为零
                rot_norm_i = np.linalg.norm(rot_i)
                rot_norm_j = np.linalg.norm(rot_j)
                if rot_norm_i > 1e-6 and rot_norm_j > 1e-6:
                    rot_sim += np.sum(rot_i * rot_j) / (rot_norm_i * rot_norm_j)
                
                # 计算平移向量的相似度 (使用余弦相似度)
                # 确保分母不为零
                trans_norm_i = np.linalg.norm(trans_i)
                trans_norm_j = np.linalg.norm(trans_j)
                if trans_norm_i > 1e-6 and trans_norm_j > 1e-6:
                    trans_sim += np.sum(trans_i * trans_j) / (trans_norm_i * trans_norm_j)
            
            # 归一化
            rot_sim /= T
            trans_sim /= T
            
            # 组合旋转和平移相似度，给予旋转更高的权重
            # 这里使用0.7和0.3的权重，可以根据需要调整
            combined_sim = 0.7 * rot_sim + 0.3 * trans_sim
            
            return combined_sim
        
        # 定义子函数：获取节点的所有邻居
        def get_neighbors(node, adj_matrix):
            """获取节点的所有邻居（同时考虑出邻居和入邻居）"""
            # 找出出邻居 (该节点指向的节点)
            out_neighbors = np.where(adj_matrix[node] > 0)[0]
            # 找出入邻居 (指向该节点的节点)
            in_neighbors = np.where(adj_matrix[:, node] > 0)[0]
            # 合并并去重
            all_neighbors = np.union1d(out_neighbors, in_neighbors)
            return all_neighbors
        
        # 定义子函数：获取所有可能的合并列表
        def get_merge_groups(adj_matrix, trans_matrices, similarity_threshold=0.9):
            """
            获取所有可能的合并组
            
            Args:
                adj_matrix: 邻接矩阵
                trans_matrices: 变换矩阵
                similarity_threshold: 相似度阈值
                
            Returns:
                merge_groups: 合并组列表，每个元素是一个包含可合并骨骼索引的列表
            """
            degrees = compute_node_degrees(adj_matrix)
            visited = np.zeros(B, dtype=bool)  # 标记节点是否已被访问
            merge_groups = []  # 存储所有合并组
            
            # 调试信息：记录各种度数的节点数量
            degree_counts = np.bincount(degrees.astype(int))
            print(f"节点度数分布: {degree_counts}")
            print(f"度为1的节点数量: {np.sum(degrees == 1)}")
            print(f"度为2的节点数量: {np.sum(degrees == 2)}")
            print(f"度大于2的节点数量: {np.sum(degrees > 2)}")
            
            # 第一次尝试：从度为1的节点开始分析
            merge_count_from_deg1 = 0
            for start_node in range(B):
                if degrees[start_node] == 1 and not visited[start_node]:
                    neighbors = get_neighbors(start_node, adj_matrix)
                    if len(neighbors) == 0:  # 孤立节点
                        continue
                        
                    neighbor = neighbors[0]  # 度为1的节点只有一个邻居
                    similarity = compute_motion_similarity(start_node, neighbor, trans_matrices)
                    
                    # 如果相似度高，则创建一个新的合并组
                    if similarity > similarity_threshold:
                        current_group = [start_node]
                        queue = [neighbor]  # 待处理的邻居队列
                        local_visited = {start_node}  # 本次BFS中已访问的节点
                        
                        # 广度优先搜索相似的邻居
                        while queue:
                            current = queue.pop(0)
                            if current in local_visited:
                                continue
                                
                            # 检查与组内所有骨骼的相似度，要求都相似
                            is_similar = True
                            connected_nodes = []
                            
                            # 找出组内与当前节点有直接连接的节点
                            for node in current_group:
                                if adj_matrix[current, node] > 0 or adj_matrix[node, current] > 0:
                                    connected_nodes.append(node)
                            
                            # 如果有直接连接的节点，则只检查与这些节点的相似度
                            if connected_nodes:
                                for node in connected_nodes:
                                    sim = compute_motion_similarity(current, node, trans_matrices)
                                    print(f"检查相似度: 节点 {current} 与 {node}, 相似度: {sim:.4f}")
                                    if sim <= similarity_threshold:
                                        is_similar = False
                                        break
                            else:
                                # 如果没有直接连接的节点，则随机检查与组内部分节点的相似度
                                sample_nodes = current_group[:min(3, len(current_group))]  # 最多取3个
                                for node in sample_nodes:
                                    sim = compute_motion_similarity(current, node, trans_matrices)
                                    print(f"检查相似度(无直接连接): 节点 {current} 与 {node}, 相似度: {sim:.4f}")
                                    if sim <= similarity_threshold:
                                        is_similar = False
                                        break
                            
                            if is_similar:
                                current_group.append(current)
                                local_visited.add(current)
                                
                                # 添加未访问的邻居到队列
                                for next_node in get_neighbors(current, adj_matrix):
                                    if next_node not in local_visited:
                                        queue.append(next_node)
                        
                        # 如果合并组包含多个骨骼，则添加到结果中
                        if len(current_group) > 1:
                            merge_groups.append(current_group)
                            # 标记这些节点已被访问
                            for node in current_group:
                                visited[node] = True
                            merge_count_from_deg1 += 1
            
            print(f"从度为1的节点开始，找到的合并组数量: {merge_count_from_deg1}")
            
            # 第二次尝试：如果没有找到足够的合并组，从度大于1的节点也尝试合并
            if len(merge_groups) == 0:
                print("从度为1的节点没有找到合并组，尝试从所有未访问节点开始...")
                # 重置visited数组
                visited = np.zeros(B, dtype=bool)
                
                for start_node in range(B):
                    if not visited[start_node]:
                        neighbors = get_neighbors(start_node, adj_matrix)
                        if len(neighbors) == 0:  # 孤立节点
                            continue
                        
                        # 找出与start_node相似度最高的邻居
                        best_neighbor = None
                        best_similarity = -1
                        
                        for neighbor in neighbors:
                            similarity = compute_motion_similarity(start_node, neighbor, trans_matrices)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_neighbor = neighbor
                        
                        # 如果找到了相似度高的邻居，尝试合并
                        if best_similarity > similarity_threshold and best_neighbor is not None:
                            current_group = [start_node]
                            queue = [best_neighbor]  # 待处理的邻居队列
                            local_visited = {start_node}  # 本次BFS中已访问的节点
                            
                            # 广度优先搜索相似的邻居
                            while queue:
                                current = queue.pop(0)
                                if current in local_visited:
                                    continue
                                
                                # 检查与组内所有骨骼的相似度
                                is_similar = True
                                connected_nodes = []
                                
                                # 找出组内与当前节点有直接连接的节点
                                for node in current_group:
                                    if adj_matrix[current, node] > 0 or adj_matrix[node, current] > 0:
                                        connected_nodes.append(node)
                                
                                # 如果有直接连接的节点，则只检查与这些节点的相似度
                                if connected_nodes:
                                    for node in connected_nodes:
                                        sim = compute_motion_similarity(current, node, trans_matrices)
                                        print(f"检查相似度: 节点 {current} 与 {node}, 相似度: {sim:.4f}")
                                        if sim <= similarity_threshold:
                                            is_similar = False
                                            break
                                else:
                                    # 如果没有直接连接的节点，则随机检查与组内部分节点的相似度
                                    sample_nodes = current_group[:min(3, len(current_group))]  # 最多取3个
                                    for node in sample_nodes:
                                        sim = compute_motion_similarity(current, node, trans_matrices)
                                        print(f"检查相似度(无直接连接): 节点 {current} 与 {node}, 相似度: {sim:.4f}")
                                        if sim <= similarity_threshold:
                                            is_similar = False
                                            break
                                
                                if is_similar:
                                    current_group.append(current)
                                    local_visited.add(current)
                                    
                                    # 添加未访问的邻居到队列
                                    for next_node in get_neighbors(current, adj_matrix):
                                        if next_node not in local_visited:
                                            queue.append(next_node)
                            
                            # 如果合并组包含多个骨骼，则添加到结果中
                            if len(current_group) > 1:
                                print(f"找到新的合并组 (大小: {len(current_group)}): {current_group}")
                                merge_groups.append(current_group)
                                # 标记这些节点已被访问
                                for node in current_group:
                                    visited[node] = True
                
                print(f"从所有节点尝试后，找到的合并组总数: {len(merge_groups)}")
            
            # 输出一些找到的合并组示例
            if merge_groups:
                for i, group in enumerate(sorted(merge_groups, key=len, reverse=True)[:3]):
                    print(f"合并组 #{i+1} (大小: {len(group)}): {group}")
                    # 打印组内骨骼之间的相似度
                    print("组内骨骼相似度矩阵:")
                    for a in group:
                        sims = []
                        for b in group:
                            if a != b:
                                sim = compute_motion_similarity(a, b, trans_matrices)
                                sims.append(f"{sim:.4f}")
                            else:
                                sims.append("1.0000")
                        print(f"  骨骼 {a}: {sims}")
            
            return merge_groups
        
        # 定义子函数：处理合并组，选择代表骨骼
        def process_merge_groups(merge_groups, adj_matrix):
            """
            处理合并组，为每个组选择一个代表骨骼
            
            Args:
                merge_groups: 合并组列表
                adj_matrix: 邻接矩阵
                
            Returns:
                bone_mapping: 原始骨骼到新骨骼的映射数组
                keep_bones: 保留的骨骼索引列表
            """
            degrees = compute_node_degrees(adj_matrix)
            bone_mapping = np.arange(B)  # 初始化映射，默认每个骨骼映射到自己
            
            for group in merge_groups:
                # 找出组内度最大的骨骼作为代表
                representative = group[0]
                max_degree = degrees[representative]
                
                for node in group[1:]:
                    if degrees[node] > max_degree:
                        max_degree = degrees[node]
                        representative = node
                
                # 将组内其他骨骼映射到代表骨骼
                for node in group:
                    if node != representative:
                        bone_mapping[node] = representative
            
            # 确定保留的骨骼列表（去除重复）
            keep_bones = []
            for i in range(B):
                if bone_mapping[i] == i:  # 如果骨骼映射到自己，说明是代表骨骼
                    keep_bones.append(i)
            
            return bone_mapping, keep_bones
        
        # 定义子函数：创建新的骨骼索引
        def create_new_indices(bone_mapping, keep_bones):
            """
            创建新的骨骼索引映射
            
            Args:
                bone_mapping: 原始骨骼到代表骨骼的映射
                keep_bones: 保留的骨骼索引列表
                
            Returns:
                new_indices: 代表骨骼到新索引的映射字典
                new_mapping: 原始骨骼到新索引的映射数组
            """
            new_indices = {}  # 代表骨骼到新索引的映射
            for i, bone in enumerate(keep_bones):
                new_indices[bone] = i
            
            # 创建从原始骨骼到新索引的直接映射
            new_mapping = np.zeros(B, dtype=int)
            for i in range(B):
                representative = bone_mapping[i]
                new_mapping[i] = new_indices[representative]
            
            return new_indices, new_mapping
        
        # 定义子函数：更新邻接矩阵
        def update_adjacency(old_adj, new_mapping, new_bone_count):
            """更新邻接矩阵"""
            new_adj = np.zeros((new_bone_count, new_bone_count))
            
            for i in range(B):
                for j in range(B):
                    if old_adj[i, j] > 0:
                        new_i = new_mapping[i]
                        new_j = new_mapping[j]
                        if new_i != new_j:  # 避免自环
                            new_adj[new_i, new_j] = 1
            
            return new_adj
        
        # 定义子函数：更新变换矩阵 - 简化为只提取保留骨骼的变换
        def extract_transformations(old_trans, keep_bones, new_mapping):
            """
            从原始变换矩阵中提取保留骨骼的变换
            
            Args:
                old_trans: 原始变换矩阵, 形状为(T, B, 4, 4)
                keep_bones: 保留的骨骼列表
                new_mapping: 原始骨骼到新索引的映射
                
            Returns:
                new_trans: 提取的变换矩阵，形状为(T, new_bone_count, 4, 4)
            """
            new_bone_count = len(keep_bones)
            new_trans = np.zeros((T, new_bone_count, 4, 4))
            
            # 只保留代表骨骼的变换矩阵
            for t in range(T):
                for old_idx in keep_bones:
                    new_idx = new_mapping[old_idx]
                    new_trans[t, new_idx] = old_trans[t, old_idx]
            
            return new_trans
        
        # 定义子函数：计算新的骨骼位置
        def extract_bones(old_bones, keep_bones, new_mapping):
            """
            提取新的骨骼位置
            
            Args:
                old_bones: 原始骨骼位置, 形状为(B, 3)
                keep_bones: 保留的骨骼列表
                new_mapping: 原始骨骼到新索引的映射
                
            Returns:
                new_bones: 新的骨骼位置, 形状为(new_bone_count, 3)
            """
            new_bone_count = len(keep_bones)
            new_bones = np.zeros((new_bone_count, 3))
            
            # 只保留代表骨骼的位置
            for old_idx in keep_bones:
                new_idx = new_mapping[old_idx]
                new_bones[new_idx] = old_bones[old_idx]
            
            return new_bones
        
        # 定义子函数：更新顶点标签
        def update_labels(old_labels, new_mapping):
            """更新顶点标签"""
            if old_labels is None:
                return None
                
            new_labels = np.zeros_like(old_labels)
            for i in range(len(old_labels)):
                if old_labels[i] >= 0 and old_labels[i] < B:
                    new_labels[i] = new_mapping[old_labels[i]]
            
            return new_labels
        
        # 主函数逻辑开始
        # 1. 获取所有可能的合并组
        merge_groups = get_merge_groups(adjacency_matrix, transformations)
        
        # 2. 处理合并组，选择代表骨骼
        bone_mapping, keep_bones = process_merge_groups(merge_groups, adjacency_matrix)
        
        # 3. 创建新的骨骼索引
        new_indices, new_mapping = create_new_indices(bone_mapping, keep_bones)
        new_bone_count = len(keep_bones)
        
        # 4. 更新邻接矩阵
        new_adjacency = update_adjacency(adjacency_matrix, new_mapping, new_bone_count)
        
        # 5. 提取变换矩阵
        new_transformations = extract_transformations(transformations, keep_bones, new_mapping)
        
        # 6. 提取新的骨骼位置
        new_bones = extract_bones(skeleton_points, keep_bones, new_mapping)
        
        # 7. 更新顶点标签
        new_labels = update_labels(labels, new_mapping)
        
        # 8. 打印合并信息
        print(f"原始骨骼数量: {B}")
        print(f"合并后骨骼数量: {new_bone_count}")
        print(f"合并比例: {(B - new_bone_count) / B * 100:.2f}%")
        print(f"合并组数量: {len(merge_groups)}")
        
        # 可视化合并后的邻接矩阵
        if new_bone_count < B:  # 只有在实际发生合并时才可视化
            print("可视化合并后的骨骼邻接结构...")
            self.visualize_adjacency_matrix(new_adjacency, new_bones, title="合并后骨骼邻接结构")
        
        # 9. 返回结果，始终包含new_bones
        if labels is not None:
            return new_adjacency, new_transformations, new_bones, new_labels
        else:
            return new_adjacency, new_transformations, new_bones

    def compute_motion_similarity_external(self, bone_i, bone_j, transformations):
        """
        计算两个骨骼之间的运动相似度的外部接口
        
        Args:
            bone_i: 第一个骨骼的索引
            bone_j: 第二个骨骼的索引
            transformations: 变换矩阵，形状为(T, B, 4, 4)
        
        Returns:
            similarity: 两个骨骼之间的运动相似度
        """
        T = transformations.shape[0]
        rot_sim = 0
        trans_sim = 0
        
        for t in range(T):
            # 提取旋转矩阵
            rot_i = transformations[t, bone_i, :3, :3]
            rot_j = transformations[t, bone_j, :3, :3]
            
            # 提取平移向量
            trans_i = transformations[t, bone_i, :3, 3]
            trans_j = transformations[t, bone_j, :3, 3]
            
            # 计算旋转矩阵的相似度
            rot_norm_i = np.linalg.norm(rot_i)
            rot_norm_j = np.linalg.norm(rot_j)
            if rot_norm_i > 1e-6 and rot_norm_j > 1e-6:
                rot_sim += np.sum(rot_i * rot_j) / (rot_norm_i * rot_norm_j)
            
            # 计算平移向量的相似度
            trans_norm_i = np.linalg.norm(trans_i)
            trans_norm_j = np.linalg.norm(trans_j)
            if trans_norm_i > 1e-6 and trans_norm_j > 1e-6:
                trans_sim += np.sum(trans_i * trans_j) / (trans_norm_i * trans_norm_j)
        
        # 归一化
        rot_sim /= T
        trans_sim /= T
        
        # 组合旋转和平移相似度
        combined_sim = 0.7 * rot_sim + 0.3 * trans_sim
        
        return combined_sim


# 示例用法
if __name__ == "__main__":
    # 加载数据
    motion_analysis_results = np.load("/home/ubuntu/research/code/GaussianAnimator/tempexp/trex/motion_analysis_results.npy", 
                                    allow_pickle=True).item()
    moving_points = motion_analysis_results['moving_points']
    bones = motion_analysis_results['bones']
    joints_index = motion_analysis_results['joints_index']
    root_indx = motion_analysis_results['root_index']
    moving_mask = motion_analysis_results['moving_mask']
    motion_magnitudes = motion_analysis_results['motion_magnitudes']
    poses = motion_analysis_results['poses']
    rest_pose = motion_analysis_results['rest_pose']
    skeleton_joints = moving_points[joints_index]

    # 创建SSDR计算器实例
    calculator = DemBonesCalculator()

    if os.path.exists("result_can_be_visualize.npy"):
        result_can_be_visualize = np.load("result_can_be_visualize.npy", allow_pickle=True).item()
        weights = result_can_be_visualize["weights"]
        skeleton = result_can_be_visualize["skeleton"]
    else:
        # 运行SSDR
        weights, transformations, rmse, reconstruction = calculator.compute_ssdr(
            poses, rest_pose, skeleton_joints, None, max_iters=10)
        result_can_be_visualize = {
            "rest_pose": rest_pose,
            "weights": weights,
            "skeleton": skeleton_joints,
            "transformations": transformations,
        }
        np.save("result_can_be_visualize.npy", result_can_be_visualize)

    # 可视化
    calculator.visualize_4d_skinning(poses, weights, skeleton_joints, fps=30, bone_size=0.05)
    
    # 获取动态骨骼序列
    motion_skeleton_seq = calculator.get_max_weights_bone_seq(weights, poses)