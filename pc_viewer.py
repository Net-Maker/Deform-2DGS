import numpy as np
import time
from tqdm import tqdm
import open3d as o3d

class VisualizeObjects:
    def __init__(self):
        self.vis = None
        self.geometries = {}
        self.play_state = {'playing': True, 'frame_index': 0}
        self.opt = None
        
    def setup_visualizer(self, background_color=[0.1, 0.1, 0.1]):
        """初始化可视化器"""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.opt = self.vis.get_render_option()
        self.opt.background_color = np.array(background_color)
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        return self.opt
        
    def add_points(self, points, colors=None, point_size=3.0):
        """添加点云"""
        if self.opt:
            self.opt.point_size = point_size
            
        pcd = o3d.geometry.PointCloud()
        if isinstance(points, list) or (isinstance(points, np.ndarray) and len(points.shape) == 3):
            # 点云序列
            pcd.points = o3d.utility.Vector3dVector(points[0])
            self.geometries['points'] = {
                'object': pcd,
                'sequence': points,
                'type': 'points'
            }
        else:
            # 单帧点云
            pcd.points = o3d.utility.Vector3dVector(points)
            self.geometries['points'] = {
                'object': pcd,
                'sequence': [points],
                'type': 'points'
            }
            
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([1, 1, 1])
            
        self.vis.add_geometry(pcd)
        return pcd
        
    def add_bones(self, positions, colors=None, bone_size=0.05):
        """添加骨骼"""
        if isinstance(positions, list) or (isinstance(positions, np.ndarray) and len(positions.shape) == 3):
            # 骨骼序列
            bone_sequence = positions
            current_positions = positions[0]
        else:
            # 单帧骨骼
            bone_sequence = [positions]
            current_positions = positions
            
        if colors is None:
            colors = [
                np.array([np.sin(i * 2.0944), np.sin(i * 2.0944 + 2.0944),
                         np.sin(i * 2.0944 + 4.1888)]) * 0.5 + 0.5
                for i in range(len(current_positions))
            ]
            
        bone_spheres = []
        for i, pos in enumerate(current_positions):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=bone_size)
            sphere.translate(pos)
            sphere.paint_uniform_color(colors[i])
            sphere.compute_vertex_normals()
            self.vis.add_geometry(sphere)
            bone_spheres.append(sphere)
            
        self.geometries['bones'] = {
            'object': bone_spheres,
            'sequence': bone_sequence,
            'type': 'bones'
        }
        return bone_spheres
        
    def update_frame(self, frame_idx):
        """更新一帧的数据"""
        for key, geom in self.geometries.items():
            if geom['type'] == 'points':
                if frame_idx < len(geom['sequence']):
                    pcd = geom['object']
                    pcd.points = o3d.utility.Vector3dVector(geom['sequence'][frame_idx])
                    self.vis.update_geometry(pcd)
            elif geom['type'] == 'bones':
                if frame_idx < len(geom['sequence']):
                    bone_spheres = geom['object']
                    positions = geom['sequence'][frame_idx]
                    for sphere, pos in zip(bone_spheres, positions):
                        current_pos = np.asarray(sphere.vertices)[0]
                        translation = pos - current_pos
                        sphere.translate(translation)
                        sphere.compute_vertex_normals()
                        self.vis.update_geometry(sphere)
                    
    def get_max_frames(self):
        """获取最大帧数"""
        max_frames = 1
        for geom in self.geometries.values():
            if isinstance(geom['sequence'], (list, np.ndarray)):
                max_frames = max(max_frames, len(geom['sequence']))
        return max_frames
        
    def visualize(self, fps=30, output_path=None, loop=True, background_color=[0.1, 0.1, 0.1]):
        """主可视化函数"""
        try:
            max_frames = self.get_max_frames()
            
            # 检查是否有几何体
            if not self.geometries:
                print("错误：没有可视化对象")
                return
                
            # 视频导出
            if output_path:
                try:
                    import cv2
                    width, height = 1920, 1080
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    for frame_idx in tqdm(range(max_frames), desc="导出视频"):
                        self.update_frame(frame_idx)
                        self.vis.poll_events()
                        self.vis.update_renderer()
                        
                        img = np.asarray(self.vis.capture_screen_float_buffer())
                        img = (img * 255).astype(np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out.write(img)
                    
                    out.release()
                    print(f"视频已保存至: {output_path}")
                    
                except Exception as e:
                    print(f"保存视频时出错: {e}")
                    output_path = None
            
            # 交互式显示
            if not output_path:
                self.play_state = {'playing': True, 'frame_index': 0}
                last_update_time = time.time()
                
                print("\n交互式控制说明:")
                print("Space: 暂停/继续")
                print("Esc: 退出")
                print("鼠标左键: 旋转")
                print("鼠标右键: 平移")
                print("鼠标滚轮: 缩放")
                
                try:
                    while True:
                        current_time = time.time()
                        if current_time - last_update_time >= 1.0/fps and self.play_state['playing'] and max_frames > 1:
                            frame_idx = self.play_state['frame_index']
                            self.update_frame(frame_idx)
                            
                            # 更新帧索引
                            self.play_state['frame_index'] = (frame_idx + 1) % max_frames if loop else min(frame_idx + 1, max_frames - 1)
                            last_update_time = current_time
                        
                        if not self.vis.poll_events():
                            break
                        self.vis.update_renderer()
                        
                except KeyboardInterrupt:
                    print("\n用户中断了可视化。")
                except Exception as e:
                    print(f"\n可视化过程中出错: {e}")
                    
        finally:
            try:
                self.vis.destroy_window()
            except:
                pass

def visualize(points=None, bones=None, weights=None, **kwargs):
    """统一的可视化接口"""
    vis = VisualizeObjects()
    
    # 设置可视化器
    background_color = kwargs.get('background_color', [0.1, 0.1, 0.1])
    vis.setup_visualizer(background_color)
    
    # 检查点云数据的类型并转换为numpy数组
    if points is not None:
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        point_colors = kwargs.get('point_colors', None)
        if point_colors is not None and isinstance(point_colors, torch.Tensor):
            point_colors = point_colors.detach().cpu().numpy()
            
        point_size = kwargs.get('point_size', 3.0)
        vis.add_points(points, point_colors, point_size)
        
    if bones is not None:
        if isinstance(bones, torch.Tensor):
            bones = bones.detach().cpu().numpy()
            
        bone_colors = kwargs.get('bone_colors', None)
        if bone_colors is not None and isinstance(bone_colors, torch.Tensor):
            bone_colors = bone_colors.detach().cpu().numpy()
            
        bone_size = kwargs.get('bone_size', 0.05)
        vis.add_bones(bones, bone_colors, bone_size)
        
    if weights is not None and points is not None:
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
            
        # 使用权重生成点云颜色
        points_shape = points.shape
        if len(points_shape) == 3:  # (T, N, 3)
            num_points = points_shape[1]
        else:  # (N, 3)
            num_points = points_shape[0]
            
        if weights.shape[0] != num_points:
            weights = weights.T
            
        point_colors = np.zeros((weights.shape[0], 3))
        for i in range(weights.shape[1]):
            color = np.array([np.sin(i * 2.0944), np.sin(i * 2.0944 + 2.0944),
                            np.sin(i * 2.0944 + 4.1888)]) * 0.5 + 0.5
            point_colors += weights[:, i:i+1] * color
            
        # 确保颜色在有效范围内
        point_colors = np.clip(point_colors, 0, 1)
        vis.add_points(points, point_colors)
    
    # 提取参数并调用可视化
    fps = kwargs.get('fps', 30)
    output_path = kwargs.get('output_path', None)
    loop = kwargs.get('loop', True)
    
    vis.visualize(fps=fps, output_path=output_path, loop=loop, background_color=background_color)

# 处理torch导入
try:
    import torch
except ImportError:
    pass

import torch
from pytorch3d.ops import knn_points

def compute_knn_weights(poses, k=20, sigma=2000):
    """
    计算基于KNN的权重矩阵
    
    Args:
        poses: (T, N, 3) 点云序列或(N, 3)单帧点云
        k: int KNN的邻居数量
        sigma: float 高斯核参数，控制权重衰减速度
    
    Returns:
        weights: (N, N) 权重矩阵，每行表示一个点对其他点的权重
        knn_idx: (N, k) KNN索引
        knn_weights: (N, k) KNN权重
    """
    # 确保输入是tensor且在GPU上
    if isinstance(poses, list):
        poses = torch.tensor(poses[0], device='cuda')  # 使用第一帧
    elif isinstance(poses, torch.Tensor):
        if len(poses.shape) == 3:
            poses = poses[0]  # 使用第一帧
        if not poses.is_cuda:
            poses = poses.cuda()
    else:
        poses = torch.tensor(poses, device='cuda')
    
    # 确保是contiguous的
    poses = poses.contiguous()
    
    # 计算KNN
    knn_result = knn_points(poses.unsqueeze(0), poses.unsqueeze(0), K=k+1)
    knn_idx = knn_result.idx.squeeze(0)[:, 1:].contiguous()  # [N, K] 去掉自身
    knn_dists = knn_result.dists.squeeze(0)[:, 1:].sqrt().contiguous()  # [N, K]
    
    # 计算权重
    knn_weights = torch.exp(-sigma * knn_dists**2)  # 高斯核
    
    # 归一化权重
    knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True)
    
    # 构建完整的权重矩阵（稀疏表示）
    N = poses.shape[0]
    weights = torch.zeros((N, N), device='cuda')
    
    # 使用scatter填充权重
    row_idx = torch.arange(N, device='cuda').unsqueeze(1).expand(-1, k)
    weights[row_idx, knn_idx] = knn_weights
    
    return weights, knn_idx, knn_weights

# 使用示例
def visualize_with_weights(poses):
    """
    使用KNN权重可视化点云，用高对比度颜色方案
    
    Args:
        poses: (T, N, 3) 点云序列或(N, 3)单帧点云
    """
    # 计算权重
    weights, knn_idx, knn_weights = compute_knn_weights(poses, k=20, sigma=2000)
    
    # 生成颜色
    N = weights.shape[0]
    colors = torch.zeros((N, 3), device='cuda')
    
    # 选择随机中心点以增加局部变化
    center_indices = torch.randperm(N, device='cuda')[:32]
    
    # 使用高对比度颜色方案 - 方法1：增加颜色频率
    for i, center_idx in enumerate(center_indices):
        # 增加颜色频率，使颜色变化更快
        freq = 6.0944  # 原来是2.0944，增大以提高频率
        color = torch.tensor([
            np.sin(i * freq),
            np.sin(i * freq + 2.0944),
            np.sin(i * freq + 4.1888)
        ], device='cuda') * 0.5 + 0.5
        
        # 使用权重矩阵的一列来着色
        colors += weights[:, center_idx:center_idx+1] * color.unsqueeze(0)
    
    # 确保颜色在有效范围内
    colors = torch.clamp(colors, 0, 1)
    
    # 可视化
    visualize(
        points=poses,
        point_colors=colors.cpu().numpy(),
        point_size=5.0,
        background_color=[0.1, 0.1, 0.1]
    )
    
    return weights, knn_idx, knn_weights

# 另一种高对比度颜色方案
def visualize_with_high_contrast(poses):
    """
    使用高对比度颜色方案可视化点云
    """
    # 计算权重
    weights, knn_idx, knn_weights = compute_knn_weights(poses, k=20, sigma=2000)
    
    # 生成颜色
    N = weights.shape[0]
    colors = torch.zeros((N, 3), device='cuda')
    
    # 选择随机中心点，确保邻近中心点在空间上有较大差异
    center_indices = torch.randperm(N, device='cuda')[:32]
    
    # 计算中心点之间的距离
    def hsv_to_rgb(h, s, v):
        """HSV转RGB，h:[0,1], s:[0,1], v:[0,1]"""
        h = h * 6.0
        i = torch.floor(h)
        f = h - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        
        rgb = torch.zeros((h.shape[0], 3), device=h.device)
        mask = (i == 0)
        rgb[mask] = torch.stack([v[mask], t[mask], p[mask]], dim=1)
        mask = (i == 1)
        rgb[mask] = torch.stack([q[mask], v[mask], p[mask]], dim=1)
        mask = (i == 2)
        rgb[mask] = torch.stack([p[mask], v[mask], t[mask]], dim=1)
        mask = (i == 3)
        rgb[mask] = torch.stack([p[mask], q[mask], v[mask]], dim=1)
        mask = (i == 4)
        rgb[mask] = torch.stack([t[mask], p[mask], v[mask]], dim=1)
        mask = (i == 5)
        rgb[mask] = torch.stack([v[mask], p[mask], q[mask]], dim=1)
        
        return rgb
    
    # 为中心点分配HSV颜色，确保相邻中心点颜色差异大
    # hue值用黄金比例分布，确保最大颜色差异
    hues = torch.zeros(32, device='cuda')
    for i in range(32):
        hues[i] = (i * 0.618033988749895) % 1.0  # 黄金比例分布
    
    saturations = torch.ones(32, device='cuda')  # 饱和度
    values = torch.ones(32, device='cuda')       # 亮度
    
    # 转换为RGB
    center_colors = hsv_to_rgb(hues, saturations, values)
    
    # 根据权重生成颜色
    for i, center_idx in enumerate(center_indices):
        colors += weights[:, center_idx:center_idx+1] * center_colors[i].unsqueeze(0)
    
    # 增强对比度（可选）
    contrast = 1.2  # 对比度增强因子
    colors = ((colors - 0.5) * contrast + 0.5)
    colors = torch.clamp(colors, 0, 1)
    
    # 可视化
    visualize(
        points=poses,
        point_colors=colors.cpu().numpy(),
        point_size=5.0,
        background_color=[0.1, 0.1, 0.1]
    )
    
    return weights, knn_idx, knn_weights

# 离散颜色方案，最大化对比度 - 修复版
def visualize_with_discrete_colors(poses):
    """
    使用离散颜色方案可视化点云，最大化邻近点的对比度
    """
    # 用空间聚类而不是KNN权重来区分区域
    if isinstance(poses, list) or (isinstance(poses, np.ndarray) and len(poses.shape) == 3):
        points = torch.tensor(poses[0], device='cuda')
    else:
        points = torch.tensor(poses, device='cuda')
    
    # 确保是contiguous的
    points = points.contiguous()
    
    # 用K-means进行聚类
    N = points.shape[0]
    num_clusters = 25  # 可以调整聚类数量
    
    # 随机选择初始聚类中心
    if N < num_clusters:
        num_clusters = N
    
    # 随机初始化聚类中心
    center_indices = torch.randperm(N, device='cuda')[:num_clusters]
    centers = points[center_indices]
    
    # 简单的K-means迭代 (5次足够看到效果)
    for _ in range(5):
        # 计算每个点到每个中心的距离
        dists = torch.cdist(points, centers)  # (N, num_clusters)
        
        # 分配每个点到最近的中心
        cluster_ids = torch.argmin(dists, dim=1)  # (N,)
        
        # 更新中心
        for c in range(num_clusters):
            cluster_points = points[cluster_ids == c]
            if len(cluster_points) > 0:
                centers[c] = cluster_points.mean(dim=0)
    
    # 创建高对比度颜色映射
    discrete_colors = torch.tensor([
        [1.0, 0.0, 0.0],  # 红
        [0.0, 1.0, 0.0],  # 绿
        [0.0, 0.0, 1.0],  # 蓝
        [1.0, 1.0, 0.0],  # 黄
        [1.0, 0.0, 1.0],  # 品红
        [0.0, 1.0, 1.0],  # 青
        [1.0, 0.5, 0.0],  # 橙
        [0.5, 0.0, 1.0],  # 紫
        [0.0, 0.8, 0.2],  # 亮绿
        [0.8, 0.2, 0.0],  # 棕
        [0.0, 0.2, 0.8],  # 深蓝
        [0.5, 0.5, 0.0],  # 橄榄
        [0.3, 0.6, 0.0],  # 草绿
        [0.6, 0.0, 0.3],  # 紫红
        [0.0, 0.3, 0.6],  # 蓝绿
        [1.0, 0.3, 0.3],  # 粉红
    ], device='cuda')
    
    # 确保颜色足够
    if num_clusters > len(discrete_colors):
        repeat_times = (num_clusters // len(discrete_colors)) + 1
        discrete_colors = discrete_colors.repeat(repeat_times, 1)
    discrete_colors = discrete_colors[:num_clusters]
    
    # 随机打乱颜色顺序，增加邻近簇的颜色对比
    perm = torch.randperm(num_clusters)
    discrete_colors = discrete_colors[perm]
    
    # 为每个点分配颜色
    colors = discrete_colors[cluster_ids]
    
    # 可视化
    visualize(
        points=poses,
        point_colors=colors.cpu().numpy(),
        point_size=5.0,
        background_color=[0.1, 0.1, 0.1]
    )
    
    return cluster_ids

# 使用颜色梯度的方法，确保最大对比度
def visualize_with_color_gradients(poses):
    """
    使用空间坐标直接映射到颜色，确保相邻区域有不同颜色
    """
    if isinstance(poses, list) or (isinstance(poses, np.ndarray) and len(poses.shape) == 3):
        points = torch.tensor(poses[0], device='cuda')
    else:
        points = torch.tensor(poses, device='cuda')
    
    N = points.shape[0]
    
    # 计算点云的边界范围
    min_coords, _ = torch.min(points, dim=0)
    max_coords, _ = torch.max(points, dim=0)
    
    # 归一化坐标到[0,1]范围
    normalized_points = (points - min_coords) / (max_coords - min_coords + 1e-8)
    
    # 方法1: 使用空间坐标直接映射到RGB（产生渐变效果）
    # colors = normalized_points
    
    # 方法2: 使用正弦函数增加变化频率（产生条纹效果）
    freq = 8.0  # 控制条纹密度
    colors = torch.zeros((N, 3), device='cuda')
    colors[:, 0] = 0.5 + 0.5 * torch.sin(freq * normalized_points[:, 0] * 2 * torch.pi)  # R
    colors[:, 1] = 0.5 + 0.5 * torch.sin(freq * normalized_points[:, 1] * 2 * torch.pi)  # G
    colors[:, 2] = 0.5 + 0.5 * torch.sin(freq * normalized_points[:, 2] * 2 * torch.pi)  # B
    
    # 方法3: 使用空间坐标的组合计算颜色（产生更复杂的图案）
    # colors[:, 0] = 0.5 + 0.5 * torch.sin(freq * (normalized_points[:, 0] + normalized_points[:, 1]) * torch.pi)
    # colors[:, 1] = 0.5 + 0.5 * torch.sin(freq * (normalized_points[:, 1] + normalized_points[:, 2]) * torch.pi)
    # colors[:, 2] = 0.5 + 0.5 * torch.sin(freq * (normalized_points[:, 2] + normalized_points[:, 0]) * torch.pi)
    
    # 可视化
    visualize(
        points=poses,
        point_colors=colors.cpu().numpy(),
        point_size=5.0,
        background_color=[0.1, 0.1, 0.1]
    )
    
    return colors

# 结合空间位置和聚类的高对比度方法
def visualize_with_spatial_clusters(poses, num_clusters=24):
    """
    结合空间位置和聚类的高对比度可视化
    """
    if isinstance(poses, list) or (isinstance(poses, np.ndarray) and len(poses.shape) == 3):
        points = torch.tensor(poses[0], device='cuda')
    else:
        points = torch.tensor(poses, device='cuda')
    
    # 确保是contiguous的
    points = points.contiguous()
    N = points.shape[0]
    
    # 1. 空间划分 - 将点云空间划分为网格
    min_coords, _ = torch.min(points, dim=0)
    max_coords, _ = torch.max(points, dim=0)
    
    # 每个维度划分为3个区域，总共有3^3=27个区域
    grid_size = 3
    cell_size = (max_coords - min_coords) / grid_size
    
    # 计算每个点属于哪个网格单元
    cell_indices = torch.floor((points - min_coords) / cell_size).long()
    # 将3D网格单元索引转换为1D索引
    cell_id = cell_indices[:, 0] * grid_size * grid_size + cell_indices[:, 1] * grid_size + cell_indices[:, 2]
    
    # 2. 为每个网格单元分配对比鲜明的颜色
    num_cells = grid_size ** 3
    
    # 创建对比鲜明的颜色映射
    cell_colors = torch.zeros((num_cells, 3), device='cuda')
    
    # 使用HSV颜色空间生成对比鲜明的颜色
    for i in range(num_cells):
        h = (i * 0.618033988749895) % 1.0  # 黄金比例
        s = 0.8
        v = 0.9
        
        # 简化的HSV到RGB转换
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        cell_colors[i] = torch.tensor([r+m, g+m, b+m], device='cuda')
    
    # 3. 随机打乱颜色，确保空间相邻区域颜色差异更大
    perm = torch.randperm(num_cells)
    cell_colors = cell_colors[perm]
    
    # 4. 为每个点分配颜色
    colors = cell_colors[cell_id]
    
    # 可视化
    visualize(
        points=poses,
        point_colors=colors.cpu().numpy(),
        point_size=5.0,
        background_color=[0.1, 0.1, 0.1]
    )
    
    return colors

# 使用增强对比度版本
def visualize_with_enhanced_contrast(poses):
    """
    使用离散颜色方案可视化点云，最大化邻近点的对比度
    特别优化相邻簇之间的颜色对比
    """
    # 用空间聚类而不是KNN权重来区分区域
    if isinstance(poses, list) or (isinstance(poses, np.ndarray) and len(poses.shape) == 3):
        points = torch.tensor(poses[0], device='cuda')
    else:
        points = torch.tensor(poses, device='cuda')
    
    # 确保是contiguous的
    points = points.contiguous()
    
    # 用K-means进行聚类
    N = points.shape[0]
    num_clusters = 32  # 增加聚类数量以获得更细致的分割
    
    # 随机选择初始聚类中心
    if N < num_clusters:
        num_clusters = N
    
    # 随机初始化聚类中心，但使用更好的初始策略
    # 先选一个随机点作为第一个中心
    center_indices = [torch.randint(0, N, (1,), device='cuda').item()]
    centers = [points[center_indices[0]]]
    
    # 使用最大距离策略选择剩余中心点（k-means++思想）
    for i in range(1, num_clusters):
        # 计算每个点到所有已选中心的最小距离
        min_dists = torch.min(torch.stack([torch.sum((points - center)**2, dim=1) for center in centers]), dim=0)[0]
        
        # 选择距离最大的点作为下一个中心
        next_center_idx = torch.argmax(min_dists).item()
        center_indices.append(next_center_idx)
        centers.append(points[next_center_idx])
    
    centers = torch.stack(centers)
    
    # K-means迭代优化
    for _ in range(8):  # 增加迭代次数以获得更好的聚类
        # 计算每个点到每个中心的距离
        dists = torch.cdist(points, centers)  # (N, num_clusters)
        
        # 分配每个点到最近的中心
        cluster_ids = torch.argmin(dists, dim=1)  # (N,)
        
        # 更新中心
        for c in range(num_clusters):
            cluster_points = points[cluster_ids == c]
            if len(cluster_points) > 0:
                centers[c] = cluster_points.mean(dim=0)
    
    # 计算簇之间的邻接关系
    cluster_graph = torch.zeros((num_clusters, num_clusters), device='cuda')
    
    # 使用点的KNN关系建立簇之间的邻接关系
    knn_result = knn_points(points.unsqueeze(0), points.unsqueeze(0), K=15)
    knn_idx = knn_result.idx.squeeze(0)  # [N, K]
    
    # 统计簇之间的连接
    for i in range(N):
        cluster_i = cluster_ids[i]
        for j in knn_idx[i]:
            if i != j:
                cluster_j = cluster_ids[j]
                if cluster_i != cluster_j:
                    cluster_graph[cluster_i, cluster_j] += 1
                    cluster_graph[cluster_j, cluster_i] += 1
    
    # 定义更丰富的高对比度颜色集
    discrete_colors = torch.tensor([
        [1.00, 0.00, 0.00],  # 红
        [0.00, 1.00, 0.00],  # 绿
        [0.00, 0.00, 1.00],  # 蓝
        [1.00, 1.00, 0.00],  # 黄
        [1.00, 0.00, 1.00],  # 洋红
        [0.00, 1.00, 1.00],  # 青
        [1.00, 0.50, 0.00],  # 橙
        [0.50, 0.00, 1.00],  # 紫
        [0.00, 0.75, 0.25],  # 亮绿
        [0.75, 0.25, 0.00],  # 棕
        [0.25, 0.00, 0.75],  # 靛蓝
        [0.75, 0.75, 0.00],  # 亮黄
        [0.00, 0.50, 0.50],  # 蓝绿
        [0.50, 0.50, 0.00],  # 橄榄
        [0.50, 0.00, 0.50],  # 紫红
        [0.00, 0.00, 0.50],  # 海军蓝
        [0.50, 0.25, 0.00],  # 褐
        [0.25, 0.50, 0.00],  # 深绿
        [0.00, 0.25, 0.50],  # 靛青
        [0.50, 0.00, 0.25],  # 勃艮第
        [0.75, 0.50, 0.25],  # 沙黄
        [0.25, 0.75, 0.50],  # 薄荷
        [0.50, 0.25, 0.75],  # 薰衣草
        [0.75, 0.25, 0.50],  # 粉红
        [1.00, 0.75, 0.75],  # 浅粉
        [0.75, 1.00, 0.75],  # 浅绿
        [0.75, 0.75, 1.00],  # 浅蓝
        [0.25, 0.25, 0.25],  # 深灰
        [0.75, 0.75, 0.75],  # 浅灰
        [1.00, 1.00, 0.50],  # 浅黄
        [0.50, 1.00, 1.00],  # 浅青
        [1.00, 0.50, 1.00],  # 浅紫
    ], device='cuda')
    
    # 确保颜色足够
    if num_clusters > len(discrete_colors):
        repeat_times = (num_clusters // len(discrete_colors)) + 1
        discrete_colors = discrete_colors.repeat(repeat_times, 1)
    discrete_colors = discrete_colors[:num_clusters]
    
    # 基于图着色算法分配颜色，确保相邻簇颜色最大化差异
    color_assignment = torch.zeros(num_clusters, dtype=torch.long, device='cuda') - 1
    
    # 计算每个簇的连接度（邻居数量）
    cluster_degree = torch.sum(cluster_graph > 0, dim=1)
    
    # 按照连接度从高到低的顺序处理簇
    sorted_clusters = torch.argsort(cluster_degree, descending=True)
    
    for cluster in sorted_clusters:
        # 找出已分配颜色的邻居
        neighbors = torch.where(cluster_graph[cluster] > 0)[0]
        neighbor_colors = torch.tensor([color_assignment[n].item() if color_assignment[n] >= 0 else -1 
                                       for n in neighbors], device='cuda')
        neighbor_colors = neighbor_colors[neighbor_colors >= 0]
        
        # 选择与邻居颜色差异最大的颜色
        if len(neighbor_colors) == 0:
            # 如果没有邻居，选择第一个颜色
            color_assignment[cluster] = 0
        else:
            # 计算所有颜色与所有邻居颜色的差异
            all_colors = torch.arange(len(discrete_colors), device='cuda')
            
            # 创建一个差异度量来选择最佳颜色
            best_diff = -1
            best_color = 0
            
            for color_idx in all_colors:
                if color_idx not in neighbor_colors:
                    # 计算与邻居颜色的差异
                    this_color = discrete_colors[color_idx]
                    diff_sum = 0
                    for n_color in neighbor_colors:
                        neighbor_color = discrete_colors[n_color]
                        # 计算颜色差异（欧氏距离）
                        diff = torch.sqrt(torch.sum((this_color - neighbor_color) ** 2))
                        diff_sum += diff
                    
                    # 取平均差异
                    avg_diff = diff_sum / len(neighbor_colors) if len(neighbor_colors) > 0 else diff_sum
                    
                    if avg_diff > best_diff:
                        best_diff = avg_diff
                        best_color = color_idx
            
            color_assignment[cluster] = best_color
    
    # 应用颜色分配
    optimized_colors = discrete_colors[color_assignment]
    
    # 为每个点分配颜色
    colors = optimized_colors[cluster_ids]
    
    # 可视化
    visualize(
        points=poses,
        point_colors=colors.cpu().numpy(),
        point_size=5.0,
        background_color=[0.05, 0.05, 0.05]  # 更暗的背景增强对比度
    )
    
    return cluster_ids, color_assignment