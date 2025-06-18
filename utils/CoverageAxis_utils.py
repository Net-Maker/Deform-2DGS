import os
import torch
import trimesh
import numpy as np
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from utils.skeletonizer import *
from utils.compute_adj import *
import pygeodesic.geodesic as geodesic


# 相机视角配置数组
CAMERA_POSITIONS = {
    "front": {
        "front": [0.0, 0.0, -1.0],  # 正面朝前
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, -1.0, 0.0],    # 图像正立
        "zoom": 0.8
    },
    "back": {
        "front": [0.0, 0.0, 1.0],   # 从背面看
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, -1.0, 0.0],    # 保持图像正立
        "zoom": 0.8
    },
    "left": {
        "front": [-1.0, 0.0, 0.0],  # 从左侧看
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, -1.0, 0.0],    # 保持图像正立
        "zoom": 0.8
    },
    "right": {
        "front": [1.0, 0.0, 0.0],   # 从右侧看
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, -1.0, 0.0],    # 保持图像正立
        "zoom": 0.8
    },
    "top": {
        "front": [0.0, -1.0, 0.0],  # 从上方看
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],     # 保持Z轴向前，与front视角一致
        "zoom": 0.8
    },
    "bottom": {
        "front": [0.0, 1.0, 0.0],   # 从下方看
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, 0.0, -1.0],    # 保持Z轴向后，与front视角一致
        "zoom": 0.8
    },
    "isometric": {
        "front": [-1.0, -1.0, -1.0],  # 等轴测视图
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, -1.0, 0.0],    # 保持图像正立
        "zoom": 0.8
    }
}


class CoverageAxisSolver:
    def __init__(self,
                 surface_sample_num=3000,
                 inner_points_method="random",
                 random_sample_number=500000,
                 max_iter=200,
                 reg_radius=1,
                 reg=1,
                 output_path=None):
        """
        初始化CoverageAxis求解器
        
        参数:
            surface_sample_num: 表面采样点数量
            dilation: 膨胀系数
            inner_points_method: 内部点生成方法 ["random", "voronoi"]
            random_sample_number: 随机采样点数量
            max_iter: 最大迭代次数
            reg_radius: 半径正则化系数
            reg: 距离正则化系数
        """
        self.surface_sample_num = surface_sample_num
        self.inner_points_method = inner_points_method
        self.random_sample_number = random_sample_number
        self.max_iter = max_iter
        self.reg_radius = reg_radius
        self.reg = reg
        self.output_path = output_path
    @staticmethod
    def _multi_indexing(index: torch.Tensor, shape: torch.Size, dim=-2):
        shape = list(shape)
        back_pad = len(shape) - index.ndim
        for _ in range(back_pad):
            index = index.unsqueeze(-1)
        expand_shape = shape
        expand_shape[dim] = -1
        return index.expand(*expand_shape)

    @staticmethod
    def _multi_gather(values: torch.Tensor, index: torch.Tensor, dim=-2):
        return values.gather(dim, CoverageAxisSolver._multi_indexing(index, values.shape, dim))

    @staticmethod
    def _winding_number(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Parallel implementation of the Generalized Winding Number of points on the mesh
        O(n_points * n_faces) memory usage, parallelized execution
        1. Project tris onto the unit sphere around every points
        2. Compute the signed solid angle of the each triangle for each point
        3. Sum the solid angle of each triangle
        Parameters
        ----------
        pts    : torch.Tensor, (n_points, 3)
        verts  : torch.Tensor, (n_verts, 3)
        faces  : torch.Tensor, (n_faces, 3)
        This implementation is also able to take a/multiple batch dimension
        """
        # projection onto unit sphere: verts implementation gives a little bit more performance
        uv = verts[..., None, :, :] - pts[..., :, None, :]  # n_points, n_verts, 3
        uv = uv / uv.norm(dim=-1, keepdim=True)  # n_points, n_verts, 3

        # gather from the computed vertices (will result in a copy for sure)
        expanded_faces = faces[..., None, :, :].expand(*faces.shape[:-2], pts.shape[-2], *faces.shape[-2:])  # n_points, n_faces, 3

        u0 = CoverageAxisSolver._multi_gather(uv, expanded_faces[..., 0])  # n, f, 3
        u1 = CoverageAxisSolver._multi_gather(uv, expanded_faces[..., 1])  # n, f, 3
        u2 = CoverageAxisSolver._multi_gather(uv, expanded_faces[..., 2])  # n, f, 3

        e0 = u1 - u0  # n, f, 3
        e1 = u2 - u1  # n, f, 3
        del u1

        # compute solid angle signs
        sign = (torch.cross(e0, e1) * u2).sum(dim=-1).sign()

        e2 = u0 - u2
        del u0, u2

        l0 = e0.norm(dim=-1)
        del e0

        l1 = e1.norm(dim=-1)
        del e1

        l2 = e2.norm(dim=-1)
        del e2

        # compute edge lengths: pure triangle
        l = torch.stack([l0, l1, l2], dim=-1)  # n_points, n_faces, 3

        # compute spherical edge lengths
        l = 2 * (l/2).arcsin()  # n_points, n_faces, 3

        # compute solid angle: preparing: n_points, n_faces
        s = l.sum(dim=-1) / 2
        s0 = s - l[..., 0]
        s1 = s - l[..., 1]
        s2 = s - l[..., 2]

        # compute solid angle: and generalized winding number: n_points, n_faces
        eps = 1e-10  # NOTE: will cause nan if not bigger than 1e-10
        solid = 4 * (((s/2).tan() * (s0/2).tan() * (s1/2).tan() * (s2/2).tan()).abs() + eps).sqrt().arctan()
        signed_solid = solid * sign  # n_points, n_faces

        winding = signed_solid.sum(dim=-1) / (4 * torch.pi)  # n_points

        return winding

    @staticmethod
    def _compute_min_distances(X, selected_pts):
        """计算最小距离"""
        distances = np.linalg.norm(X[:, np.newaxis] - selected_pts, axis=2)
        min_distances = np.min(distances, axis=1)
        return min_distances

    def _heuristic_alg(self, D, candidate, radius_list, threshold=0.95, penalty='stand'):
        """启发式算法求解覆盖问题"""
        m, n = D.shape
        S = np.arange(m)
        A = []
        grade = []
        pbar = tqdm(range(self.max_iter))
        
        for i in pbar:
            score = np.sum(D[S], axis=0).astype(float)
            
            # 如果score全为0，说明无法继续优化，直接退出
            if np.all(score == 0):
                break
            
            # 修改标准化计算逻辑
            score_std = np.std(score, ddof=1)
            if score_std > 1e-10:  # 只在标准差不为0时进行标准化
                score = (score - np.mean(score)) / score_std
            else:
                score = np.zeros_like(score)  # 如果所有值相同，则所有标准化分数都为0
                
            if len(A) > 0:
                loss = self._compute_min_distances(candidate, candidate[A])
                loss_std = np.std(loss, ddof=1)
                if loss_std > 1e-10:
                    loss = (loss - np.mean(loss)) / loss_std
                else:
                    loss = np.zeros_like(loss)
                score += self.reg * loss
                
            if penalty == 'stand':
                loss_radius = 1 / radius_list
                radius_std = np.std(loss_radius, ddof=1)
                if radius_std > 1e-10:
                    loss_radius = (loss_radius - np.mean(loss_radius)) / radius_std
                else:
                    loss_radius = np.zeros_like(loss_radius)
            else:
                radius_max = np.max(radius_list)
                loss_radius = 0.1 * radius_max / radius_list
                
            score -= self.reg_radius * loss_radius
            i_k = np.argmax(score)
            A.append(i_k)
            grade.append(score[i_k])
            S = S[D[S, i_k] == 0]
            
            pbar.set_description(f'Coverage rate: {1 - len(S) / m:.4f}')
            
            # 如果已经达到完全覆盖或目标阈值，则退出
            if len(S) == 0 or (1 - len(S) / m) > threshold:
                break
                
        coverage_rate = len(S) / m
        A = np.array(A)
        return A, grade, coverage_rate

    def solve(self, mesh, threshold=1, dilation=0.05):
        """
        计算mesh的Coverage Axis
        
        参数:
            mesh: trimesh.Trimesh对象
            
        返回:
            selected_points: numpy数组,选择的内部点坐标 (N, 3)
            selected_radius: numpy数组,对应的半径值 (N, 1) 
            coverage_rate: float,覆盖率
        """
        # 获取mesh信息
        mesh_faces = np.array(mesh.faces)
        mesh_vertices = np.array(mesh.vertices)
        
        point_set = trimesh.sample.sample_surface(mesh, self.surface_sample_num)[0]
        
        # 归一化
        min_coords = np.min(mesh_vertices, axis=0)
        max_coords = np.max(mesh_vertices, axis=0)
        scale = np.max(max_coords - min_coords)
        #self.scale = scale
        mesh_vertices = (mesh_vertices - min_coords) / scale
        point_set = (point_set - min_coords) / scale
        
        # 生成内部点
        if self.inner_points_method == "random":
            if os.path.exists(os.path.join(self.output_path, 'random_inner_points.npy')):
                inner_points = np.load(os.path.join(self.output_path, 'random_inner_points.npy'))
            else:
                inner_points = self._generate_random_inner_points(mesh_vertices, mesh_faces)
                if os.path.exists(self.output_path):
                    np.save(os.path.join(self.output_path, 'random_inner_points.npy'), inner_points)
                else:
                    os.makedirs(self.output_path, exist_ok=True)
                    np.save(os.path.join(self.output_path, 'random_inner_points.npy'), inner_points)
            # visualize_spheres(inner_points, np.ones(inner_points.shape[0]) * self.dilation )
        else:
            raise NotImplementedError("目前只支持random方法生成内部点")
        
        # 计算半径和构建覆盖矩阵
        inner_points_g = torch.tensor(inner_points).cuda().double()
        point_set_g = torch.tensor(point_set).cuda().double()
        D, radius_list, radius_ori = self._compute_coverage_matrix(inner_points_g, point_set_g, dilation)
        
        # 启发式算法求解
        value_pos, grade, coverage_rate = self._heuristic_alg(D, inner_points_g.cpu().numpy(), radius_list, threshold)
        
        # 获取结果并反归一化
        selected_points = inner_points[value_pos] * scale + min_coords
        selected_radius = radius_ori[value_pos] * scale
        
        # self.seleted_points = selected_points
        # self.seleted_radius = np.reshape(selected_radius, -1)
        # self.coverage_rate = 1 - coverage_rate
        return selected_points, selected_radius, coverage_rate, scale
        print("NOTE:======  CoverageAxis result: selected_points_shape: {}, selected_radius_shape: {}, coverage_rate: {}  ======".format(selected_points.shape, selected_radius.shape, 1 - coverage_rate))

    def _generate_random_inner_points(self, mesh_vertices, mesh_faces):
        """生成随机内部点"""
        print("NOTE:======  CoverageAxis:Generating random inner points  ======")
        min_x, min_y, min_z = np.min(mesh_vertices, axis=0)
        max_x, max_y, max_z = np.max(mesh_vertices, axis=0)
        
        P_x = (max_x) * np.random.random((self.random_sample_number, 1)) * 1.3 + min_x - 0.1
        P_y = (max_y) * np.random.random((self.random_sample_number, 1)) * 1.3 + min_y - 0.1
        P_z = (max_z) * np.random.random((self.random_sample_number, 1)) * 1.3 + min_z - 0.1
        P = np.concatenate((P_x, P_y, P_z), axis=1)
        
        winding_con = []
        for i in tqdm(range(0, len(P), 5000)):
            start = i
            end = i + 5000
            winding = self._winding_number(torch.tensor(P[start:end, :]).cuda().double(),
                                       torch.tensor(mesh_vertices).cuda().double(),
                                       torch.tensor(mesh_faces).cuda().long())
            winding_con.append(winding)
        winding_con = torch.cat(winding_con, dim=0)
        return P[winding_con.cpu().numpy() > 0.5]

    def _compute_coverage_matrix(self, inner_points_g, point_set_g, dilation):
        """计算覆盖矩阵"""
        dist = torch.cdist(inner_points_g, point_set_g, p=2)
        radius = dist.topk(1, largest=False).values
        radius_ori = radius.cpu().numpy()
        radius = radius + dilation
        radius_list = np.reshape(radius_ori, -1)
        
        radius_g = radius[:, 0].unsqueeze(0).repeat(len(point_set_g), 1)
        D = torch.cdist(point_set_g, inner_points_g, p=2)
        D = torch.gt(radius_g, D).type(torch.int)
        
        return D.cpu().numpy(), radius_list, np.reshape(radius_ori, -1)

    def save_CoverageAxis_results(self, selected_points, selected_radius, coverage_rate, scale):
        """保存结果到文件"""
        self.seleted_points = selected_points
        self.seleted_radius = selected_radius
        self.coverage_rate = coverage_rate
        self.scale = scale
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存点云
        points_mesh = trimesh.points.PointCloud(self.seleted_points)
        points_mesh.export(os.path.join(self.output_path, 'selected_points.obj'))
        
        # 保存结果文本
        result_txt = os.path.join(self.output_path, 'coverage_axis_result.txt')
        with open(result_txt, 'w') as f:
            f.write(f'Coverage Rate: {self.coverage_rate}\n')
            f.write('Point_X Point_Y Point_Z Radius\n')
            for point, radius in zip(self.seleted_points, self.seleted_radius):
                f.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {radius:.6f}\n')

    @staticmethod
    def read_results(result_txt_path):
        """读取保存的结果文件"""
        points = []
        radii = []
        coverage_rate = None
        
        with open(result_txt_path, 'r') as f:
            lines = f.readlines()
            coverage_rate = float(lines[0].split(':')[1].strip())
            for line in lines[2:]:
                x, y, z, r = map(float, line.strip().split())
                points.append([x, y, z])
                radii.append([r])
        
        return np.array(points), np.array(radii), coverage_rate

    def vis_spheres(self, radius, seleted_points=None, output_path=None, camera_position=None):
        """Visualize spheres using Open3D"""
        if seleted_points is None:
            visualize_spheres(self.seleted_points, radius, output_path, camera_position)
        else:
            visualize_spheres(seleted_points, radius, output_path, camera_position)

    def vis_connections(self, adjacency, seleted_points=None, output_path=None, camera_position=None):
        if seleted_points is None:
            visualize_connections(self.seleted_points, adjacency, output_path, camera_position)
        else:
            visualize_connections(seleted_points, adjacency, output_path, camera_position)

    def get_motion_magnitudes(self, poses):
        """计算每个点在时序上的运动幅度"""
        motion_magnitudes = []
        for point_idx in range(self.seleted_points.shape[0]):
            distances = cdist([self.seleted_points[point_idx]], poses[0])
            nearest_idx = np.argmin(distances)
            trajectory = poses[:, nearest_idx, :]
            displacement = np.max(np.linalg.norm(trajectory - trajectory[0], axis=1))
            motion_magnitudes.append(displacement)
        return np.array(motion_magnitudes)

    def get_geodesic_distance(self, points1, points2, mesh):
        """
        计算两个点集之间的测地线距离
        
        参数:
            points1: numpy数组，第一个点集，shape为(N1, 3)
            points2: numpy数组，第二个点集，shape为(N2, 3)
            mesh: trimesh.Trimesh对象，用于计算测地线距离的网格
            
        返回:
            distances: numpy数组，shape为(N1, N2)，表示两个点集之间的测地线距离
        """
        # 确保mesh是trimesh对象
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
        
        # 初始化距离矩阵
        distances = np.zeros((len(points1), len(points2)))
        
        # 使用pygeodesic计算测地线距离
        geo_solver = geodesic.PyGeodesicAlgorithmExact(mesh.vertices, mesh.faces)
        
        # 对每个点对计算测地线距离
        for i in tqdm(range(len(points1)), desc="计算测地线距离"):
            for j in range(len(points2)):
                # 找到网格上最近的点
                dist_to_vertices = np.linalg.norm(mesh.vertices - points1[i], axis=1)
                closest_vertex_idx1 = np.argmin(dist_to_vertices)
                
                dist_to_vertices = np.linalg.norm(mesh.vertices - points2[j], axis=1)
                closest_vertex_idx2 = np.argmin(dist_to_vertices)
                
                # 计算测地线距离
                distance, _ = geo_solver.geodesicDistance(closest_vertex_idx1, closest_vertex_idx2)
                distances[i, j] = distance
        
        return distances

    def init_graph(self, poses, mesh, dilation, visualize=True, geodesic_threshold=2):
        """
        初始化图结构，区分运动和静止的点
        
        参数:
            poses: numpy数组，shape为(T, N, 3)，T为时间序列长度，N为点数
            mesh: 网格对象
            dilation: 膨胀系数
            visualize: 是否可视化结果
            geodesic_threshold: 测地线距离与欧氏距离的比值阈值，小于此值认为两点应该相连
        """
        print("NOTE:======  CoverageAxis:Computing adjacency matrix from distance and shape  ======")
        self.dist_matrix = cdist(self.seleted_points, self.seleted_points)
        # self.geodesic_dist_matrix = self.get_geodesic_distance(self.seleted_points, self.seleted_points, mesh)
        radius = self.seleted_radius + np.ones(self.seleted_points.shape[0]) * (dilation * self.scale)
        
        # 计算基于距离的邻接矩阵
        # coarse_inner_points = self.seleted_points

        moving_points = self.seleted_points[self.valid_joint_mask]
        moving_radius = radius[self.valid_joint_mask]
        moving_dist_matrix = self.dist_matrix[self.valid_joint_mask][:, self.valid_joint_mask]

        self.dist_adjacency = compute_adjacency_with_dist(moving_points, moving_radius, mesh, moving_dist_matrix, poses[0], geodesic_threshold=geodesic_threshold)

        self.mat_adjacency = get_adjacency_from_MAT()
        
        # 可视化结果
        if visualize:

            camera_position_top = CAMERA_POSITIONS["top"] # for debug process, check the Line 10 for more positions
            camera_position_side = CAMERA_POSITIONS["left"]
            camera_position_front = CAMERA_POSITIONS["front"]
            
            # 可视化膨胀球体
            self.vis_spheres(radius, output_path=self.output_path+"/spheres_dilated_top.png", camera_position=camera_position_top)
            self.vis_spheres(radius, output_path=self.output_path+"/spheres_dilated_side.png", camera_position=camera_position_side)
            self.vis_spheres(radius, output_path=self.output_path+"/spheres_dilated_front.png", camera_position=camera_position_front)
            # 可视化原始球体
            self.vis_spheres(self.seleted_radius, output_path=self.output_path+"/spheres_ori_side.png", camera_position=camera_position_side)
            self.vis_spheres(self.seleted_radius, output_path=self.output_path+"/spheres_ori_front.png", camera_position=camera_position_front)
            self.vis_spheres(self.seleted_radius, output_path=self.output_path+"/spheres_ori_top.png", camera_position=camera_position_top)
            # 可视化连接
            self.vis_connections(self.dist_adjacency, output_path=self.output_path+"/connections_side.png", camera_position=camera_position_side, seleted_points=moving_points)
            self.vis_connections(self.dist_adjacency, output_path=self.output_path+"/connections_front.png", camera_position=camera_position_front, seleted_points=moving_points)
            self.vis_connections(self.dist_adjacency, output_path=self.output_path+"/connections_top.png", camera_position=camera_position_top, seleted_points=moving_points)
            
            # 交互可视化
            self.vis_spheres(radius, seleted_points=moving_points)
            self.vis_connections(self.dist_adjacency, seleted_points=moving_points)

            # 可视化骨架
            self.skeleton_pure_shape()
        return self.dist_adjacency, self.moving_mask

    def get_moving_mask(self, poses, motion_threshold=0.01):
        """
        根据运动幅度计算运动点
        """
        motion_magnitudes = self.get_motion_magnitudes(poses)
        self.moving_mask = motion_magnitudes > motion_threshold
        self.moving_joints = self.seleted_points[self.moving_mask]
        self.static_joints = self.seleted_points[~self.moving_mask]
        self.valid_joint_mask = self.moving_mask.copy()  # 初始化valid_joint_mask，与moving_mask相同
        print(f"初始化valid_joint_mask: 有{np.sum(self.valid_joint_mask)}个有效关节")
        return self.moving_mask


    def update_after_bone_removal(self, bone_transforms, vert_assignments, removed_bones):
        """
        只负责同步kept_bones和valid_joint_mask
        """
        num_bones = bone_transforms.shape[0]
        kept_bones = np.array([i for i in range(num_bones) if i not in removed_bones])

        # 更新骨骼变换
        updated_bone_transforms = bone_transforms[kept_bones]
        updated_vert_assignments = vert_assignments  # 不再做重新分配

        # 更新valid_joint_mask
        if hasattr(self, 'moving_mask') and hasattr(self, 'valid_joint_mask'):
            moving_indices = np.where(self.moving_mask)[0]
            for bone_idx in removed_bones:
                if bone_idx < len(moving_indices):
                    original_idx = moving_indices[bone_idx]
                    self.valid_joint_mask[original_idx] = False
            self.vaild_joints = self.seleted_points[self.valid_joint_mask]

        print(f"更新后的骨骼数量: {len(kept_bones)}")
        print(f"被移除的骨骼: {removed_bones}")
        return updated_bone_transforms, updated_vert_assignments, kept_bones

    def vertice_assignments(self, rest_pose, poses, visualize=True, motion_threshold=0.01, output_path=None):
        """
        根据CoverageAxis的结果计算每个点对应的骨骼索引
        """
        if os.path.exists(os.path.join(output_path, "vertice_assignments.npy")):
            result = np.load(os.path.join(output_path, "vertice_assignments.npy"), allow_pickle=True).item()
            self.vert_assignments = result["vert_assignments"]
            self.bone_transforms = result["bone_transforms"]
            if "valid_joint_mask" in result and result["valid_joint_mask"] is not None:
                self.valid_joint_mask = result["valid_joint_mask"]
                print(f"从文件加载valid_joint_mask，有{np.sum(self.valid_joint_mask)}个有效关节")
                self.vaild_joints = self.seleted_points[self.valid_joint_mask]
            #self.standard_bone_transforms = convert_to_standard_bone_transform(self.bone_transforms, rest_pose)
            # bone_transforms shape: (B, T, 4, 3)
            # 打印vert_assignments中包含的骨骼索引
            return


        from utils.ssdr_utils import init_bone_label
        bone_transforms, vert_assignments, empty_bones = init_bone_label(rest_pose, poses, self.moving_joints)
        # bone_transforms shape: (B, T, 4, 3) empty_bones的索引是相对于moving_joints的索引
        
        # 更新顶点分配和骨骼变换，处理已删除的骨骼
        self.bone_transforms, self.vert_assignments, kept_bones = self.update_after_bone_removal(bone_transforms, vert_assignments, empty_bones)
        self.moving_joints = self.moving_joints[kept_bones] # 更新保留的骨骼
        
        # 设置有效关节
        if hasattr(self, 'valid_joint_mask'):
            self.vaild_joints = self.seleted_points[self.valid_joint_mask]
        
        
        #self.standard_bone_transforms = convert_to_standard_bone_transform(bone_transforms, rest_pose)
        # shape: (B, T, 4, 4)


        if visualize:
            N = self.vert_assignments.shape[0]  # 顶点数量
            B = len(self.moving_joints)  # 骨骼数量
                
            # 创建one-hot编码的权重矩阵
            weights_one_hot = np.zeros((N, B))
            for i in range(N):
                bone_idx = self.vert_assignments[i]
                if bone_idx < B:  # 确保索引不越界
                    weights_one_hot[i, bone_idx] = 1.0
            
            # 使用动态视角参数
            if hasattr(self, 'vaild_joints') and len(self.vaild_joints) > 0:
                skeleton_points = self.vaild_joints
                root = self.vaild_joints[0] if len(self.vaild_joints) > 0 else self.seleted_points[0]
            else:
                skeleton_points = self.moving_joints
                root = self.moving_joints[0] if len(self.moving_joints) > 0 else self.seleted_points[0]
                
            visualise_skeletonizer(
                skeleton_points=skeleton_points, 
                root=root, 
                joints=skeleton_points, 
                bones=skeleton_points, 
                pcd=rest_pose, 
                weights=weights_one_hot,
                add_labels=False,
                add_vaild_bones=False,
                bone_width=20.0,
                point_size=2.0,
                joint_size=10.0,
                bone_color=np.array([0.1, 0.1, 0.1])
            )
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            result = {
                "vert_assignments": self.vert_assignments,
                "bone_transforms": self.bone_transforms,
                "empty_bones": empty_bones,
                "valid_joint_mask": self.valid_joint_mask if hasattr(self, 'valid_joint_mask') else None
                #"standard_bone_transforms": self.standard_bone_transforms
            }
            np.save(os.path.join(output_path, "vertice_assignments.npy"), result)

    def skeleton_from_adjacency(self, matrix, adjacency):
        """
        根据邻接矩阵计算skeleton
        """
        distance_graph = matrix * adjacency
        D = shortest_path(distance_graph, directed=True, method='FW')
        root_indx = D.sum(1).argmin()
        graph = adjacency_to_graph(distance_graph)
        joints_index, bones = bfs(graph, root_indx, 0.01)
        bones, joints_index = post_process_bones(bones, joints_index, self.seleted_points, remove_redundant=True)
        # bones = build_minimum_spanning_tree(self.seleted_points[joints_index], bones, root_indx, self.seleted_points)
        return joints_index, root_indx, bones

    def skeleton_pure_shape(self, visualize=True):
        """
        只根据CoverageAXis的结果计算skeleton，不考虑运动，用于可视化和对比图
        """
        print("NOTE:======  Skeletonization:Computing Skeleton fromdistance graph  ======")
        joints_index, root_indx, bones = self.skeleton_from_adjacency(self.dist_matrix, self.dist_adjacency)
        if visualize:
            # visualization函数参数：
            # skeleton_points: 骨架点
            # root: 根节点
            # joints: 关节点
            # bones: 骨骼
            # pcd: 点云
            # weights: 权重
            visualise_skeletonizer(
                self.seleted_points, 
                self.seleted_points[root_indx], 
                self.seleted_points[joints_index], 
                bones, 
                self.seleted_points, 
                np.zeros_like(self.seleted_points))

    def skeleton_with_motion(self, poses, visualize=True):
        """
        基于运动分析结果计算skeleton，考虑运动相似性和半径影响
        
        参数:
            poses: numpy数组，shape为(T, N, 3)，时序点云序列
            visualize: 是否可视化结果
        """
        print("NOTE:======  CoverageAxis:Computing weighted graph for skeleton extraction  ======")
        
        # 获取运动点及其属性，并考虑valid_joint_mask
        if hasattr(self, 'valid_joint_mask'):
            # 使用有效的运动点进行计算
            valid_moving_mask = np.logical_and(self.moving_mask, self.valid_joint_mask)
            print(f"使用valid_joint_mask: 共{np.sum(valid_moving_mask)}个有效运动节点")
            moving_points = self.seleted_points[valid_moving_mask]
            
            # dist_adjacency和相关矩阵都是基于valid_joint_mask计算的
            # 所以我们直接使用，但需要确保moving_dist_matrix也是基于相同点集
            moving_dist_matrix = self.dist_matrix[self.valid_joint_mask][:, self.valid_joint_mask]
            moving_adjacency = self.dist_adjacency  # 直接使用，已经是31x31
            moving_radius = self.seleted_radius[self.valid_joint_mask]
            
            # 获取有效运动点在原始moving_points中的索引
            valid_indices = np.where(self.valid_joint_mask[self.moving_mask])[0]
            
            # 检查骨骼变换矩阵的维度是否与moving_points匹配
            # bone_transforms shape: (B, T, 4, 3)
            if self.bone_transforms.shape[0] != len(valid_indices):
                print(f"警告: 骨骼变换矩阵维度 ({self.bone_transforms.shape[0]}) 与有效运动点数量 ({len(valid_indices)}) 不匹配")
                print(f"使用有效运动点子集: {valid_indices}")
                
                # 如果骨骼变换矩阵已经是基于moving_joints，需要根据valid_indices选择对应部分
                if self.bone_transforms.shape[0] == np.sum(self.moving_mask):
                    self.bone_transforms = self.bone_transforms[valid_indices]
        else:
            # 原始逻辑
            moving_points = self.seleted_points[self.moving_mask]
            moving_dist_matrix = self.dist_matrix[self.moving_mask][:, self.moving_mask]
            moving_adjacency = self.dist_adjacency
            moving_radius = self.seleted_radius[self.moving_mask]
            
            # 检查骨骼变换矩阵的维度是否与moving_points匹配
            if self.bone_transforms.shape[0] != len(moving_points):
                print(f"警告: 骨骼变换矩阵维度 ({self.bone_transforms.shape[0]}) 与运动点数量 ({len(moving_points)}) 不匹配")
                print("这可能是因为某些骨骼已被移除。使用现有的骨骼变换矩阵...")
        
        # 构建带权图
        weighted_distance_graph = moving_dist_matrix.copy()
        
        num_motion_points = len(moving_points)
        
        # 计算变换矩阵的相似性只用于实际骨骼的数量
        valid_bone_count = min(self.bone_transforms.shape[0], num_motion_points)
        
        for i in range(valid_bone_count):
            for j in range(valid_bone_count):
                if i < num_motion_points and j < num_motion_points and moving_adjacency[i, j] == 1:
                    # 1. 计算运动相似性权重
                    transform_i = self.bone_transforms[i]  # (T, 4, 3)
                    transform_j = self.bone_transforms[j]  # (T, 4, 3)
                    
                    # 计算运动方向的相似度 - 使用旋转部分(前3行)
                    motion_sim = np.mean(np.sum(transform_i[:, :3, :] * transform_j[:, :3, :], axis=(1,2)) / 
                                      (np.linalg.norm(transform_i[:, :3, :], axis=(1,2)) * 
                                       np.linalg.norm(transform_j[:, :3, :], axis=(1,2)) + 1e-6))
                    motion_sim_normalized = (motion_sim + 1) / 2  # 映射到[0,1]
                    
                    # 2. 计算半径权重
                    radius_weight = 1.0 / (0.1 + 0.9 * (moving_radius[i] + moving_radius[j]) / 2)
                    
                    # 合并权重：运动相似度越高，权重越小；半径越大，权重越小
                    weighted_distance_graph[i,j] *= (1.0 + motion_sim_normalized) * radius_weight
        
        print("NOTE:======  Skeletonization:Computing skeleton from weighted graph  ======")
        joints_index, root_indx, bones = self.skeleton_from_adjacency(weighted_distance_graph, moving_adjacency)
        
        if visualize:
            visualise_skeletonizer(
                moving_points,
                moving_points[root_indx],
                moving_points[joints_index],
                bones,
                moving_points,
                np.zeros_like(moving_points),
            )
            self.visualize_weighted_graph(weighted_distance_graph, moving_points, moving_adjacency)
        
        return moving_points, bones, joints_index, root_indx

    def bfs_with_weights(self, graph, weighted_distance_graph, root_indx):
        """
        基于权重的广度优先搜索来构建骨架，考虑邻接矩阵和连通性
        """
        print("NOTE:======  CoverageAxis:Running weighted BFS  ======")
        visited = set()
        joints = [root_indx]  # 初始只包含根节点
        bones = []  # 存储骨骼连接关系
        visited.add(root_indx)
        
        # 使用优先队列存储(权重, 当前节点, 父节点)
        from queue import PriorityQueue
        queue = PriorityQueue()
        
        # 获取root的直接相连节点 - 使用weighted_distance_graph而不是self.dist_adjacency
        neighbours = np.where(weighted_distance_graph[root_indx] > 0)[0]
        for n in neighbours:
            queue.put((weighted_distance_graph[root_indx, n], n, root_indx))
        
        while not queue.empty():
            weight, current, parent = queue.get()
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # 使用weighted_distance_graph检查连接
            connected_nodes = np.where(weighted_distance_graph[current] > 0)[0]
            unvisited_connected = [n for n in connected_nodes if n not in visited]
            
            is_joint = len(connected_nodes) > 2 or len(unvisited_connected) == 0
            
            if is_joint:
                parent_joint_idx = -1
                for i, joint in enumerate(joints):
                    if joint == parent:
                        parent_joint_idx = i
                        break
                    
                if parent_joint_idx != -1:
                    bones.append([parent_joint_idx, len(joints)])
                    joints.append(current)
                    
                    for next_node in unvisited_connected:
                        if next_node not in visited:
                            queue.put((weighted_distance_graph[current, next_node], 
                                     next_node, current))
            else:
                for next_node in unvisited_connected:
                    queue.put((weighted_distance_graph[current, next_node], 
                             next_node, parent))
        
        print("NOTE:======  CoverageAxis:BFS found {} joints and {} bones  ======".format(
            len(joints), len(bones)))
        return joints, bones

    def visualize_weighted_graph(self, weighted_graph, points, adjacency, output_path=None, camera_position=None):
        """
        可视化带权图，边的颜色从红色(权重小)渐变到蓝色(权重大)
        使用圆柱体表示边，便于观察
        
        参数:
            weighted_graph: numpy数组，带权图的权重矩阵
            points: numpy数组，点的坐标 (N, 3)
            adjacency: numpy数组，邻接矩阵
            output_path: str，保存图片的路径
            camera_position: dict，相机位置配置
        """
        print("NOTE:======  CoverageAxis:Visualizing weighted graph with gradient colors  ======")
        
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 根据连接数设置点的大小（颜色）
        connection_counts = np.sum(adjacency, axis=1)
        max_connections = np.max(connection_counts)
        normalized_counts = connection_counts / max_connections
        point_colors = np.zeros((len(points), 3))
        point_colors[:, 0] = normalized_counts  # 根据连接数设置红色通道
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        # 获取权重的最大值和最小值用于归一化
        weights = weighted_graph[weighted_graph > 0]
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # 添加点云
        vis.add_geometry(pcd)
        
        # 遍历所有边，创建圆柱体
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if adjacency[i,j] == 1:
                    # 获取两个端点
                    point1 = points[i]
                    point2 = points[j]
                    
                    # 计算圆柱体参数
                    direction = point2 - point1
                    length = np.linalg.norm(direction)
                    direction = direction / length
                    
                    # 计算归一化权重
                    weight = weighted_graph[i,j]
                    normalized_weight = (weight - min_weight) / (max_weight - min_weight)
                    
                    # 创建圆柱体
                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                        radius=0.01,  # 圆柱体半径
                        height=length  # 圆柱体长度
                    )
                    
                    # 计算旋转矩阵
                    z_axis = np.array([0, 0, 1])
                    rotation_axis = np.cross(z_axis, direction)
                    rotation_angle = np.arccos(np.dot(z_axis, direction))
                    
                    if np.linalg.norm(rotation_axis) > 0:
                        # 应用旋转
                        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                            rotation_axis / np.linalg.norm(rotation_axis) * rotation_angle
                        )
                        cylinder.rotate(R, center=np.array([0, 0, 0]))
                    
                    # 平移到正确位置
                    cylinder.translate(point1 + direction * length / 2)
                    
                    # 设置颜色
                    color = np.array([
                        1 - normalized_weight,  # R
                        0.0,                    # G
                        normalized_weight       # B
                    ])
                    cylinder.paint_uniform_color(color)
                    
                    # 添加到可视化窗口
                    vis.add_geometry(cylinder)
        
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.point_size = 10.0  # 增加点的大小
        opt.background_color = np.array([1, 1, 1])  # 设置白色背景
        
        # 设置相机视角
        if camera_position is not None:
            ctr = vis.get_view_control()
            ctr.set_front(camera_position["front"])
            ctr.set_lookat(camera_position["lookat"])
            ctr.set_up(camera_position["up"])
            ctr.set_zoom(camera_position["zoom"])
        
        # 渲染并保存
        if output_path is not None:
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(output_path)
        
        # 显示
        vis.run()
        vis.destroy_window()

def check_o3d_to_trimesh(mesh):
    """
    将open3d的mesh转换为trimesh的mesh
    """
    # 如果是open3d mesh转换为trimesh对象
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        trimesh_mesh = mesh
    return trimesh_mesh

def check_trimesh_to_o3d(mesh):
    """
    将trimesh的mesh转换为open3d的mesh
    """
    if isinstance(mesh, trimesh.Trimesh):
        return o3d.geometry.TriangleMesh(vertices=mesh.vertices, triangles=mesh.faces)
    return mesh
def get_CoverageAxis_solver(mesh, dilation, output_path):
    """
    获取mesh的Coverage Axis结果
    
    参数:
        mesh: open3d.geometry.TriangleMesh对象
        
    返回:
        selected_points: numpy数组,选择的内部点坐标 (N, 3)
        selected_radius: numpy数组,对应的半径值 (N, 1)
        coverage_rate: float,覆盖率
    """
    
    
    solver = CoverageAxisSolver(
                 surface_sample_num=7000,
                 inner_points_method="random",
                 random_sample_number=500000,
                 max_iter=200,
                 reg_radius=1,
                 reg=1,
                 output_path=output_path)
    selected_points, selected_radius, coverage_rate, scale = solver.solve(check_o3d_to_trimesh(mesh), threshold=1, dilation=dilation)

    solver.save_CoverageAxis_results(selected_points, selected_radius, coverage_rate, scale)

    return solver

def convert_to_standard_bone_transform(bone_transforms, rest_bones_t):
    """
    将减去骨骼位置后的变换矩阵转换为标准的骨骼变换矩阵
    
    参数:
    bone_transforms: shape (num_bones, num_poses, 4, 3) 当前的变换矩阵
    rest_bones_t: shape (num_bones, 3) 骨骼的静止位置
    
    返回:
    standard_transforms: shape (num_bones, num_poses, 4, 4) 标准的骨骼变换矩阵
    """
    num_bones, num_poses = bone_transforms.shape[:2]
    standard_transforms = np.zeros((num_bones, num_poses, 4, 4))
    
    for bone in range(num_bones):
        for pose in range(num_poses):
            # 获取当前的旋转和平移
            R = bone_transforms[bone, pose, :3, :]  # 3x3 旋转矩阵
            t = bone_transforms[bone, pose, 3, :]   # 1x3 平移向量
            
            # 计算标准变换矩阵
            # 1. 首先将点移动到骨骼位置
            # 2. 应用旋转
            # 3. 应用平移
            # 4. 将点移回原位置
            
            # 构建完整的4x4变换矩阵
            standard_transforms[bone, pose, :3, :3] = R
            standard_transforms[bone, pose, :3, 3] = t - np.dot(R, rest_bones_t[bone])
            standard_transforms[bone, pose, 3, 3] = 1.0
            
    return standard_transforms