import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import heapq
from utils.skeletonizer import *
from scipy.sparse.csgraph import connected_components
import trimesh
from tqdm import tqdm
import pygeodesic.geodesic as geodesic
import networkx as nx

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

def convert_mesh_to_o3d_tensor(mesh):
    """
    将不同类型的mesh对象转换为Open3D tensor mesh
    
    参数:
        mesh: 输入的mesh对象，可能是trimesh或open3d对象
        
    返回:
        o3d.t.geometry.TriangleMesh: 转换后的Open3D tensor mesh对象
    """

     # 检查输入类型
    if isinstance(mesh, o3d.t.geometry.TriangleMesh):
        # 已经是tensor mesh，直接返回
        return mesh
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        # 是legacy mesh，转换为tensor mesh
        return o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    elif isinstance(mesh, trimesh.Trimesh):
        # 是trimesh对象，先转换为Open3D legacy mesh
        vertices = np.array(mesh.vertices, dtype=np.float32)
        triangles = np.array(mesh.faces, dtype=np.int32)
        
        # 创建Open3D legacy mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # 计算法向量
        o3d_mesh.compute_vertex_normals()
        
        # 转换为tensor mesh
        return o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    else:
        raise TypeError(f"不支持的mesh类型: {type(mesh)}")

def load_data(txt_path, dilation=0.02):
    """加载包含'v'前缀的球体数据"""
    # 读取原始数据（跳过空行，处理'v'前缀）
    with open(txt_path, 'r') as f:
        lines = [line.strip().split() for line in f if line.startswith('v')]
    
    # 提取数值部分（v后的4个数值：x,y,z,radius）
    data = np.array([[float(x) for x in line[1:5]] for line in lines])
    
    points = data[:, :3]
    radii = data[:, 3] + dilation
    return points, radii

def validate_connection_mesh_pure(p1, p2, scene):
    """
    使用射线检测验证两点之间的连接是否有效
    
    参数:
        p1, p2: 需要验证连接的两个点
        scene: Open3D场景对象 (RaycastingScene)，用于射线检测
    """
    # 计算方向和距离
    direction = p2 - p1
    distance = np.linalg.norm(direction)
    
    # 如果两点几乎重合，认为连接有效
    if distance < 1e-6:
        return True
    
    # 方法1: 使用射线检测 (适用于RaycastingScene)
    if isinstance(scene, o3d.t.geometry.RaycastingScene):
        # 归一化方向向量
        direction = direction / distance
        
        # 构建从p1到p2的射线
        ray = np.zeros((1, 6))
        ray[0, :3] = p1
        ray[0, 3:] = direction
        ray = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)
        
        # 射线检测
        ans = scene.cast_rays(ray)
        hit_distance = ans['t_hit'].numpy()[0]
        
        # 如果射线命中距离大于等于p1到p2的距离，则连线完全在mesh内部
        if hit_distance >= distance * 0.99:
            return True
    return False


def validate_connection_mesh(p1, p2, scene, mesh=None, all_medial_points=None, adjacency=None, epsilon=1e-6, geodesic_threshold=1.5):
    """
    检测两点之间的连接是否有效
    
    参数:
        p1, p2: 需要验证连接的两个点
        scene: Open3D场景对象 (RaycastingScene)，用于射线检测
        mesh: Open3D三角网格对象 (可选)，用于测地距离计算
        all_medial_points: 所有中轴点的列表 (可选)，用于拓扑分析
        adjacency: 现有的邻接矩阵 (可选)，用于拓扑不变性检测
        epsilon: 距离阈值，小于此值认为点重合
        geodesic_threshold: 测地线距离与欧氏距离的比值阈值，小于此值认为两点应该相连
    
    返回:
        bool: 如果连接有效返回True，否则返回False
    """
    # 计算方向和距离
    direction = p2 - p1
    distance = np.linalg.norm(direction)
    
    # 如果两点几乎重合，认为连接有效
    if distance < epsilon:
        return True
    
    # 方法1: 使用射线检测 (适用于RaycastingScene)
    if isinstance(scene, o3d.t.geometry.RaycastingScene):
        # 归一化方向向量
        direction = direction / distance
        
        # 构建从p1到p2的射线
        ray = np.zeros((1, 6))
        ray[0, :3] = p1
        ray[0, 3:] = direction
        ray = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)
        
        # 射线检测
        ans = scene.cast_rays(ray)
        hit_distance = ans['t_hit'].numpy()[0]
        
        # 如果射线命中距离大于等于p1到p2的距离，则连线完全在mesh内部
        if hit_distance >= distance * 0.99:
            return True
        
        # 如果射线检测失败，使用测地线距离作为先验, 但是这个连接必须保证拓扑不变，即没有新的环出现
        if mesh is not None:
            # 确保mesh是trimesh对象
            if isinstance(mesh, o3d.geometry.TriangleMesh):
                mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
                
            # 1. 找到最近的顶点
            dist_to_vertices = np.linalg.norm(mesh.vertices - p1, axis=1)
            closest_vertex_idx1 = np.argmin(dist_to_vertices)
            
            dist_to_vertices = np.linalg.norm(mesh.vertices - p2, axis=1)
            closest_vertex_idx2 = np.argmin(dist_to_vertices)
            
            # 2. 使用pygeodesic计算测地线距离
            geo_solver = geodesic.PyGeodesicAlgorithmExact(mesh.vertices, mesh.faces)
            geodesic_dist, _ = geo_solver.geodesicDistance(closest_vertex_idx1, closest_vertex_idx2)
            
            # 3. 计算欧氏距离
            euclidean_dist = np.linalg.norm(p2 - p1)
            
            # 4. 比较测地线距离与欧氏距离的比值
            geo_ratio = geodesic_dist / euclidean_dist
            
            if geo_ratio < geodesic_threshold:
                # 5. 拓扑不变性检测 - 确保添加连接不会形成新的环路
                if all_medial_points is not None and adjacency is not None:
                    try:
                        # 找到p1和p2在all_medial_points中的索引
                        p1_idx = np.argmin(np.linalg.norm(all_medial_points - p1, axis=1))
                        p2_idx = np.argmin(np.linalg.norm(all_medial_points - p2, axis=1))
                        
                        # 创建邻接矩阵的副本，并移除正在验证的边
                        adj_copy = adjacency.copy()
                        
                        # 直接从修改后的邻接矩阵创建图
                        G = nx.from_numpy_array(adj_copy)
                        
                        # 检查节点是否在图中
                        # 如果节点不在图中，说明它们没有任何连接，可以安全地添加新连接
                        if p1_idx not in G or p2_idx not in G:
                            return True
                        
                        # 检查是否已经存在路径 - 如果存在，添加新边会形成环
                        if nx.has_path(G, p1_idx, p2_idx):
                            # 已存在路径，添加连接会形成环
                            # visualize_connections(all_medial_points, np.triu(nx.adjacency_matrix(G).toarray()))
                            # print(f"WARNING: 添加连接会形成环，跳过连接 {p1_idx} {p2_idx}")
                            return False
                        
                        # 不会形成环，连接有效
                        return True
                    except Exception as e:
                        # 捕获任何可能的异常，确保程序不会崩溃
                        print(f"拓扑检测出错: {e}")
                        # 如果发生错误，退回到简单的测地距离比较
                        return True
                else:
                    # 如果没有提供拓扑信息，仅使用测地距离比较
                    return True
        
        return False


def validate_connection(p1, p2, scene=None, mesh=None, gs_points=None, all_medial_points=None, adjacency=None, validation_mode="both", geodesic_threshold=1.5, **kwargs):
    """
    综合验证函数，可选择验证方式
    
    参数:
        p1, p2: 需要验证连接的两个点
        scene: Open3D场景对象（用于mesh验证）
        mesh: 网格对象，用于测地距离计算
        gs_points: 高斯点云数组（用于点云验证）
        all_medial_points: 所有中轴点的列表（用于拓扑分析）
        adjacency: 现有的邻接矩阵（用于拓扑不变性检测）
        validation_mode: 验证模式，可选 "gs", "mesh", "both"
        geodesic_threshold: 测地线距离与欧氏距离的比值阈值，小于此值认为两点应该相连
        **kwargs: 传递给gs验证的其他参数
    """
    if validation_mode == "mesh":
        if gs_points is None:
            raise ValueError("需要提供gs_points进行高斯点云验证")
        return validate_connection_mesh_pure(p1, p2, scene)
        
    elif validation_mode == "geodesic":
        if scene is None:
            raise ValueError("需要提供scene进行mesh验证")
        return validate_connection_mesh(p1, p2, scene, mesh, all_medial_points, adjacency, geodesic_threshold=geodesic_threshold)
        
    
    else:
        raise ValueError("无效的验证模式，请选择 'gs', 'mesh' 或 'both'")

def compute_adjacency_with_dist(points, radii, mesh, dist_matrix, gs_points, geodesic_threshold=1.5):
    """
    计算邻接矩阵，使用mesh和gs点云信息来验证和优化连通性
    
    参数:
        points: 球体中心点坐标数组 (N,3)
        radii: 球体半径数组 (N,)
        mesh: o3d.geometry.TriangleMesh对象
        dist_matrix: 点之间的距离矩阵 (N,N)
        gs_points: 高斯点云坐标数组 (M,3)
        geodesic_threshold: 测地线距离与欧氏距离的比值阈值，小于此值认为两点应该相连
    """
    # 原始邻接矩阵计算
    radius_sum = radii[:, None] + radii
    adjacency = (dist_matrix < radius_sum).astype(int)
    deleted_adjacency = np.zeros_like(adjacency)
    np.fill_diagonal(adjacency, 0)

    # 转换为tensor mesh以使用新的API
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    # 对现有连接进行验证和修正
    n_points = len(points)
    for i in range(n_points):
        for j in range(i+1, n_points):
            if adjacency[i,j] == 1:
                # 验证连接的有效性，传递中轴点和邻接矩阵用于拓扑不变性检测
                if not validate_connection(points[i], points[j], scene, mesh, gs_points, 
                                         all_medial_points=points, adjacency=adjacency,
                                         validation_mode="mesh", geodesic_threshold=geodesic_threshold):
                    adjacency[i,j] = 0
                    adjacency[j,i] = 0
                    deleted_adjacency[i,j] = 1
                    deleted_adjacency[j,i] = 1
    
    # 计算每个点的度
    degrees = np.sum(adjacency, axis=1)
    
    # 收集所有需要验证的连接
    potential_connections = []
    for i in range(n_points):
        for j in range(i+1, n_points):
            if deleted_adjacency[i,j] == 1:
                # 使用两个点的度之和作为优先级
                total_degree = degrees[i] + degrees[j]
                potential_connections.append((i, j, total_degree))
    
    # 按照度之和从大到小排序
    potential_connections.sort(key=lambda x: x[2], reverse=True)
    
    # 按排序后的顺序验证连接
    for i, j, _ in potential_connections:
        if validate_connection(points[i], points[j], scene, mesh, gs_points, 
                             all_medial_points=points, adjacency=adjacency,
                             validation_mode="geodesic", geodesic_threshold=geodesic_threshold):
            adjacency[i,j] = 1
            adjacency[j,i] = 1
    
    return adjacency

def compute_adjacency_from_MAT(ma_file_path, QMAT_path, selected_points, temp_result_path):
    # read ma file, select poles, compute QMAT, read the result, return the adjacency matrix
    
    # MAT is topological correct, we need to use seleted points to select which point on ma file to save 
    pass

def visualize_spheres(points, radii, output_path=None, camera_position=None):
    """
    使用Open3D可视化球体，并可选择保存截图
    
    参数:
        points: 球体中心点坐标数组 (N,3)
        radii: 球体半径数组 (N,)
        output_path: 截图保存路径，如果为None则不保存截图
        camera_position: 相机位置，如果为None则使用默认位置
    """
    geometries = []
    for (x, y, z), r in zip(points, radii):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere.translate((x, y, z))
        sphere.paint_uniform_color([0.5, 0.5, 1.0])  # Light blue color
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    if output_path is None:
        # 如果不保存截图，使用原来的方式显示
        o3d.visualization.draw_geometries(geometries, window_name="Spheres Visualization", width=800, height=600)
    else:
        # 使用Visualizer来设置视角并保存截图
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Spheres Visualization", width=800, height=600)
        
        # 添加所有几何体
        for geometry in geometries:
            vis.add_geometry(geometry)
        
        # 设置相机参数
        if camera_position is not None:
            ctr = vis.get_view_control()
            ctr.set_front(camera_position["front"])
            ctr.set_lookat(camera_position["lookat"])
            ctr.set_up(camera_position["up"])
            ctr.set_zoom(camera_position["zoom"])
        else:
            ctr = vis.get_view_control()
            ctr.set_front([0.0, 0.0, -1.0])  # 设置相机前向
            ctr.set_lookat([0.0, 0.0, 0.0])  # 设置观察点
            ctr.set_up([0.0, -1.0, 0.0])     # 设置上向量
            ctr.set_zoom(0.8)                # 设置缩放比例
        
        # 渲染并保存截图
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        print(f"截图已保存至: {output_path}")

def visualize_connections_with_labels(points, adjacency):
    """
    使用matplotlib+Open3D的混合方法可视化带索引标签的点云连接
    这种方法在所有Open3D版本中都能工作
    
    参数:
        points: 点坐标数组 (N,3)
        adjacency: 邻接矩阵 (N,N)
    """
    # 创建图形和3D轴
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', s=30)
    
    # 绘制连接线
    for i in range(len(adjacency)):
        for j in np.where(adjacency[i] == 1)[0]:
            # 仅绘制一个方向以避免重复
            if i < j:
                ax.plot([points[i, 0], points[j, 0]],
                       [points[i, 1], points[j, 1]],
                       [points[i, 2], points[j, 2]], 'r-', linewidth=1)
    
    # 添加索引标签
    for i, point in enumerate(points):
        ax.text(point[0], point[1], point[2], f"{i}", color='blue', fontsize=10)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('点云连接与索引可视化')
    
    # 自动调整坐标轴范围以适应数据
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    # 添加一些边距
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
    
    plt.tight_layout()
    plt.show()

def visualize_connections(points, adjacency, output_path=None, camera_position=None, show_indices=True, label_offset=[0, 0, 0.02], use_matplotlib=False):
    """
    使用Open3D可视化点云连接，并可选择保存截图
    
    参数:
        points: 点坐标数组 (N,3)
        adjacency: 邻接矩阵 (N,N)
        output_path: 截图保存路径，如果为None则不保存截图
        camera_view: 相机视角，可选值: "front", "back", "left", "right", "top", "bottom", "isometric"
        show_indices: 是否显示索引标签
        label_offset: 标签相对于点的偏移量
        use_matplotlib: 是否使用matplotlib进行可视化（当Open3D文本渲染不可用时）
    """
    # 如果使用matplotlib可视化
    if use_matplotlib:
        visualize_connections_with_labels(points, adjacency)
        return
        
    # 使用Open3D可视化
    lines = []
    colors = []
    for i in range(len(adjacency)):
        for j in np.where(adjacency[i] == 1)[0]:
            lines.append([i, j])
            colors.append([1, 0, 0])  # Red color for lines

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    points_set = o3d.geometry.PointCloud()  
    points_set.points = o3d.utility.Vector3dVector(points)
    points_set.paint_uniform_color([0, 0, 0])
    
    geometries = [line_set, points_set]
    
    # 添加索引标签
    if show_indices:
        offset = np.array(label_offset)
        # 为每个点创建索引标签
        for i, point in enumerate(points):
            # 使用不同颜色的小球体来表示索引点
            label_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            label_sphere.compute_vertex_normals()
            label_sphere.paint_uniform_color([0, 1, 0])  # 绿色
            label_sphere.translate(point + offset)
            geometries.append(label_sphere)
    
    if output_path is None:
        # 如果不保存截图，使用原来的方式显示
        print("正在显示点云连接与索引标签...")
        print(f"共有 {len(points)} 个点和 {len(lines)} 条连线")
        print("提示: 索引标签使用绿色小球体表示，对应点索引为 0 到 {len(points)-1}")
        print("如果需要更清晰的索引标签，请使用参数 use_matplotlib=True")
        o3d.visualization.draw_geometries(geometries, window_name="Connections Visualization", width=800, height=600)
    else:
        # 使用Visualizer来设置视角并保存截图
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Connections Visualization", width=800, height=600)
        
        # 添加所有几何体
        for geometry in geometries:
            vis.add_geometry(geometry)
        
        # 设置相机参数
        if camera_position is not None:
            ctr = vis.get_view_control()
            ctr.set_front(camera_position["front"])
            ctr.set_lookat(camera_position["lookat"])
            ctr.set_up(camera_position["up"])
            ctr.set_zoom(camera_position["zoom"])
        else:
            ctr = vis.get_view_control()
            ctr.set_front([0.0, 0.0, -1.0])  # 设置相机前向
            ctr.set_lookat([0.0, 0.0, 0.0])  # 设置观察点
            ctr.set_up([0.0, -1.0, 0.0])     # 设置上向量
            ctr.set_zoom(0.8)                # 设置缩放比例
        
        # 渲染并保存截图
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        print(f"截图已保存至: {output_path}")


def save_as_obj(points, adjacency, filename):
    """Save points and connections as an OBJ file"""
    with open(filename, 'w') as f:
        # Write vertices
        for x, y, z in points:
            f.write(f"v {x} {y} {z}\n")
        
        # Write edges (connections)
        for i in range(len(adjacency)):
            for j in np.where(adjacency[i] == 1)[0]:
                # OBJ format uses 1-based indexing
                f.write(f"l {i + 1} {j + 1}\n")
    
    print(f"Saved to {filename}")


def build_minimum_spanning_tree(joints, bones, root_point_index, points_array):
    """
    基于根节点和关节点构建最小生成树
    参数:
        joints: 关节点坐标数组 (N,3)（来自points的子集）
        bones: 当前骨骼连接列表 [[parent_idx, child_idx], ...]（使用joints的索引）
        root_point_index: 在原始points数组中的根节点索引
        points_array: 原始点云坐标数组
    返回:
        new_bones: 以root为根的生成树连接列表（使用joints的索引）
    """
    # 将原始points的根索引转换为joints数组中的索引
    root_in_joints = np.where(np.all(joints == points_array[root_point_index], axis=1))[0]
    if len(root_in_joints) == 0:
        raise ValueError("Root node not found in joints array")
    root_index = root_in_joints[0]

    # 构建带权邻接表（使用joints的局部索引）
    adj_dict = {i: [] for i in range(len(joints))}
    for u, v in bones:
        distance = np.linalg.norm(joints[u] - joints[v])
        adj_dict[u].append((v, distance))
        adj_dict[v].append((u, distance))

    # Prim算法实现（从转换后的root_index开始）
    visited = set([root_index])
    edges = []
    heap = []
    
    # 初始化优先队列（添加参数验证）
    if root_index not in adj_dict:
        raise ValueError("Root index not in adjacency dictionary")
    
    # 修改堆的初始化方式
    for v, w in adj_dict.get(root_index, []):  # 添加get方法防止KeyError
        heapq.heappush(heap, (w, root_index, v))
    
    while heap:
        weight, u, v = heapq.heappop(heap)
        
        # 添加循环检测
        if v in visited:
            continue  # 跳过已访问节点
            
        visited.add(v)
        edges.append([u, v])
        
        # 添加邻接节点存在性检查
        for neighbor, w in adj_dict.get(v, []):
            if neighbor not in visited:
                heapq.heappush(heap, (w, v, neighbor))
    
    # 添加后验证
    if len(edges) != len(joints) - 1:
        print(f"警告：生成树不完整，可能存在多个连通分量（期望边数 {len(joints)-1}，实际生成 {len(edges)}）")
    
    return edges

def post_process_bones(bones, joints_index, points, remove_redundant=False):
    """
    骨骼后处理函数，包含以下功能：
    1. 删除冗余骨骼分支（可选）
    2. 更新关节点索引
    3. 转换绝对坐标为相对索引
    
    参数:
        bones: 原始骨骼连接列表 [[parent_idx, child_idx], ...]（使用points数组的绝对索引）
        joints_index: 原始关节点在points数组中的索引列表
        points: 原始点云坐标数组
        remove_redundant: 是否删除冗余骨骼，默认为False
        
    返回:
        processed_bones: 处理后的骨骼连接列表（使用joints_index的相对索引）
        processed_joints: 处理后的关节点索引列表
    """
    if remove_redundant:
        # 提取骨骼起点终点
        starts = np.array([bone[0] for bone in bones])
        tails = np.array([bone[1] for bone in bones])
        
        # 标记有子节点的骨骼
        bone_has_child = np.isin(tails, starts)
        
        # 删除冗余骨骼
        del_indices = []
        for u_start in np.unique(starts):
            indx = np.where(u_start == starts)[0]
            has_child = bone_has_child[indx]
            
            if has_child.any():
                # 删除无子分支
                del_indices.extend(indx[~has_child])
            else:
                # 保留最长骨骼
                lengths = [np.linalg.norm(points[b[0]] - points[b[1]]) for b in np.array(bones)[indx]]
                longest = indx[np.argmax(lengths)]
                del_indices.extend([i for i in indx if i != longest])
        
        # 逆向删除避免索引错位
        processed_bones = [b for i, b in enumerate(bones) if i not in set(del_indices)]
    else:
        processed_bones = bones.copy()
    
    # 更新关节点索引
    remaining_joints = np.unique(processed_bones).astype(int)
    processed_joints = [j for j in joints_index if j in remaining_joints]
    
    # 转换绝对索引为相对索引
    index_map = {jid: idx for idx, jid in enumerate(processed_joints)}
    processed_bones = [[index_map[b[0]], index_map[b[1]]] for b in processed_bones]
    
    return processed_bones, processed_joints



if __name__ == "__main__":
    # 配置参数
    input_txt = "./output/mesh_selected_inner_points.txt"  # 输入文件路径
    candidate_path = "/home/wjx/research/code/GaussianAnimator/Coverage_Axis/input/frame_0_random.obj"
    output_adj = "./output/adjacency_matrix.txt"          # 邻接矩阵输出路径
    dilation = 0.03
    # 加载并计算
    candidate_points = trimesh.load(candidate_path)
    candidate_points = np.asarray(candidate_points.vertices)
    points, radii = load_data(input_txt, dilation)
    dist_matrix = cdist(points, points)
    
    # 在主函数中，将 trimesh 对象转换为 Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(candidate_points))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(candidate_points.faces))
    mesh_o3d.compute_vertex_normals()

    # 然后传入转换后的 mesh
    adjacency = compute_adjacency_with_dist(points, radii, mesh_o3d, dist_matrix, candidate_points)
    
    # 保存邻接矩阵
    np.savetxt(output_adj, adjacency, fmt="%d")
    print(f"邻接矩阵已保存至 {output_adj}")
    
    # 可视化球体
    visualize_spheres(points, radii)
    visualize_spheres(candidate_points, dilation*np.ones(candidate_points.shape[0]))
    
    # 可视化连接（带索引标签）
    # 1. 使用Open3D带球体标记的方式
    visualize_connections(points, adjacency, show_indices=True)
    
    # 2. 或者使用matplotlib方式获得更清晰的文本标签
    print("\n尝试使用matplotlib方式显示更清晰的索引标签...")
    visualize_connections(points, adjacency, use_matplotlib=True)

    distance_graph = dist_matrix * adjacency
    D = shortest_path(distance_graph, directed=True, method='FW')
    root_indx = D.sum(1).argmin()
    graph = adjacency_to_graph(distance_graph)
    
    
    joints_index, bones = bfs(graph, root_indx, 0.01)
    # 替换原有处理代码为函数调用
    bones, joints_index = post_process_bones(bones, joints_index, points)
    
    # 更新关节点坐标
    joints = points[joints_index]
    root = points[root_indx]
    bones = build_minimum_spanning_tree(joints, bones, root_indx, points)
    
    visualise_skeletonizer(points, root, joints, bones, points, np.zeros_like(points))

    res = {
        'skeleton_pcd': points,
        'root': root,
        'root_idx': root_indx,
        'joints': joints,
        'joints_idx': joints_index,
        'bones': bones,
    }

    np.save('./output/skeleton.npy', res)
    print("骨骼已保存至output",)