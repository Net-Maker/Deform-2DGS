"""
Code from "Template-free Articulated Neural Point Clouds for Reposable View Synthesis"
Github: https://github.com/lukasuz/Articulated-Point-NeRF
Project Page: https://lukas.uzolas.com/Articulated-Point-NeRF/
"""

import torch
import numpy as np
from skimage.morphology import skeletonize_3d as skeletonize
from skimage.morphology import remove_small_holes
from scipy.sparse.csgraph import shortest_path
from scipy.special import softmax
from skimage import filters
from cc3d import largest_k

from seaborn import color_palette
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def adjacency_to_graph(distances):
    """ 将邻接矩阵转换为图的表示形式。
    
    Arguments:
        distances: NxN np.array, 包含节点之间距离的邻接矩阵
    
    Returns:
        dict, 邻接矩阵的图表示形式
        
        返回的图数据结构为:
        {
            节点索引: {
                'neighbours': (相邻节点索引的元组,按距离排序),
                'n_distances': (到相邻节点距离的元组,升序排列)
            },
            ...
        }
        
        例如:
        {
            0: {
                'neighbours': (1, 2, 3),
                'n_distances': (0.5, 1.2, 2.1)  
            },
            1: {
                'neighbours': (0, 3),
                'n_distances': (0.5, 1.8)
            }
            ...
        }
    """
    graph = {}
    for i, node in enumerate(distances):
        adj = []
        adj_distances = []
        for j, connected in enumerate(node):
            if connected and i != j:
                adj.append(j)
                adj_distances.append(distances[i, j])

        adj = np.array(adj)
        adj_distances = np.array(adj_distances)
        sort_indicies = np.argsort(adj_distances)
        adj = tuple(adj[sort_indicies])
        adj_distances = tuple(adj_distances[sort_indicies])

        graph[i] = {
            'neighbours': adj,
            'n_distances': adj_distances
        }

    return graph

class DistQueue():
    """ Queue that sorts elements by distance.
    """
    def __init__(self) -> None:
        self._elements = np.array([], dtype=int)
        self._distances = np.array([], dtype=float)
        self._prev_joints = np.array([], dtype=int)
        self._distances_prev_joint = np.array([], dtype=float)
    
    def enqueue(self, element, distance, prev_joint, distance_prev_joint) -> None:
        # Find closest larger value
        if len(self._distances) == 0:
            indx = 0
        else:
            mask = self._distances > distance
            if not mask.any(): # no bigger elements, insert at the end
                indx = len(self._distances)
            else:  # Insert right before larger value
                indx = np.argmin(self._distances < distance)
       
        self._elements = np.insert(self._elements, indx, element)
        self._distances = np.insert(self._distances, indx, distance)
        self._prev_joints  = np.insert(self._prev_joints, indx, prev_joint)
        self._distances_prev_joint = np.insert(self._distances_prev_joint, indx, distance_prev_joint)

    def pop(self) -> tuple:
        element, self._elements = self._elements[0], self._elements[1:]
        distance, self._distances = self._distances[0], self._distances[1:]
        prev_joint, self._prev_joints = self._prev_joints[0], self._prev_joints[1:]
        distance_prev_joint, self._distances_prev_joint = self._distances_prev_joint[0], self._distances_prev_joint[1:]
        return element, distance, prev_joint, distance_prev_joint
    
    def not_empty(self) -> bool:
        return len(self._distances) > 0

def dfs(graph, start_node_indx, bone_length):
    """ 使用队列的深度优先搜索来构建骨架树。
    
    工作机制:
    1. 使用队列来模拟DFS的行为，每次取最新加入的节点
    2. 对于每个访问的节点:
       - 如果距离上一个关节点的距离超过bone_length
       - 或者是分支点(有多个未访问的邻居)
       - 或者是叶子节点(没有未访问的邻居)
       则将该节点标记为新的关节点
    
    Arguments:
        graph: dict, 邻接矩阵的图表示,存储了每个节点的邻居节点及其距离
        start_node_indx: int, 起始节点(根节点)的索引
        bone_length: num, 每个骨骼段的目标长度
    
    Returns:
        joints: list, 关节点的索引列表
        bones: list, 骨骼连接列表 [start_joint, end_joint]
    """
    visited = []
    joints = [start_node_indx]  # 初始只包含根节点
    bones = []  # 存储骨骼连接关系
    visited.append(start_node_indx)
    queue = DistQueue()
    queue.enqueue(start_node_indx, 0., start_node_indx, 0.)
    
    # 记录每个节点的父节点，用于构建树结构
    parent_map = {start_node_indx: None}
    
    while queue.not_empty():
        indx, cm_distance, prev_joint, distance_prev_joint = queue.pop()
        node = graph[indx]
        
        # 获取未访问的邻居节点
        neighbours_to_visit = [n for n in node['neighbours'] if n not in visited]
        
        # 判断是否应该成为关节点:
        # 1. 距离上一个关节点太远
        # 2. 是分支点(有多个未访问邻居)
        # 3. 是叶子节点(没有未访问邻居)
        is_joint = (distance_prev_joint >= bone_length or 
                   len(neighbours_to_visit) > 1 or 
                   len(neighbours_to_visit) == 0)
        
        if is_joint:
            bones.append([prev_joint, indx])  # 添加新的骨骼连接
            joints.append(indx)  # 将当前点标记为关节点
            prev_joint = indx  # 更新上一个关节点
            distance_prev_joint = 0  # 重置距离计数
        
        # DFS特性：将未访问的邻居按距离倒序加入队列（这样最近的节点会最后加入，最先被处理）
        neighbours_distances = [(n, d) for n, d in zip(neighbours_to_visit, node['n_distances'][:len(neighbours_to_visit)])]
        neighbours_distances.sort(key=lambda x: x[1], reverse=True)  # 按距离倒序排序
        
        # 继续遍历未访问的邻居
        for neighbour, distance in neighbours_distances:
            if neighbour not in visited:
                visited.append(neighbour)
                parent_map[neighbour] = indx  # 记录父节点
                # 更新累积距离
                nn_cm_distance = cm_distance + distance
                nn_distance_prev_joint = distance_prev_joint + distance
                queue.enqueue(neighbour, nn_cm_distance, prev_joint, nn_distance_prev_joint)
    
    # 移除根节点到第一个关节的无效骨骼连接
    if len(bones) > 0 and bones[0][0] == bones[0][1]:
        bones = bones[1:]
    
    return joints, bones


def bfs(graph, start_node_indx, bone_length):
    """ 广度优先搜索来寻找骨架的关节点和骨骼。
    
    工作机制:
    1. 从起始节点(root)开始,使用优先队列(DistQueue)按距离顺序遍历图
    2. 对于每个访问的节点:
       - 如果距离上一个关节点的距离超过bone_length,或者是叶子节点(没有未访问的邻居)
       - 则将该节点标记为新的关节点,并与上一个关节点之间创建一个骨骼连接
    3. 然后将该节点的未访问邻居加入队列继续遍历
    4. 这样可以自适应地将点云骨架分解成关节点和骨骼段
    
    Arguments:
        graph: dict, 邻接矩阵的图表示,存储了每个节点的邻居节点及其距离
        start_node_indx: int, 起始节点(根节点)的索引
        bone_length: num, 每个骨骼段的目标长度(体素坐标)
    
    Returns:
        joints: list, 关节点的索引列表。这些点是骨骼的连接处,包括:
               - 根节点
               - 分支点(有多个子骨骼的点) 
               - 端点(叶子节点)
               - 骨骼长度达到阈值处的点
        bones: list, 骨骼连接列表。每个骨骼是[start_joint, end_joint],
               表示两个关节点之间的骨骼段。这些骨骼段构成了完整的骨架结构。
    """
    visited = []
    joints = [start_node_indx]  # 初始只包含根节点
    bones = []  # 存储骨骼连接关系
    visited.append(start_node_indx)
    queue = DistQueue()
    queue.enqueue(start_node_indx, 0., start_node_indx, 0.)

    while queue.not_empty():        
        indx, cm_distance, prev_joint, distance_prev_joint = queue.pop()
        node = graph[indx]

        # 获取未访问的邻居节点
        neighbours_to_visit = [n for n in node['neighbours'] if n not in visited]
        # 当距离超过阈值或到达叶子节点时创建新的骨骼
        add_bone = (distance_prev_joint >= bone_length) or len(neighbours_to_visit) == 0

        if add_bone:
            bones.append([prev_joint, indx])  # 添加新的骨骼连接
            joints.append(indx)  # 将当前点标记为关节点
            prev_joint = indx  # 更新上一个关节点
            distance_prev_joint = 0  # 重置距离计数

        # 继续遍历未访问的邻居
        for i, neighbour in enumerate(neighbours_to_visit):
            if neighbour not in visited:
                visited.append(neighbour)
                # 更新累积距离
                nn_cm_distance = cm_distance + node['n_distances'][i]
                nn_distance_prev_joint = distance_prev_joint + node['n_distances'][i]
                queue.enqueue(neighbour, nn_cm_distance, prev_joint, nn_distance_prev_joint)
    
    return joints, bones

def create_skeleton_mst(graph, start_node_indx, bone_length):
    """使用最小生成树(MST)算法提取骨架结构
    
    工作机制:
    1. 将图转换为邻接矩阵表示
    2. 使用Prim算法构建MST
    3. 基于MST和距离阈值提取关节点和骨骼
    
    Arguments:
        graph: dict, 邻接矩阵的图表示
        start_node_indx: int, 起始节点索引
        bone_length: float, 骨骼长度阈值
        
    Returns:
        joints: list, 关节点索引列表
        bones: list, 骨骼连接列表 [start_joint, end_joint]
    """
    import numpy as np
    from scipy.sparse.csgraph import minimum_spanning_tree
    
    # 1. 构建邻接矩阵
    n_nodes = len(graph)
    adj_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        if i in graph:
            for j, dist in zip(graph[i]['neighbours'], graph[i]['n_distances']):
                adj_matrix[i,j] = dist
                adj_matrix[j,i] = dist  # 确保对称
    
    # 2. 计算MST
    mst = minimum_spanning_tree(adj_matrix)
    mst = mst.toarray()
    
    # 3. 提取骨架结构
    joints = [start_node_indx]  # 初始包含根节点
    bones = []
    visited = set([start_node_indx])
    
    def process_node(parent_idx, parent_joint_idx):
        # 获取当前节点的所有子节点
        children = np.where((mst[parent_idx] > 0) | (mst[:,parent_idx] > 0))[0]
        
        for child_idx in children:
            if child_idx not in visited:
                visited.add(child_idx)
                
                # 计算到父节点的距离
                dist = max(mst[parent_idx,child_idx], mst[child_idx,parent_idx])
                
                # 判断是否应该成为关节点:
                # 1. 距离超过阈值
                # 2. 有多个子节点(分支点)
                # 3. 是叶子节点
                unvisited_children = [c for c in np.where((mst[child_idx] > 0) | 
                                                        (mst[:,child_idx] > 0))[0] 
                                    if c not in visited]
                
                is_joint = (dist >= bone_length or 
                           len(unvisited_children) > 1 or 
                           len(unvisited_children) == 0)
                
                if is_joint:
                    joints.append(child_idx)
                    bones.append([parent_joint_idx, len(joints)-1])
                    process_node(child_idx, len(joints)-1)
                else:
                    process_node(child_idx, parent_joint_idx)
    
    # 从根节点开始处理
    process_node(start_node_indx, 0)
    
    return joints, bones

def dist_batch(p, a, b):
    """ Vectorized point-to-line distance

    Arguments:
        p: Nx3 torch.tensor, points
        a: Mx3 torch.tensor, start of lines
        b: Mx3 torch.tensor, end of lines

    Returns:
        MxN torch.tensor, distance from each point to each line
    """
    assert len(a) == len(b), "Same batch size needed for a and b"

    p = p[None, :, :]
    s = b - a
    w = p - a[:, None, :]
    ps = (w * s[:, None, :]).sum(-1)
    res = torch.zeros((a.shape[0], p.shape[1]), dtype=p.dtype)

    # ps <= 0
    ps_smaller_mask = ps <= 0
    lower_mask = torch.where(ps_smaller_mask)
    res[lower_mask] = torch.norm(w[lower_mask], dim=-1)

    # ps > 0 and ps >= l2
    l2 = (s * s).sum(-1)
    ps_mask = ~ps_smaller_mask

    temp_mask_l2 = ps >= l2[:, None]
    upper_mask = torch.where(ps_mask & temp_mask_l2)
    res[upper_mask] = torch.norm(p[0][upper_mask[1]] - b[upper_mask[0]], dim=-1)

    # ps > 0 and ps < l2
    within_mask = torch.where(ps_mask & ~temp_mask_l2)
    res[within_mask] = torch.norm(
        p[0][within_mask[1]] - (a[within_mask[0]] + (ps[within_mask] / l2[within_mask[0]]).unsqueeze(-1) * s[within_mask[0]]), dim=-1)

    return res

def weight_from_bones(joints, bones, pcd, theta=0.05):
    """ 计算每个点的蒙皮权重。

    该函数计算点云中每个点相对于骨骼的蒙皮权重。权重表示每个点受各个骨骼影响的程度。
    计算过程如下:
    1. 计算每个点到每根骨骼的距离
    2. 将距离转换为权重,使用指数衰减函数
    3. 使用softmax归一化权重

    参数:
        joints: Nx3 np.array, 关节点坐标
        bones: 长度为N-1的列表,每个元素包含父骨骼和子骨骼的索引
        pcd: Mx3 np.array, 点云坐标
        theta: num, softmax的温度参数,控制权重分布的平滑程度
    
    返回:
        weights: MxN np.array, 每个点相对于每根骨骼的权重
    """
    # 初始化存储骨骼距离的数组
    bone_distances = np.zeros((len(bones), len(pcd)))

    # 计算每个点到每根骨骼的距离
    # 需要将输入转换为torch张量以使用vectorized距离计算
    bone_distances = dist_batch(
        torch.tensor(pcd),
        torch.tensor(np.array([joints[bone[0]] for bone in bones])).float(),  # 骨骼起点
        torch.tensor(np.array([joints[bone[1]] for bone in bones])).float(),  # 骨骼终点
        ).cpu().numpy()

    # 将距离转换为权重,使用指数衰减函数
    weights = (1 / (0.5 * np.e ** bone_distances + 1e-6)).T  # 添加1e-6避免除零
    # 使用softmax归一化权重,使每个点的权重和为1
    weights = softmax(weights / theta, axis=1)

    return weights

def preprocess_volume(alpha_volume, threshold, sigma=1):
    """
    Arguments:
        alpha_volume: LxMxN np.array, alpha volume before thresholding
        threshold: num, threshold for the alpha volume
        sigma: num, sigma for the gaussian filtering of the alpha volume
    
    Returns:
        LxMxN np.array, binary volume after thresholding
    """
    if sigma > 0:
        alpha_volume = filters.gaussian(alpha_volume, sigma=sigma, preserve_range=True)
    binary_volume = alpha_volume > threshold
    binary_volume = remove_small_holes(binary_volume.astype(bool), area_threshold=2**8,)
    binary_volume = largest_k(binary_volume, connectivity=26, k=1).astype(int)
    
    return binary_volume.astype(bool)

def create_skeleton(alpha_volume, grid_xyz, bone_length=10., threshold=0.05, sigma=0, weight_theta=0.1, bone_heursitic=True):
    """
    参数:
        alpha_volume: LxMxN np.array, 阈值处理前的alpha体素
        grid_xyz: LxMxN np.array, 体素的坐标网格
        bone_length: num, 每根骨骼的大致长度(不是世界坐标系下的长度,而是体素坐标系下的长度)
        threshold: num, alpha体素的阈值
        sigma: num, alpha体素高斯滤波的sigma值。注意:仅用于骨架生成
        weight_theta: num, softmax缩放的theta参数
        bone_heursitic: bool, 是否使用骨骼启发式方法
    
    返回值:
        包含点云和所有运动学组件的字典
    """

    ## Preprocessing, assume that we have one blob
    # 通过opacity和阈值生成体素mask
    binary_volume = preprocess_volume(alpha_volume, threshold=threshold, sigma=0)
    # 如果sigma大于0,则进行高斯滤波
    if sigma > 0:
        binary_volume_smooth = preprocess_volume(alpha_volume, threshold=threshold, sigma=sigma)
    else:
        binary_volume_smooth = binary_volume
    
    # Create integer volume grid, easier to work with
    xv, yv, zv = np.meshgrid(
        np.arange(0, grid_xyz.shape[0]),
        np.arange(0, grid_xyz.shape[1]),
        np.arange(0, grid_xyz.shape[2]),
        indexing='ij'
    )
    grid = np.concatenate([
        np.expand_dims(xv, axis=-1),
        np.expand_dims(yv, axis=-1),
        np.expand_dims(zv, axis=-1)
    ], axis=-1)
    # 用的scikit-image的skeletonize函数 效果见网页https://imagej.net/plugins/skeletonize3d 类似3D的MediumAxis
    # NOTE： 这里的非常原始，我们单纯将其换为Coverage Axis都可能提升效果

    skeleton = skeletonize(binary_volume_smooth) == 255
    # 这个skeleton非常dense
    points = grid[skeleton].reshape(-1, 3)
    # print(f"points shape: {points.shape}")

    ## Graphify - 将骨架点云转换为图结构
    # Neighbours are points within a NxNxN grid, N=3
    # 计算所有点对之间的坐标差的绝对值
    # points[:,None,:] 和 points[None,:,:] 通过广播机制生成所有点对的组合
    # 结果shape为 (n_points, n_points, 3), 每个元素表示两点在xyz三个维度上的坐标差的绝对值
    offset = np.abs(points[:,None,:] - points[None,:,:])
    
    # 找出相邻的点对(在3x3x3的网格内)
    # offset <= 1 判断每个维度的差值是否<=1,即判断两点是否在3x3x3网格内相邻
    # logical_and.reduce 对3个维度取与操作,只有当两点在所有维度都相邻时才为True
    # 结果为邻接矩阵,shape为(n_points, n_points),表示点之间的接关系
    NN = np.logical_and.reduce(offset <= 1, axis=-1)
    
    # 计算所有点对之间的欧氏距离
    # 结果shape为(n_points, n_points),每个元素表示对应两点间的欧氏距离
    distances = np.sqrt(np.sum((points[:,None,:] - points[None,:,:])**2, axis=-1))

    # 将邻接矩阵与距离矩阵相乘,得到带权重的邻接矩阵
    # 只有相邻点之间才有距离值,其他位置为0
    distance_graph = NN * distances

    # 使用Floyd-Warshall算法计算所有点对之间的最短路径距离
    # directed=True表示有向图,method='FW'指定使用Floyd-Warshall算法
    # 返回shape为(n_points, n_points)的距离矩阵D,D[i,j]表示点i到点j的最短路径长度
    D = shortest_path(distance_graph, directed=True, method='FW')
    root_indx = D.sum(1).argmin()
    # 前面构建了图，接下来就开始从图中构建骨架了，包括前面用最短路径找到的根节点

    graph = adjacency_to_graph(distance_graph)

    joints, bones = bfs(graph, root_indx, bone_length) # 通过bfs获得了初步的关节点和骨骼，是一个树结构
    # 提取每个骨骼的起点和终点
    starts = np.array([bone[0] for bone in bones])  # 所有骨骼的起点
    tails = np.array([bone[1] for bone in bones])   # 所有骨骼的终点

    # 判断每个骨骼是否有子骨骼(即该骨骼的终点是否是其他骨骼的起点)
    bone_has_child = []
    for i in range(len(bones)):
        bone_has_child.append(tails[i] in starts)

    # 启发式地清理冗余的骨骼
    if bone_heursitic:
        bone_has_child = np.array(bone_has_child)
        del_indices = []  # 存储要删除的骨骼索引
        
        # 对于每个唯一的起点
        for u_start in np.unique(starts):
            indx = np.where(u_start == starts)[0]  # 找到从该起点出发的所有骨骼
            
            # 如果该起点的某些骨骼有子骨骼
            if bone_has_child[indx].any():
                # 删除该起点的所有没有子骨骼的分支(可能是噪声)
                for i in indx:
                    if not bone_has_child[i]:
                        del_indices.append(i)
            else:
                # 如果该起点的所有骨骼都没有子骨骼,只保留最长的一个
                distances_temp = []
                for i in indx:
                    bone = bones[i]
                    # 计算骨骼长度
                    distances_temp.append(np.sqrt(np.sum(points[int(bone[0])] - points[bone[1]])**2))
                
                longest_indx = np.argmax(distances_temp)  # 找到最长骨骼
                # 删除其他较短的骨骼
                for i, ii in enumerate(indx):
                    if i != longest_indx:
                        del_indices.append(ii)

        # 从后向前删除冗余骨骼,避免索引错误
        del_indices.sort()
        del_indices.reverse()
        for i in del_indices:
            del bones[i]
        
        # 更新关节点列表,只保留骨骼连接处的关节点
        new_joints = list(np.unique(bones).astype(int))  # 移除不必要的关节点
        joints = [joint for joint in joints if joint in new_joints]  # 保持原始关节点的顺序

    ## Turn absolute bone coordinates into indices
    rel_bones = []
    for bone in bones:
        b1, b2 = bone
        b1 = int(np.where(np.array(joints) == b1)[0])
        b2 = int(np.where(np.array(joints) == b2)[0])
        rel_bones.append([b1, b2])
    bones = rel_bones

    ## Transform from grid space into real-world coordinates
    xyz_max = grid_xyz.max(axis=0).max(axis=0).max(axis=0)
    xyz_min = grid_xyz.min(axis=0).min(axis=0).min(axis=0)
    vol_max = np.array(binary_volume.shape)
    points = (points / vol_max[None,:]) * (xyz_max - xyz_min) + xyz_min
    points = points.astype(np.float32)

    ## Calculate weights
    pcd = grid_xyz[binary_volume > 0]
    weights = weight_from_bones(points[joints], bones, pcd, theta=weight_theta)

    res = {
        'skeleton_pcd': points,
        'root': points[root_indx],
        'joints': points[joints],
        'bones': bones,
        'pcd': pcd,
        'weights': weights,
    }

    return res

def visualise_skeletonizer(skeleton_points, root, joints, bones=None, pcd=None, weights=None, old_joints=None, old_bones=None, add_labels=True, add_vaild_bones=True, bone_width=10.0, point_size=10.0, joint_size=15.0, bone_color=[0.1, 0.1, 0.8]):
    cs = {
        'root': np.array([[1., 0., 0.]]),
        'joint': np.array([[0., 0., 1.]]),
        'bone': np.array([bone_color]),
        'point': np.array([[0., 0., 0.]])
    }

    # Add joints and root
    # joint_points = root.reshape(1, 3)
    # cols = cs['root']
    # joint_points = np.concatenate([joint_points, joints], axis=0)
    joint_points = joints
    cols = np.concatenate([cs['joint'], cs['joint'].repeat(len(joints) - 1, axis=0)], axis=0)

    # Add bones if available
    if bones is not None:
        col_bones = cs['bone'].repeat(len(bones), axis=0)
    else:
        col_bones = None
    
    # Add weights if available
    if weights is not None and pcd is not None:
        col_palette = np.array(color_palette("husl", weights.shape[1]))
        col_palette = np.random.rand(*col_palette.shape)
        
        # Add weights
        cols_weights = (np.expand_dims(weights, axis=-1) * col_palette).sum(axis=1)

        # Weight Visualisation
        weight_pcd = o3d.geometry.PointCloud()
        weight_pcd.points = o3d.utility.Vector3dVector(pcd)
        weight_pcd.colors = o3d.utility.Vector3dVector(cols_weights)
    else:
        weight_pcd = None

    # Skeleton Visualisation
    joint_pcd = o3d.geometry.PointCloud()
    joint_pcd.points = o3d.utility.Vector3dVector(joint_points)
    joint_pcd.colors = o3d.utility.Vector3dVector(cols)

    skeleton_pcd = o3d.geometry.PointCloud()
    skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_points)
    skeleton_pcd.colors = o3d.utility.Vector3dVector(cs['point'].repeat(len(skeleton_points), 0))


    # 在创建line_set之前添加验证
    if bones is not None and len(bones) > 0 and len(joints) > 0 and add_vaild_bones:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(joints)
        line_set.lines = o3d.utility.Vector2iVector(bones)
        line_set.colors = o3d.utility.Vector3dVector(col_bones)
        has_valid_bones = True
    else:
        has_valid_bones = False
        line_set = None
        if bones is None:
            print("Info: No bones provided, skipping bone visualization")
        elif len(bones) == 0:
            print("Warning: Empty bones array provided")
        elif len(joints) == 0:
            print("Warning: Empty joints array provided")


    gui.Application.instance.initialize()

    window = gui.Application.instance.create_window("Skeleton-Viewer", 1024, 750)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)

    window.add_child(scene)

    bp_material = rendering.MaterialRecord()
    bp_material.point_size = point_size

    mp_material = rendering.MaterialRecord()
    mp_material.point_size = joint_size  # 增大关节点的尺寸

    # 为骨骼连线创建材质
    bone_material = rendering.MaterialRecord()
    bone_material.shader = "unlitLine"  # 使用unlitLine着色器
    bone_material.line_width = bone_width  # 增大骨骼连线的宽度
    bone_material.base_color = [1.0, 1.0, 1.0, 1.0]  # 设置基础颜色

    if weight_pcd is not None:
        scene.scene.add_geometry("Weights", weight_pcd, bp_material)
    scene.scene.add_geometry("Skeleton", skeleton_pcd, mp_material)
    scene.scene.add_geometry("Joints", joint_pcd, mp_material)
    if has_valid_bones and line_set is not None:
        scene.scene.add_geometry("Bones", line_set, bone_material)  # 使用新的bone_material

    bounds = skeleton_pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())

    labels = [
        (root, 'root (j0)')
    ]

    joint_to_idx = {}
    for i in range(1, len(joints)): # skip root, joint == 0
        if old_joints is not None:
            x = np.where(np.all(joints[i] == old_joints, axis=1))[0][0]
        else:
            x = i
        joint_to_idx[i] = x
        labels.append((joints[i], f'j{x}'))

    if bones is not None and add_vaild_bones:
        for i in range(len(bones)):
            bs, be = bones[i]
            pos = (joints[bs] + joints[be]) / 2

            if old_bones is not None:
                x =  joint_to_idx[be]
            else:
                x = be

            labels.append((pos, f'b{x}'))        

    if add_labels:
        for item in labels:
            scene.add_3d_label(item[0], item[1])

    gui.Application.instance.run()

    
def check_bone_validity(start_pos, end_pos, point_cloud, num_samples=10, k=9, distance_threshold=0.01):
    """检查骨骼是否在点云内部
    
    通过在骨骼上均匀采样多个点，检查每个采样点是否都在点云内部来判断。
    
    参数:
        start_pos: (3,) 骨骼起点
        end_pos: (3,) 骨骼终点
        point_cloud: (N, 3) 点云
        num_samples: int, 在骨骼上采样的点数量
        k: int, 每个采样点检查的近邻数量
        distance_threshold: float, 允许的最大距离
    
    返回:
        bool: 骨骼是否在点云内部
    """
    from scipy.spatial import cKDTree
    
    # 构建KD树(只需构建一次)
    kdtree = cKDTree(point_cloud)
    
    # 在骨骼上均匀采样点
    t = np.linspace(0, 1, num_samples)[:, None]
    sample_points = start_pos[None, :] * (1 - t) + end_pos[None, :] * t
    
    # 对每个采样点:
    # 1. 找到k个最近邻
    # 2. 计算到这些近邻的平均距离
    # 3. 检查这些距离是否都小于阈值
    distances, _ = kdtree.query(sample_points, k=k)
    avg_distances = np.mean(distances, axis=1)
    
    # 所有采样点的平均距离都需要小于阈值
    return np.all(avg_distances < distance_threshold)

def find_root_from_weights(W, rest_bones_t):
    """通过权重矩阵W选择影响最大的骨骼点作为root点
    
    算法思路:
    1. 计算每个骨骼点影响的点的数量(权重大于阈值的点数)
    2. 计算每个骨骼点的总影响权重
    3. 计算每个骨骼点的平均影响范围(到受影响点的平均距离)
    4. 综合以上因素选择最佳root点
    
    参数:
        W: np.ndarray, shape (N, M) 权重矩阵, N是点云中的点数, M是骨骼点数
        rest_bones_t: np.ndarray, shape (M, 3) 骨骼点的3D坐标
        
    返回:
        int: root点的索引
    """
    # 设置权重阈值
    weight_threshold = 0.01
    
    # 计算每个骨骼点影响的点的数量
    influenced_points = (W > weight_threshold).sum(axis=0)  # shape (M,)
    
    # 计算每个骨骼点的总影响权重
    total_weights = W.sum(axis=0)  # shape (M,)
    
    # 归一化影响点数量和总权重
    influenced_points_norm = influenced_points / influenced_points.max()
    total_weights_norm = total_weights / total_weights.max()
    
    # 计算骨骼点的中心性(到其他骨骼点的平均距离的倒数)
    distances = np.sqrt(((rest_bones_t[:, None] - rest_bones_t[None, :]) ** 2).sum(axis=2))
    centrality = 1 / (distances.mean(axis=1) + 1e-6)  # 添加小值避免除零
    centrality_norm = centrality / centrality.max()
    
    # 综合评分
    # 这里我们给予不同指标不同的权重
    scores = (
        0.4 * influenced_points_norm +  # 影响点数量
        0.2 * total_weights_norm +      # 总影响权重
        1.0 * centrality_norm          # 中心性
    )
    
    # 返回得分最高的骨骼点索引
    root_idx = np.argmax(scores)
    
    return root_idx



def create_skeleton_from_center(rest_bones_t, W, rest_pose, bone_length=0.2, bone_heursitic=False, 
                              distance_threshold=0.2, point_cloud_threshold=0.01):
    """以点云中心为root构建骨架"""
    from scipy.sparse.csgraph import minimum_spanning_tree
    
    # 先将中心点作为root这一部分去掉，选出影响最大的点作为root点
    # # 计算点云中心
    # cloud_center = np.mean(rest_pose, axis=0)
    
    # # 将中心点添加到骨骼点集
    # points = np.vstack([cloud_center[None, :], rest_bones_t])
    points = rest_bones_t

    root_indx = find_root_from_weights(W, points)  # 中心点作为root
    num_bones = len(points)
    
    # 计算距离矩阵
    distances = np.sqrt(np.sum((points[:,None,:] - points[None,:,:])**2, axis=-1))
    
    # 构建邻接矩阵，考虑点云约束
    adjacency = np.zeros_like(distances, dtype=bool)
    for i in range(num_bones):
        for j in range(i+1, num_bones):
            if check_bone_validity(
                points[i], points[j], rest_pose, 
                distance_threshold=point_cloud_threshold
            ):
                adjacency[i,j] = adjacency[j,i] = True
    
    # 构建代价矩阵
    cost_matrix = np.where(adjacency, distances, np.inf)
    
    # 计算最小生成树
    mst = minimum_spanning_tree(cost_matrix)
    mst = mst.toarray()
    
    # 将MST转换为bones列表
    bones = []
    joints = [root_indx]
    visited = set([root_indx])
    
    def add_children(parent_idx):
        children = np.where((mst[parent_idx] > 0) | (mst[:, parent_idx] > 0))[0]
        for child_idx in children:
            if child_idx not in visited:
                visited.add(child_idx)
                joints.append(child_idx)
                bones.append([joints.index(parent_idx), joints.index(child_idx)])
                add_children(child_idx)
    
    # 从根节点开始构建树
    add_children(root_indx)
    
    # 启发式清理（如果需要）
    if bone_heursitic:
        # ... (保持原来的启发式清理代码不变)
        pass
    
    res = {
        'skeleton_pcd': points,
        'root': points[root_indx],
        'root_idx': root_indx,
        'joints': points[joints],
        'joints_idx': joints,
        'bones': bones,
    }
    
    return res


def create_skeleton_from_ssdr(rest_bones_t, W, rest_pose, bone_length=0.2, bone_heursitic=False, 
                            distance_threshold=0.2, k_neighbors=3, point_cloud_threshold=0.05):
    """通过最小生成树算法构建骨架，考虑点云约束
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    
    points = rest_bones_t
    num_bones = len(points)
    
    # 计算距离矩阵
    distances = np.sqrt(np.sum((points[:,None,:] - points[None,:,:])**2, axis=-1))
    
    # 构建邻接矩阵，考虑点云约束
    adjacency = np.zeros_like(distances, dtype=bool)
    for i in range(num_bones):
        for j in range(i+1, num_bones):
            # 检查距离是否在阈值内
            if check_bone_validity(
                points[i], points[j], rest_pose, 
                distance_threshold=point_cloud_threshold
            ):
                adjacency[i,j] = adjacency[j,i] = True
    
    # 构建代价矩阵
    cost_matrix = np.where(adjacency, distances, np.inf)
    
    # 可选：结合权重相关性
    # bone_correlation = np.abs(W.T @ W)
    # bone_correlation = bone_correlation / bone_correlation.max()
    # cost_matrix = np.where(adjacency, distances / (bone_correlation + 1e-6), np.inf)
    
    # 使用Floyd-Warshall算法找到根节点
    D = shortest_path(cost_matrix, directed=True, method='FW')
    root_indx = D.sum(1).argmin()
    
    # 计算最小生成树
    mst = minimum_spanning_tree(cost_matrix)
    mst = mst.toarray()
    
    # 将MST转换为bones列表
    bones = []
    joints = [root_indx]
    visited = set([root_indx])
    
    def add_children(parent_idx):
        children = np.where((mst[parent_idx] > 0) | (mst[:, parent_idx] > 0))[0]
        for child_idx in children:
            if child_idx not in visited:
                visited.add(child_idx)
                joints.append(child_idx)
                bones.append([joints.index(parent_idx), joints.index(child_idx)])
                add_children(child_idx)
    
    # 从根节点开始构建树
    add_children(root_indx)
    
    # 启发式清理（如果需要）
    if bone_heursitic:
        # 提取骨骼的起点和终点
        starts = np.array([bone[0] for bone in bones])
        tails = np.array([bone[1] for bone in bones])
        
        # 判断每个骨骼是否有子骨骼
        bone_has_child = []
        for i in range(len(bones)):
            bone_has_child.append(tails[i] in starts)
        
        bone_has_child = np.array(bone_has_child)
        del_indices = []
        
        for u_start in np.unique(starts):
            indx = np.where(u_start == starts)[0]
            
            if bone_has_child[indx].any():
                for i in indx:
                    if not bone_has_child[i]:
                        del_indices.append(i)
            else:
                distances_temp = []
                for i in indx:
                    bone = bones[i]
                    distances_temp.append(np.sqrt(np.sum(
                        (points[joints[bone[0]]] - points[joints[bone[1]]])**2
                    )))
                
                longest_indx = np.argmax(distances_temp)
                for i, ii in enumerate(indx):
                    if i != longest_indx:
                        del_indices.append(ii)
        
        # 从后向前删除冗余骨骼
        del_indices.sort()
        del_indices.reverse()
        for i in del_indices:
            del bones[i]
        
        # 更新关节点列表
        used_joints = set()
        for bone in bones:
            used_joints.add(joints[bone[0]])
            used_joints.add(joints[bone[1]])
        joints = [j for j in joints if j in used_joints]
        
        # 更新骨骼索引
        bones = [[joints.index(joints[b[0]]), joints.index(joints[b[1]])] for b in bones]
    
    res = {
        'skeleton_pcd': points,
        'root': points[root_indx],
        'joints': points[joints],
        'bones': bones,
    }
    
    return res
# 使用示例:
def test_create_skeleton_from_ssdr(result_path, output_path, bone_length=0.2, bone_heursitic=False, 
                            distance_threshold=0.2, point_cloud_threshold=0.05, visualize=False, save_result=False):
    # 加载SSDR结果
    results = np.load(result_path, allow_pickle=True).item()
    print(results.keys())
    rest_bones_t = results["rest_bones"]
    W = results["W"]
    rest_pose = results["rest_pose"]
    print(f"W shape: {W.shape}")
    print(f"rest_bones_t shape: {rest_bones_t.shape}")
    print(f"rest_pose shape: {rest_pose.shape}")
    #归一化
    rest_bones_t = rest_bones_t / np.max(np.abs(rest_bones_t))
    rest_pose = rest_pose / np.max(np.abs(rest_pose))
    
    # 创建骨骼结构
    skeleton = create_skeleton_from_center(rest_bones_t, W, rest_pose, bone_length=bone_length, bone_heursitic=bone_heursitic, 
                              distance_threshold=distance_threshold, point_cloud_threshold=point_cloud_threshold)
    
    # 添加数据验证
    print(f"Number of joints: {len(skeleton['joints'])}")
    print(f"Number of bones: {len(skeleton['bones'])}")
    print(f"Joints: {skeleton['joints_idx']}")
    print(f"First few bones: {skeleton['bones'][:3]}")
    
    
    # 确保所有骨骼索引都在有效范围内
    num_joints = len(skeleton['joints'])
    valid_bones = []
    for bone in skeleton['bones']:
        if 0 <= bone[0] < num_joints and 0 <= bone[1] < num_joints:
            valid_bones.append(bone)
    skeleton['bones'] = valid_bones
    
    if visualize:
        visualise_skeletonizer(
            skeleton_points=rest_bones_t,
            root=skeleton['root'],
            joints=skeleton['joints'],
            bones=skeleton['bones'],
            pcd=rest_pose,
            weights=W
        )

        visualise_skeletonizer(
            skeleton_points=rest_bones_t,
            root=skeleton['root'],
            joints=skeleton['joints'],
            bones=skeleton['bones'],
            pcd=rest_bones_t,
            weights=W
        )
    if save_result:
        np.save(output_path + "/skeleton.npy", skeleton)
    
    return skeleton




if __name__ == "__main__":
    # 读取TiNeuVox的信息
    # alpha_volume = np.load('./data/alpha_volume_f16.npy')
    # with open("./data/grid.txt", 'r') as f:  
    #     lines = f.readlines()
    #     for i in range(len(lines)):
    #         lines[i] = np.array(lines[i].replace('\n', '').split(','), dtype=float)
    # # bounding box信息和网格shape(x,y,z)
    # min = lines[0]
    # max = lines[1]
    # shape = lines[2].astype(int)

    # # 通过bounding box信息和网格shape生成网格坐标，与TiNeuVox的网格一致
    # xv, yv, zv = np.meshgrid(
    #     np.linspace(min[0], max[0], shape[0]),
    #     np.linspace(min[1], max[1], shape[1]),
    #     np.linspace(min[2], max[2], shape[2]),
    #     indexing='ij')

    # grid_xyz = np.concatenate([
    #     np.expand_dims(xv, axis=-1),
    #     np.expand_dims(yv, axis=-1),
    #     np.expand_dims(zv, axis=-1)
    # ], axis=-1)

    # # 创建骨架
    # res = create_skeleton(alpha_volume, grid_xyz, bone_length=10., sigma=1, weight_theta=0.03)
    # visualise_skeletonizer(*res.values())
    test_create_skeleton_from_ssdr("/home/wjx/research/code/GaussianAnimator/LBS-GS/tempexp/trex/ssdr/ssdr_result_iter20000.npy")