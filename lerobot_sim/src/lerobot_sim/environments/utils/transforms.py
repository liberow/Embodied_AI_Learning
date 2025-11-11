import numpy as np

# ==============================
# 位姿矩阵（4x4）相关函数
#   - T: 4x4 矩阵 T 表示一个刚体在空间中的位姿
#   - p: 平移向量就是物体在空间的 xyz 坐标
#   - R: 旋转矩阵表示物体的朝向
# ==============================

def transform_to_pose(T: np.ndarray):
    """
    从变换矩阵 T 提取平移向量 p 和旋转矩阵 R
    参数:
        T: 4x4 numpy 数组
    返回:
        p: (3,) 平移向量
        R: (3,3) 旋转矩阵
    """
    p = T[:3, 3]
    R = T[:3, :3]
    return p, R

def transform_to_position(T: np.ndarray):
    """
    从变换矩阵 T 提取平移向量 p
    """
    return T[:3, 3]

def transform_to_rotation(T: np.ndarray):
    """
    从变换矩阵 T 提取旋转矩阵 R
    """
    return T[:3, :3]

def pose_to_transform(p: np.ndarray, R: np.ndarray):
    """
    已知平移向量和旋转矩阵，构造 4x4 的变换矩阵 T
    """
    T = np.block([
        [R, p.reshape(3, 1)], 
        [np.zeros((1, 3)), 1]
    ])
    return T

# ==============================
# 欧拉角（roll, pitch, yaw）相关函数
#   - rpy: 欧拉角，表示物体的朝向
#   - roll: 绕 x 轴旋转
#   - pitch: 绕 y 轴旋转
#   - yaw: 绕 z 轴旋转
# ==============================

def rpy_to_rotation(rpy: np.ndarray):
    """
    欧拉角 roll, pitch, yaw -> 旋转矩阵
    参数:
        rpy: (3,) array, 单位: 弧度
    返回:
        R: (3,3) 旋转矩阵
    """
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R

def rpy_to_rotation_ordered(rpy: np.ndarray, order=[0, 1, 2]):
    """
    欧拉角 -> 旋转矩阵，支持自定义旋转顺序
    order: [0,1,2] 对应 x-roll, y-pitch, z-yaw 的顺序
    """
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    matrices = [Rx, Ry, Rz]
    R = matrices[order[0]] @ matrices[order[1]] @ matrices[order[2]]
    return R

def rotation_to_rpy(R: np.ndarray, unit='rad'):
    """
    旋转矩阵 -> 欧拉角
    unit: 'rad' 或 'deg'
        rad: 弧度
        deg: 度
    """
    roll  = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    rpy = np.array([roll, pitch, yaw])
    if unit == 'deg':
        rpy = np.degrees(rpy)
    return rpy

# ==============================
# 四元数相关函数
# ==============================

def rotation_to_quaternion(R: np.ndarray):
    """
    旋转矩阵 -> 四元数 [w, x, y, z]
    """
    R = np.asarray(R, dtype=np.float64)
    K = np.zeros((4, 4))
    K[0,0] = R[0,0]-R[1,1]-R[2,2]
    K[1,0] = R[0,1]+R[1,0]; K[1,1] = R[1,1]-R[0,0]-R[2,2]
    K[2,0] = R[0,2]+R[2,0]; K[2,1] = R[1,2]+R[2,1]; K[2,2] = R[2,2]-R[0,0]-R[1,1]
    K[3,0] = R[2,1]-R[1,2]; K[3,1] = R[0,2]-R[2,0]; K[3,2] = R[1,0]-R[0,1]; K[3,3] = R[0,0]+R[1,1]+R[2,2]
    K /= 3.0

    vals, vecs = np.linalg.eigh(K)
    q = vecs[:, np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q

def quaternion_to_rotation(q: np.ndarray):
    """
    四元数 -> 旋转矩阵
    q: [w, x, y, z]
    """
    w, x, y, z = q
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])
    return R

# ==============================
# 坐标系与旋转操作
# ==============================

def skew(vector: np.ndarray):
    """
    向量 -> 反对称矩阵
    """
    x, y, z = vector
    return np.array([[0, -z, y],[z,0,-x],[-y,x,0]])

def rodrigues(axis: np.ndarray, angle_rad: float):
    """
    Rodrigues 公式: 旋转轴 + 旋转角 -> 旋转矩阵
    """
    axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    R = np.eye(3) + np.sin(angle_rad)*K + (1-np.cos(angle_rad))*(K@K)
    return R

def get_rotation_from_two_points(p_from: np.ndarray, p_to: np.ndarray):
    """
    根据两个点计算旋转矩阵，使向量 p_from->p_to 对齐
    """
    if np.linalg.norm(p_to - p_from) < 1e-8:
        return np.eye(3)
    v_from = np.array([1e-10, -1e-10, 1.0])
    v_to = (p_to - p_from)/np.linalg.norm(p_to - p_from)
    v_cross = np.cross(v_from, v_to)
    S = skew(v_cross)
    if np.linalg.norm(v_cross) < 1e-12:
        return np.eye(3)
    R = np.eye(3) + S + S@S*(1-np.dot(v_from, v_to))/(np.linalg.norm(v_cross)**2)
    return R

# ==============================
# 坐标系转换 (Y-up Z-front -> Z-up X-front)
# ==============================

def rotation_yuzf_to_zuxf(R: np.ndarray):
    """
    CMU-MoCap 坐标系(Y-up Z-front) -> 通用机器人坐标系(Z-up X-front)
    """
    R_offset = rpy_to_rotation(np.radians([-90,0,-90]))
    return R_offset @ R

def transform_yuzf_to_zuxf(T: np.ndarray):
    """
    同上，但处理 4x4 变换矩阵
    """
    p, R = transform_to_pose(T)
    return pose_to_transform(p, rotation_yuzf_to_zuxf(R))

# ==============================
# 深度图 -> 点云
# ==============================

def depth_to_pointcloud(depth_img: np.ndarray, cam_matrix: np.ndarray):
    """
    将深度图转换为点云
    输出: [H, W, 3] xyz 坐标
    """
    fx, fy = cam_matrix[0,0], cam_matrix[1,1]
    cx, cy = cam_matrix[0,2], cam_matrix[1,2]
    h, w = depth_img.shape
    indices = np.indices((h, w), dtype=np.float32).transpose(1,2,0)

    z = depth_img
    x = (indices[...,1]-cx) * z / fx
    y = (indices[...,0]-cy) * z / fy
    # 注意坐标轴调整
    xyz = np.stack([z, -x, -y], axis=-1)
    return xyz
