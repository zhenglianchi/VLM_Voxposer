import numpy as np

def quaternion_distance_euclidean(q1, q2):
    """
    计算两个四元数的欧几里得距离
    :param q1: 第一个四元数 [w, x, y, z]
    :param q2: 第二个四元数 [w, x, y, z]
    :return: 欧几里得距离
    """
    return np.linalg.norm(np.array(q1) - np.array(q2))

def quaternion_distance_angle(q1, q2):
    """
    计算两个四元数的角度距离（最小旋转角度）
    :param q1: 第一个四元数 [w, x, y, z]
    :param q2: 第二个四元数 [w, x, y, z]
    :return: 角度距离（弧度制）
    """
    # 计算四元数内积
    dot_product = np.dot(q1, q2)
    # 取绝对值以处理 q 和 -q 的等价性
    cos_phi = np.abs(dot_product)
    # 防止数值误差导致 cos_phi 超出 [-1, 1] 范围
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    # 计算角度距离
    angle_distance = 2 * np.arccos(cos_phi)
    return angle_distance

# 示例：计算两个四元数的距离
q1 = [0.707, 0.0, 0.707, 0.0]  # 绕Y轴旋转90度
q2 = [0.924, 0.0, 0.383, 0.0]  # 绕Y轴旋转45度

# 计算欧几里得距离
euclidean_dist = quaternion_distance_euclidean(q1, q2)
print(f"欧几里得距离: {euclidean_dist:.4f}")

# 计算角度距离
angle_dist = quaternion_distance_angle(q1, q2)
print(f"角度距离（弧度）: {angle_dist:.4f}")
print(f"角度距离（角度）: {np.degrees(angle_dist):.4f}°")