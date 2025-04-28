import numpy as np

def quaternion_slerp(q0, q1, t):
    """
    手动实现球面线性插值 (SLERP)。
    :param q0: 起始四元数 [x, y, z, w]
    :param q1: 目标四元数 [x, y, z, w]
    :param t: 插值参数，范围 [0, 1]
    :return: 插值后的四元数 [x, y, z, w]
    """
    # 确保输入是 numpy 数组
    q0 = np.array(q0)
    q1 = np.array(q1)

    # 计算点积以检测夹角
    dot = np.dot(q0, q1)

    # 如果点积小于 0，则反转 q1 以选择最短路径
    if dot < 0:
        q1 = -q1
        dot = -dot

    # 夹角接近 0 或 π 时，退化为线性插值
    if dot > 0.9995:
        return q0 * (1 - t) + q1 * t

    # 计算夹角 theta
    theta = np.arccos(dot)

    # 使用 SLERP 公式计算插值
    sin_theta = np.sin(theta)
    q_interp = (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1

    return q_interp

def interpolate_quaternions(q_start, q_end, num_points=10):
    """
    在两个四元数之间生成插值轨迹。
    :param q_start: 起始四元数 [x, y, z, w]
    :param q_end: 目标四元数 [x, y, z, w]
    :param num_points: 插值点的数量（包括起点和终点）
    :return: 包含插值四元数的列表
    """
    t_values = np.linspace(0, 1, num_points)  # 生成插值参数
    interpolated_quaternions = []

    for t in t_values:
        q_interp = quaternion_slerp(q_start, q_end, t)
        interpolated_quaternions.append(q_interp)

    return interpolated_quaternions

# 示例：定义起始四元数和目标四元数
q_current = [0, 0, 0, 1]  # 当前四元数 [x, y, z, w]
q_target = [0, 1, 0, 0]   # 目标四元数 [x, y, z, w]

# 在当前四元数和目标四元数之间生成 10 个点（包括起点和终点）
trajectory = interpolate_quaternions(q_current, q_target, num_points=10)

# 打印结果
for i, q in enumerate(trajectory):
    print(f"Point {i}: {q}")