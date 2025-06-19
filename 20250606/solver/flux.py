import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from type_transform import trans_conservative2primitive
import config


def conflux_roe(u_l, u_r, s, gamma=config.GAMMA):
    """
    使用 Roe 格式计算二维欧拉方程的数值通量
    参数:
        U_L: 左侧守恒变量 [rho, rho_u, rho_v, rho_E]
        U_R: 右侧守恒变量 [rho, rho_u, rho_v, rho_E]
        gamma: 比热比
        s:面向量，shape = (2)
    返回:
        f: 面通量向量，shape = (4,)
    """
    # 1.计算守恒变量并旋转守恒变量
    m_rotate = compute_rotation_matrix_2d(s)
    u_l[1:3] = m_rotate @ u_l[1:3]
    u_r[1:3] = m_rotate @ u_r[1:3]

    # 2.计算垂直于边坐标系的数值通量
    flux = compute_roe_flux_2d_local(u_l, u_r, gamma)

    # 3.旋转回去
    m_rotate = compute_rotation_back_matrix_2d(s)
    flux[1:3] = m_rotate.T @ flux[1:3]
    return flux


def compute_roe_flux_2d_local(u_l, u_r, gamma=1.4, delta_coef=0.1):
    """
    使用 Roe 格式计算二维欧拉方程在法向坐标系下的数值通量
    参数:
        U_L: 左侧守恒变量 [rho, rho_u, rho_v, rho_E]
        U_R: 右侧守恒变量 [rho, rho_u, rho_v, rho_E]
        gamma: 比热比
        Vt: 网格速度（用于滑动网格或 ALE）
        grd_type: 网格类型（0: 运动网格，1: 静止网格）
        delta_coef: 熵修正的控制系数
    返回:
        F: 面通量向量，shape = (4,)
    """

    def roe_average(WL, WR):
        rho_L, u_L, v_L, H_L = WL
        rho_R, u_R, v_R, H_R = WR
        sqrt_rho_L = np.sqrt(rho_L)
        sqrt_rho_R = np.sqrt(rho_R)
        denom = sqrt_rho_L + sqrt_rho_R

        u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
        v_roe = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / denom
        H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom
        a_roe = np.sqrt((gamma - 1.0) * (H_roe - 0.5 * (u_roe ** 2 + v_roe ** 2)))
        return np.array([a_roe, u_roe, v_roe, H_roe])

    # 左右原始变量
    WL, p_L = trans_conservative2primitive(u_l, gamma, return_pressure=True)
    WR, p_R = trans_conservative2primitive(u_r, gamma, return_pressure=True)

    W_roe = roe_average(WL, WR)
    a, u, v, H = W_roe
    V2 = u ** 2 + v ** 2

    # 波速特征值
    lam = np.array([u - a, u, u, u + a])
    delta = delta_coef * (abs(u) + a)

    # 熵修正
    for i in range(4):
        if abs(lam[i]) < delta:
            lam[i] = 0.5 * (lam[i] ** 2 / delta + delta)

    # 差分
    dU = u_r - u_l
    drho = dU[0]
    dmomentum = dU[1:3]
    dE = dU[3]

    alpha = np.zeros(4)
    alpha[1:3] = dmomentum - W_roe[1:3] * drho
    rm = dE - np.dot(W_roe[1:3], alpha[1:3])
    alpha[0] = 0.5 * ((drho * (W_roe[3] - u ** 2) + u * dmomentum[0] - rm) / a ** 2)
    alpha[3] = drho - alpha[0]
    alpha *= lam  # 波强乘特征值

    # 右特征向量矩阵 R
    R = np.zeros((4, 4))
    R[:, 0] = [1, u - a, v, H - u * a]
    R[:, 1] = [0, 0, 1, v]
    R[:, 2] = [1, u, v, 0.5 * (V2)]
    R[:, 3] = [1, u + a, v, H + u * a]

    dF = R @ alpha

    # 物理通量
    F_L = np.array([
        u_l[1],                         # rho * u
        u_l[1] * WL[1] + p_L,           # rho*u*u + p
        u_l[1] * WL[2],                 # rho*u*v
        WL[3] * u_l[1]                  # rho*H*u
    ])
    F_R = np.array([
        u_r[1],
        u_r[1] * WR[1] + p_R,
        u_r[1] * WR[2],
        WR[3] * u_r[1]
    ])

    return 0.5 * (F_L + F_R - dF)


def compute_rotation_matrix_2d(normal):
    """
    构造二维旋转矩阵 T，使得局部坐标系 x' 轴沿 normal 方向，y' 轴为其正交方向。
    参数:
        normal: 面法向量，shape = (2,)
    返回:
        T: shape = (2, 2) 的旋转矩阵，列向量为局部坐标系的基向量。
           用于将 [u, v] 投影到局部 [u_n, u_t]（法向、切向）速度分量。
    """
    nx, ny = normal
    norm = np.sqrt(nx**2 + ny**2)
    if norm == 0:
        raise ValueError("normal vector cannot be zero")
    nx, ny = nx / norm, ny / norm  # 归一化

    # 第一列是法向方向，第二列是与其正交的切向方向
    m_rotate = np.array([[nx, ny],
                         [-ny,  nx]])
    return m_rotate


def compute_rotation_back_matrix_2d(normal):
    """
    构造二维旋转矩阵 T，使得局部坐标系 x' 轴沿 normal 方向，y' 轴为其正交方向。
    参数:
        normal: 面法向量，shape = (2,)
    返回:
        T: shape = (2, 2) 的旋转矩阵，列向量为局部坐标系的基向量。
           用于将 [u, v] 投影到局部 [u_n, u_t]（法向、切向）速度分量。
    """
    nx, ny = normal
    # 第一列是法向方向，第二列是与其正交的切向方向
    m_rotate = np.array([[nx, ny],
                         [-ny,  nx]])
    return m_rotate


def reconstruct_interface_state(blocks, id0, dir_fix, m=config.N_C, gamma=config.GAMMA):
    """
    重构截面两侧的守恒状态变量 U_L, U_R
    参数:
        blk: 一个 block 对象，包含 .U, .volume, .S, .W (原始变量或特征变量)
        id0: 中心单元的索引，例如 [i, j]
        dir_fix: 插值方向 (1/2,1/-1) 第一维表示 i/j方向，第二维表示正负方向
        m: 守恒量个数
        gamma: 比热比
    返回:
        U_stat: shape (m, 2)，左 (0)、右 (1) 两侧守恒变量
    """
    # 获取 stencil
    w_stencil, w_roe = get_w_stencil(blocks, id0, dir_fix, m, gamma=gamma, stencil_size=4, bias=1)

    # 根据体积/法向计算 eps（模拟 Reconepsmin_c + ReconepsScal_c）
    geo = blocks["geo"]
    vol = geo[tuple(id0)][2]

    if dir_fix == [1, 1]:
       # S2
       S_vec = geo[tuple(id0)][5:7]
    elif dir_fix == [1, -1]:
       # S4
       S_vec = geo[tuple(id0)][9:11]
    elif dir_fix == [2, 1]:
       # S3
       S_vec = geo[tuple(id0)][7:9]
    elif dir_fix == [2, -1]:
       # S1
       S_vec = geo[tuple(id0)][3:5]

    length = vol / np.linalg.norm(S_vec)
    eps = max(1e-6, 0.1 * length ** 1.0)

    wl = muscl_vanAlbada2(w_stencil[:, 0:3], wid=1, eps=eps)
    wr = muscl_vanAlbada2(w_stencil[:, 1:4], wid=2, eps=eps)

    # 判断是否守恒量合法（ρ>0, E>0）
    if wl[0] <= 0 or wr[0] <= 0 or wl[-1] <= 0 or wr[-1] <= 0:
        # 退化为一阶近邻取值
        u_l = blocks['fluid'][id0[0], id0[1], :]
        id1 = list(id0)
        id1[dir_fix[0] - 1] = id1[dir_fix[0] - 1] + dir_fix[1]
        u_r = blocks['fluid'][id1[0], id1[1], :]
        u_l[0] = abs(u_l[0])
        u_r[0] = abs(u_r[0])
        u_l[-1] = abs(u_l[-1])
        u_r[-1] = abs(u_r[-1])
    else:
        mr = compute_right_eigenvector(w_roe, gamma)
        u_l = mr @ wl
        u_r = mr @ wr

    return np.stack([u_l, u_r], axis=1)  # shape (m, 2)


def get_w_stencil(blocks, id0, dir_fix, m, gamma=1.4, stencil_size=4, bias=1):
    """
    提取用于插值的特征变量模板（W_stencil）和 Roe 平均状态（W_roe）
    参数:
        blk: dict，包含 block 数据，至少包含：
            - 'U': ndarray, shape = (m, ni, nj[, nk])，守恒变量
        id0: list[int], 当前 cell 的索引[i, j]
        dir_fix: 插值方向 (1/2,1/-1) 第一维表示 i/j方向，第二维表示正负方向
        m: int, 守恒变量个数
        gamma: float, 比热比，默认 1.4
        stencil_size: int, stencil 点数（默认 3）
        bias: int, stencil 起始偏移量（默认 1）
    返回:
        W_stencil: ndarray, shape = (m, stencil_size)，特征变量模板
        W_roe: ndarray, shape = (m,)，Roe 平均状态变量（用于特征矩阵）
    """
    u = blocks['fluid']
    w_stencil = np.zeros((m, stencil_size))
    w_roe = np.zeros(m)

    # 1. 提取相邻两个 cell 的守恒变量用于 Roe 平均
    id1 = id0.copy()
    id2 = id0.copy()
    id2[dir_fix[0] - 1] = id2[dir_fix[0] - 1] + dir_fix[1]
    u1 = get_u_at(u, id1)
    u2 = get_u_at(u, id2)

    # 2. 计算 Roe 平均状态变量（W_roe）
    w1 = trans_conservative2primitive(u1, gamma)
    w2 = trans_conservative2primitive(u2, gamma)
    sqrt_rho1 = np.sqrt(abs(w1[0]))
    sqrt_rho2 = np.sqrt(abs(w2[0]))

    # pho_hat
    rho_roe = sqrt_rho1 * sqrt_rho2

    rm_ratio = sqrt_rho2 / sqrt_rho1
    w_roe[1:] = (w1[1:] + rm_ratio * w2[1:]) / (1 + rm_ratio)

    # 声速 a = sqrt((gamma-1) * (H - 0.5 * (u^2 + v^2)))
    v2_roe = np.dot(w_roe[1:m - 1], w_roe[1:m - 1])  # u^2 + v^2
    h_hat = w_roe[m - 1]
    w_roe[0] = np.sqrt((gamma - 1) * abs(h_hat - 0.5 * v2_roe))

    # 3. 构造左特征矩阵并投影为特征变量
    mr = compute_right_eigenvector(w_roe, gamma)
    ml = np.linalg.inv(mr)

    # 4. 构造 stencil
    for iii in range(stencil_size):
        id_stencil = id0.copy()
        id_stencil[dir_fix[0] - 1] += (iii * dir_fix[1]) - bias
        u_sten = get_u_at(u, id_stencil)
        w_stencil[:, iii] = ml @ u_sten

    return w_stencil, w_roe


def muscl_vanAlbada2(w_stencil, wid, eps):
    """
    二阶 MUSCL 插值 + van Albada 限制器（带 kai=1/3 非线性校正）
    参数：
        W_stencil: ndarray, shape=(m, 3)
            m 个变量，每个变量的 stencil（3 点），分别是：
              - WID=1: W_{i-1}, W_i, W_{i+1}
              - WID=2: W_i, W_{i+1}, W_{i+2}
        WID: int
            1 表示构造左状态 WL（偏向当前），2 表示构造右状态 WR（偏向前一个）
        eps: float
            限制器中的小量，避免除零或在极小梯度下退化
    返回：
        W_stat: ndarray, shape=(m,)
            插值后的状态（WL 或 WR）
    """
    m = w_stencil.shape[0]

    # 差分：delta- 与 delta+
    delta_m = w_stencil[:, 1] - w_stencil[:, 0]  # delta-
    delta_p = w_stencil[:, 2] - w_stencil[:, 1]  # delta+

    # van Albada 斜率限制器因子 rm
    numerator = 2.0 * delta_m * delta_p + eps
    denominator = delta_m ** 2 + delta_p ** 2 + eps
    rm = numerator / denominator  # shape = (m,)

    # 系数系数 coef = 1/4 * (3 - 2 * WID)
    coef = 0.25 * (3 - 2 * wid)

    # 校正项：kai = 1/3
    kai_term = (delta_p if wid == 1 else delta_m) - (delta_m if wid == 1 else delta_p)
    correction = (delta_m + delta_p) + (rm * kai_term) / 3.0

    # 插值
    w_stat = w_stencil[:, 1] + coef * rm * correction

    return w_stat


def get_u_at(u, idx):
    """从守恒量数组 U 中获取指定位置的值"""
    if len(idx) == 2:
        return u[idx[0], idx[1], :]
    elif len(idx) == 3:
        return u[idx[0], idx[1], idx[2], :]
    else:
        raise ValueError("Unsupported dimension")


def compute_right_eigenvector(w, gamma=1.4):
    """
    Roe 平均流量下欧拉对流通量雅可比矩阵的右特征向量矩阵
    参数：
        W: [a, u, v, H]（Roe 平均声速、速度、总焓）
    返回：
        TR: ndarray (4, 4)，右特征向量矩阵
    """
    a, u, v, H = w
    tr = np.zeros((4, 4))
    ua = u * a

    # λ1 = u - a
    tr[:, 0] = [1.0, u - a, v, H - ua]

    # λ2 = u
    tr[:, 1] = [1.0, u, v, 0.5 * (u**2 + v**2)]

    # λ3 = u
    tr[:, 2] = [0.0, 0.0, 1.0, v]

    # λ4 = u + a
    tr[:, 3] = [1.0, u + a, v, H + ua]

    return tr
