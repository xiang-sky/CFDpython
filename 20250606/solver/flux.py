import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from type_transform import trans_conservative2primitive
from type_transform import trans_primitive2conservative
import config


def conflux_ausm(u_l, u_r, s, gamma=config.GAMMA):
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

    flux = compute_ausm_flux_2d_local(u_l, u_r, s, gamma)

    return flux


def compute_ausm_flux_2d_local(w_l, w_r, s, gamma=1.4):
    """
    使用 van Leer FVS 格式计算二维欧拉方程在法向坐标系下的数值通量（局部）。
    输入:
        u_l: ndarray (4,)  左侧守恒变量 [rho, rho*u, rho*v, rho*E]
        u_r: ndarray (4,)  右侧守恒变量
        gamma: 比热比
    返回:
        flux: ndarray (4,)  数值通量
    """
    # 守恒变量转原始变量
    pl = trans_conservative2primitive(w_l, gamma, return_pressure=True)
    pr = trans_conservative2primitive(w_r, gamma, return_pressure=True)

    rho_l, u_l, v_l, p_l = pl
    rho_r, u_r, v_r, p_r = pr

    # 面向量单位化
    n_unit = s
    # n_unit = s / np.linalg.norm(s)

    # 求逆变速度
    v_n_l = u_l * n_unit[0] + v_l * n_unit[1]
    v_n_r = u_r * n_unit[0] + v_r * n_unit[1]

    # 两侧声速
    a_l = np.sqrt((gamma * p_l) / rho_l)
    a_r = np.sqrt((gamma * p_r) / rho_r)

    # 两侧马赫数(使用逆变速度)
    ma_l = v_n_l / a_l
    ma_r = v_n_r / a_r

    # 两侧分裂马赫数
    p_l_plus = 0
    ma_l_plus = 0
    p_r_plus = 0
    ma_r_plus = 0

    if ma_l >= 1:
        p_l_plus = p_l
        ma_l_plus = ma_l
    elif abs(ma_l) < 1:
        p_l_plus = 0.25 * p_l * ((ma_l + 1) ** 2) * (2 - ma_l)
        ma_l_plus = 0.25 * ((ma_l + 1) ** 2)
    elif ma_l < -1:
        p_l_plus = 0
        ma_l_plus = 0

    if ma_r >= 1:
        p_r_plus = 0
        ma_r_plus = 0
    elif abs(ma_r) < 1:
        p_r_plus = 0.25 * p_r * ((ma_r - 1) ** 2) * (2 + ma_r)
        ma_r_plus = -0.25 * ((ma_r - 1) ** 2)
    elif ma_r < -1:
        p_r_plus = p_r
        ma_r_plus = ma_r

    ma_face = ma_l_plus + ma_r_plus

    h_l = (gamma / (gamma - 1)) * (p_l / rho_l) + 0.5 * (u_l ** 2 + v_l ** 2)
    h_r = (gamma / (gamma - 1)) * (p_r / rho_r) + 0.5 * (u_r ** 2 + v_r ** 2)

    f_l = np.array([rho_l * a_l,
                    rho_l * a_l * u_l,
                    rho_l * a_l * v_l,
                    rho_l * a_l * h_l
                    ])

    f_r = np.array([rho_r * a_r,
                    rho_r * a_r * u_r,
                    rho_r * a_r * v_r,
                    rho_r * a_r * h_r
                    ])

    f_p = np.array([0.0,
                    n_unit[0] * (p_l_plus + p_r_plus),
                    n_unit[1] * (p_l_plus + p_r_plus),
                    0.0
                    ])

    # 总通量
    f_face = (0.5 * ma_face * (f_l + f_r)) - (0.5 * abs(ma_face) * (f_r - f_l)) + f_p

    return f_face


def reconstruct_interface_state(blocks, id0, idp,  dir_fix, m=config.N_C, gamma=config.GAMMA):
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
    p_stencil = get_p_stencil(blocks, id0, dir_fix, m, gamma=gamma, stencil_size=4, bias=1)

    # 根据体积/法向计算 eps
    geo = blocks["geo"]
    vol = geo[tuple(idp)][2]

    if dir_fix == [1, 1]:
        # S2
        s_vec = geo[tuple(idp)][5:7]
    elif dir_fix == [1, -1]:
        # S4
        s_vec = geo[tuple(idp)][9:11]
    elif dir_fix == [2, 1]:
        # S3
        s_vec = geo[tuple(idp)][7:9]
    elif dir_fix == [2, -1]:
        # S1
        s_vec = geo[tuple(idp)][3:5]

    length = vol / np.linalg.norm(s_vec)
    eps = max(1e-6, 1.0 * length ** 2.5)  # Reconepsmin_c=1e-6, ReconepsScal_c=1.0, ReconepsExp_c=2.5

    pl = muscl(p_stencil[:, 0:3], wid=1, eps=eps)
    pr = muscl(p_stencil[:, 1:4], wid=2, eps=eps)

    # 判断是否守恒量合法（ρ>0, E>0）
    if pl[0] <= 0 or pr[0] <= 0 or pl[-1] <= 0 or pr[-1] <= 0:
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
        u_l = trans_primitive2conservative(pl, gamma)
        u_r = trans_primitive2conservative(pr, gamma)

    return np.stack([u_l, u_r], axis=1)  # shape (m, 2)


def get_p_stencil(blocks, id0, dir_fix, m, gamma=1.4, stencil_size=4, bias=1):
    """
    提取用于插值的特征变量模板（W_stencil）和 Roe 平均状态（W_roe）
    参数:
        blk: dict，包含 block 数据，至少包含：
            - 'U': ndarray, shape = (m, ni, nj[, nk])，守恒变量
        id0: list[int], 当前 cell 的索引[i, j]
        dir_fix: 插值方向 (1/2,1/-1) 第一维表示 i/j方向，第二维表示正负方向
        m: int, 守恒变量个数
        gamma: float, 比热比，默认 1.4
        stencil_size: int, stencil 点数（默认 4）
        bias: int, stencil 起始偏移量（默认 1）
    返回:
        p_stencil: ndarray, shape = (m, stencil_size)，守恒变量模板
    """
    u = blocks['fluid']
    p_stencil = np.zeros((m, stencil_size))

    for i in range(stencil_size):
        id_stencil = id0.copy()
        id_stencil[dir_fix[0] - 1] += dir_fix[1] * (i - bias)
        p_stencil[:, i] = trans_conservative2primitive(u[tuple(id_stencil)], gamma)

    return p_stencil


def muscl(w_stencil, wid, eps):
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
