import numpy as np
import sys
import os
from numba import njit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from type_transform import trans_conservative2primitive
from type_transform import trans_primitive2conservative
import config


@njit
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


@njit
def compute_ausm_flux_2d_local(w_l, w_r, s, gamma):
    rho_l, u_l, v_l, p_l = trans_conservative2primitive(w_l, gamma)
    rho_r, u_r, v_r, p_r = trans_conservative2primitive(w_r, gamma)

    # 面向量单位化（如需）暂略，默认单位化
    n0, n1 = s[0], s[1]

    # 法向速度
    v_n_l = u_l * n0 + v_l * n1
    v_n_r = u_r * n0 + v_r * n1

    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)

    ma_l = v_n_l / a_l
    ma_r = v_n_r / a_r

    p_l_plus = 0.0
    ma_l_plus = 0.0
    p_r_plus = 0.0
    ma_r_plus = 0.0

    if ma_l >= 1.0:
        p_l_plus = p_l
        ma_l_plus = ma_l
    elif np.abs(ma_l) < 1.0:
        ma_l_plus = 0.25 * (ma_l + 1.0) ** 2
        p_l_plus = 0.25 * p_l * (ma_l + 1.0) ** 2 * (2.0 - ma_l)

    if ma_r <= -1.0:
        p_r_plus = p_r
        ma_r_plus = ma_r
    elif np.abs(ma_r) < 1.0:
        ma_r_plus = -0.25 * (ma_r - 1.0) ** 2
        p_r_plus = 0.25 * p_r * (ma_r - 1.0) ** 2 * (2.0 + ma_r)

    ma_face = ma_l_plus + ma_r_plus

    h_l = (gamma / (gamma - 1.0)) * (p_l / rho_l) + 0.5 * (u_l ** 2 + v_l ** 2)
    h_r = (gamma / (gamma - 1.0)) * (p_r / rho_r) + 0.5 * (u_r ** 2 + v_r ** 2)

    f_l = np.zeros(4)
    f_r = np.zeros(4)
    f_p = np.zeros(4)

    f_l[0] = rho_l * a_l
    f_l[1] = f_l[0] * u_l
    f_l[2] = f_l[0] * v_l
    f_l[3] = f_l[0] * h_l

    f_r[0] = rho_r * a_r
    f_r[1] = f_r[0] * u_r
    f_r[2] = f_r[0] * v_r
    f_r[3] = f_r[0] * h_r

    p_sum = p_l_plus + p_r_plus
    f_p[1] = n0 * p_sum
    f_p[2] = n1 * p_sum

    f_face = np.zeros(4)
    for i in range(4):
        f_face[i] = 0.5 * ma_face * (f_l[i] + f_r[i]) \
                    - 0.5 * np.abs(ma_face) * (f_r[i] - f_l[i]) \
                    + f_p[i]

    return f_face


@njit
def reconstruct_interface_state(
    fluid_ext, geo, id0, idp, dir_fix, m, gamma
):
    """
    使用数组重构截面两侧的守恒状态变量 U_L, U_R
    参数:
        fluid_ext: 扩展后的守恒量数组（含 ghost），shape=(ni+2g, nj+2g, m)
        geo: 原始网格几何数组，shape=(ni, nj, 11)
        id0: 当前在 fluid_ext 中的索引 [i, j]
        idp: 对应 geo 中的非 ghost 网格索引 [i, j]
        dir_fix: 插值方向 [1或2, ±1]
        m: 守恒量个数
        gamma: 比热比
    返回:
        w_stat: shape=(m, 2)，左右两侧守恒变量
    """
    # 获取 stencil
    p_stencil = get_p_stencil(fluid_ext, id0, dir_fix, m, gamma=gamma, stencil_size=4, bias=1)

    # 提取体积和面法向量
    vol = geo[idp[0], idp[1], 2]

    if dir_fix == [1, 1]:
        s_vec = geo[idp[0], idp[1], 5:7]  # S2
    elif dir_fix == [1, -1]:
        s_vec = geo[idp[0], idp[1], 9:11]  # S4
    elif dir_fix == [2, 1]:
        s_vec = geo[idp[0], idp[1], 7:9]  # S3
    elif dir_fix == [2, -1]:
        s_vec = geo[idp[0], idp[1], 3:5]  # S1
    else:
        raise ValueError("Invalid dir_fix")

    # 计算 eps（避免除零）
    length = vol / np.linalg.norm(s_vec)
    eps = max(1e-6, 1.0 * length ** 2.5)

    # MUSCL 插值（宽度3）
    pl = muscl(p_stencil[:, 0:3], wid=1, eps=eps)
    pr = muscl(p_stencil[:, 1:4], wid=2, eps=eps)

    # 状态合法性检查
    if pl[0] <= 0 or pr[0] <= 0 or pl[-1] <= 0 or pr[-1] <= 0:
        # 退化为一阶近似
        u_l = fluid_ext[id0[0], id0[1], :]
        id1 = list(id0)
        id1[dir_fix[0] - 1] += dir_fix[1]
        u_r = fluid_ext[id1[0], id1[1], :]

        u_l = u_l.copy()
        u_r = u_r.copy()
        u_l[0] = abs(u_l[0])
        u_r[0] = abs(u_r[0])
        u_l[-1] = abs(u_l[-1])
        u_r[-1] = abs(u_r[-1])
    else:
        u_l = trans_primitive2conservative(pl, gamma)
        u_r = trans_primitive2conservative(pr, gamma)

    w_stat = np.empty((m, 2))
    for i in range(m):
        w_stat[i, 0] = u_l[i]
        w_stat[i, 1] = u_r[i]

    return w_stat


@njit
def get_p_stencil(u, id0, dir_fix, m, gamma=1.4, stencil_size=4, bias=1):
    """
    提取插值模板（原始变量）
    参数：
        u: ndarray，守恒变量数组 (ni+2g, nj+2g, m)
    """
    p_stencil = np.zeros((m, stencil_size))
    for i in range(stencil_size):
        id_stencil0 = id0[0]
        id_stencil1 = id0[1]
        if dir_fix[0] == 1:
            id_stencil0 += dir_fix[1] * (i - bias)
        else:
            id_stencil1 += dir_fix[1] * (i - bias)
        vals = trans_conservative2primitive(u[id_stencil0, id_stencil1], gamma)
        for k in range(m):
            p_stencil[k, i] = vals[k]
    return p_stencil

@njit
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
