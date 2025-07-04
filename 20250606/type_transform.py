import numpy as np
import config


def trans_list2numpy_2d(blocks, N_C):
    """
    创建用于数值计算的数组
    参数：
        blocks (list of dict): 每个 block 是一个包含网格和边界条件的字典。
        N_DIM: 计算维数
        N_C: 守恒量个数
    """
    block_cal = []

    for blk_id, blk in enumerate(blocks):
        ni, nj = blk['xc'].shape
        # 存储流场守恒量
        numpy_cal = np.zeros((ni, nj, N_C))

        # 存储网格几何量 1,2为xc、yc. 3为volume. 4,5,6,7,8,9,10,11为S1、S2、S3、S4
        geo = np.zeros((ni, nj, 11))
        geo[:, :, 0] = blk['xc']
        geo[:, :, 1] = blk['yc']
        geo[:, :, 2] = blk['volume']
        geo[:, :, 3:5] = blk['S1']
        geo[:, :, 5:7] = blk['S2']
        geo[:, :, 7:9] = blk['S3']
        geo[:, :, 9:11] = blk['S4']

        # 排序存储网格边界条件
        bc_sorted = [None] * 4  # 面 1 到 4
        for bc in blk['bc']:
            i1, i2, j1, j2 = bc['source']

            if i1 > i2:
                i1 = i1 - 1
            elif i1 < i2:
                i2 = i2 - 1
            elif i1 == i2 and i1 > 1:
                i1 = i1 - 1
                i2 = i2 - 1

            if j1 > j2:
                j1 = j1 - 1
            elif j1 < j2:
                j2 = j2 - 1
            elif j1 == j2 and j1 > 1:
                j1 = j1 - 1
                j2 = j2 - 1

            bc['source'] = (i1, i2, j1, j2)
            if bc['type'] == -1:
                ti1, ti2, tj1, tj2 = bc['target']

                if ti1 > ti2:
                    ti1 = ti1 - 1
                elif ti1 < ti2:
                    ti2 = ti2 - 1
                elif ti1 == ti2 and ti1 > 1:
                    ti1 = ti1 - 1
                    ti2 = ti2 - 1

                if tj1 > tj2:
                    tj1 = tj1 - 1
                elif tj1 < tj2:
                    tj2 = tj2 - 1
                elif tj1 == tj2 and tj1 > 1:
                    tj1 = tj1 - 1
                    tj2 = tj2 - 1

                bc['target'] = (ti1, ti2, tj1, tj2)

            face_id = identify_face(i1, i2, j1, j2)
            bc_sorted[face_id - 1] = bc

        # 去掉为 None 的（如果有边缺失）
        blk['bc'] = [bc for bc in bc_sorted if bc is not None]

        #合并存储
        block_cal.append({
            'geo': geo,
            'fluid': numpy_cal,
            'bc': blk['bc']
        })

    return block_cal


def trans_primitive2conservative(rho, u, v, p, w=None, gamma=1.4):
    """
    将原始变量转换为守恒变量，支持二维和三维。
    参数：
        rho : ndarray      密度 (ni, nj) 或 (ni, nj, nk)
        u, v : ndarray      速度分量
        p : ndarray         压力
        w : ndarray or None 第三维速度，若为 None 表示二维
        gamma : float       比热比

    返回：
        ndarray，守恒变量数组，形状为 (ni, nj[, nk], N_C)
    """
    # 确定维度
    ndim = 3 if w is not None else 2

    # 动能项
    kinetic = u ** 2 + v ** 2 + (w ** 2 if w is not None else 0)

    # 总能量
    E = p / (gamma - 1) + 0.5 * rho * kinetic

    # 构造守恒变量数组
    if ndim == 2:
        cons = np.stack([rho, rho * u, rho * v, E], axis=-1)  # shape: (..., 4)
    else:
        cons = np.stack([rho, rho * u, rho * v, rho * w, E], axis=-1)  # shape: (..., 5)

    return cons


def trans_conservative2primitive(U, gamma, return_pressure=False):
    """
    将守恒变量 U 转为原始变量 W: [rho, u, v, ..., H]
    支持二维或三维
    可选返回压力 p
    """
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho

    if len(U) >= 5:
        w = U[3] / rho
        kinetic = 0.5 * (u ** 2 + v ** 2 + w ** 2)
    else:
        kinetic = 0.5 * (u ** 2 + v ** 2)

    E = U[-1] / rho
    p = (gamma - 1) * (E - kinetic) * rho
    H = E + p / rho  # 总焓

    if return_pressure:
        return np.array([rho, u, v, H]), p
    else:
        return np.array([rho, u, v, H])



def trans_primitive_dl2primitive_nondl(vi, vj, p, pho, tem, e):
    """
    将无量纲原始变量还原为有量纲变量。
    参数：
        vi, vj : 无量纲速度分量
        pho    : 无量纲密度
        tem    : 无量纲温度
        e      : 无量纲总能量
        p      : 无量纲压强
    """
    return {
        "u": vi * config.V_REF,
        "v": vj * config.V_REF,
        "p": p * config.P_REF,
        "rho": pho * config.PHO_REF,
        "tem": tem * config.TEM_REF,
        "E": e * (config.V_REF * config.V_REF),
    }


def identify_face(i1, i2, j1, j2):
    if j1 == j2:
        return 1 if j1 == 1 else 3  # 下或上
    elif i1 == i2:
        return 4 if i1 == 1 else 2  # 左或右
    raise ValueError("不能识别边")
