import numpy as np
from numba import njit
from .flux import conflux_ausm
from .flux import reconstruct_interface_state
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def compute_residual_ausm(block, m=config.N_C, gamma=config.GAMMA):
    """
    计算某个块的残差矩阵 res[i,j,:] = 各方向通量之和
    参数:
        block: BlockData 实例，包含 geo、fluid、bc、s
        m: 守恒变量个数
        gamma: 比热比
    返回:
        res: 守恒量残差矩阵，shape = (ni, nj, m)
    """
    u = block.fluid          # shape = (ni, nj, m)
    geo = block.geo
    s = block.s              # shape = (ni, nj, 4, 2)
    ni, nj, _ = u.shape

    # 提取虚网格层数
    _, ghost_layer, _ = block.bc[0]['ghost_cell'].shape

    # 扩展 u 到含 ghost 的 u_rescal
    u_rescal = np.pad(u, pad_width=((ghost_layer, ghost_layer), (ghost_layer, ghost_layer), (0, 0)),
                      mode='constant', constant_values=0.3)

    # 写入 ghost_cell 到 u_rescal 的四个边
    for face_id, bc in enumerate(block.bc):
        ghost = bc['ghost_cell']
        length, _, _ = ghost.shape
        ghost_t = np.transpose(ghost, (1, 0, 2))  # 用于右/左边填充

        if face_id == 0:  # 下边
            u_rescal[ghost_layer:ghost_layer + length, 0:ghost_layer, :] = ghost[:, ::-1, :]
        elif face_id == 1:  # 右边
            u_rescal[ni + ghost_layer:ni + 2 * ghost_layer, ghost_layer:ghost_layer + length, :] = ghost_t
        elif face_id == 2:  # 上边
            u_rescal[ghost_layer:ghost_layer + length, nj + ghost_layer:nj + 2 * ghost_layer, :] = ghost
        elif face_id == 3:  # 左边
            u_rescal[0:ghost_layer, ghost_layer:ghost_layer + length, :] = ghost_t[::-1, :, :]

    res = compute_flux_core(u_rescal, geo, s, ni, nj, ghost_layer, m, gamma)

    return res


@njit
def compute_flux_core(u_rescal, geo, s, ni, nj, ghost_layer, m, gamma):
    flux_tem = np.zeros((ni, nj, m, 4))

    for i in range(ghost_layer, ni + ghost_layer):
        for j in range(ghost_layer, nj + ghost_layer):
            id0 = [i, j]
            idp = [i - ghost_layer, j - ghost_layer]

            # 下边
            w_stat = reconstruct_interface_state(u_rescal, geo, id0, idp, [2, -1], m, gamma)
            flux_tem[idp[0], idp[1], :, 0] = conflux_ausm(w_stat[:, 0], w_stat[:, 1], s[idp[0], idp[1], 0], gamma)

            # 左边
            w_stat = reconstruct_interface_state(u_rescal, geo, id0, idp, [1, -1], m, gamma)
            flux_tem[idp[0], idp[1], :, 3] = conflux_ausm(w_stat[:, 0], w_stat[:, 1], s[idp[0], idp[1], 3], gamma)

            # 右边（边界单元）
            if i == ni + ghost_layer - 1:
                w_stat = reconstruct_interface_state(u_rescal, geo, id0, idp, [1, 1], m, gamma)
                flux_tem[idp[0], idp[1], :, 1] = conflux_ausm(w_stat[:, 0], w_stat[:, 1], s[idp[0], idp[1], 1], gamma)

            # 上边（边界单元）
            if j == nj + ghost_layer - 1:
                w_stat = reconstruct_interface_state(u_rescal, geo, id0, idp, [2, 1], m, gamma)
                flux_tem[idp[0], idp[1], :, 2] = conflux_ausm(w_stat[:, 0], w_stat[:, 1], s[idp[0], idp[1], 2], gamma)

    # 内部边通量平衡
    for i in range(ni - 1):
        for j in range(nj - 1):
            flux_tem[i, j, :, 1] = -flux_tem[i + 1, j, :, 3]
            flux_tem[i, j, :, 2] = -flux_tem[i, j + 1, :, 0]

    for i in range(ni - 1):
        flux_tem[i, nj - 1, :, 1] = -flux_tem[i + 1, nj - 1, :, 3]
    for j in range(nj - 1):
        flux_tem[ni - 1, j, :, 2] = -flux_tem[ni - 1, j + 1, :, 0]

    return np.sum(flux_tem, axis=3)
