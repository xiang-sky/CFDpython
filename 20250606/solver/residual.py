import numpy as np
from flux import conflux_ausm
from flux import reconstruct_interface_state as re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def compute_residual_roe(blocks, m=config.N_C, gamma=config.GAMMA):
    """
    计算某个块的残差矩阵 res[i,j,:] = 各方向通量之和
    """
    u = blocks['fluid']        # shape = (ni, nj, 4)
    s1 = blocks['geo'][:, :, 3:5]          # 下边法向量，shape = (ni, nj, 2)
    s2 = blocks['geo'][:, :, 5:7]
    s3 = blocks['geo'][:, :, 7:9]
    s4 = blocks['geo'][:, :, 9:11]         # 左边法向量
    vol = blocks['geo'][:, :, 2]
    ni, nj, _ = u.shape

    # 根据虚网格层数扩充守恒量矩阵U
    _, ghost_layer, _ = blocks['bc'][0]['ghost_cell'].shape
    u_rescal = np.pad(u, pad_width=((ghost_layer, ghost_layer), (ghost_layer, ghost_layer)), mode='constant', constant_values=0)

    for i in range(4):
        m_bun = blocks['bc'][i]['ghost_cell']
        m_bun_tr = np.transpose(m_bun, (1, 0, 2))
        length, ghost_layer, _ = m_bun.shape

        if i == 0:  # 下边
            u_rescal[ghost_layer:ghost_layer + length, 0:ghost_layer, :] = m_bun[:, ::-1, :]

        elif i == 1:  # 右边
            u_rescal[ni + ghost_layer:ni + 2 * ghost_layer, ghost_layer:ghost_layer + length, :] = m_bun_tr

        elif i == 2:  # 上边
            u_rescal[ghost_layer:ghost_layer + length, nj + ghost_layer:nj + 2 * ghost_layer, :] = m_bun

        elif i == 3:  # 左边
            u_rescal[0:ghost_layer, ghost_layer:ghost_layer + length, :] = m_bun_tr[::-1, :, :]

    blocks['fluid'] = u_rescal

    res = np.zeros_like(u)
    # 第一列存下边，第二列存右边，第三列存上边，第四列存左边
    flux_tem = np.zeros([ni, nj, m, 4])

    for i in range(ghost_layer, ni + ghost_layer):
        for j in range(ghost_layer, nj + ghost_layer):

            # 下边面通量
            w_stat = re(blocks, [i, j], [2, -1], m, gamma)
            flux_tem[i, j, :, 0] = conflux_ausm(w_stat[:, 0], w_stat[:, 1], s1, gamma)

            # 左边面通量
            w_stat = re(blocks, [i, j], [1, -1], m, gamma)
            flux_tem[i, j, :, 3] = conflux_ausm(w_stat[:, 0], w_stat[:, 1], s4, gamma)

            if(i == ni + ghost_layer - 1):
                # 上边面通量
                w_stat = re(blocks, [i, j], [2, 1], m, gamma)
                flux_tem[i, j, :, 2] = conflux_ausm(w_stat[:, 0], w_stat[:, 1], s3, gamma)
            if(j == nj + ghost_layer - 1):
                # 右边面通量
                w_stat = re(blocks, [i, j], [2, 1], m, gamma)
                flux_tem[i, j, :, 1] = conflux_ausm(w_stat[:, 0], w_stat[:, 1], s2, gamma)

    for i in range(ghost_layer, ni + ghost_layer - 1):
        for j in range(ghost_layer, nj + ghost_layer - 1):
            flux_tem[i, j, :, 1] = - flux_tem[i + 1, j, :, 3]
            flux_tem[i, j, :, 2] = - flux_tem[i, j + 1, :, 0]

    res = np.sum(
        flux_tem[ghost_layer: ni + ghost_layer, ghost_layer: nj + ghost_layer, :, :],
        axis=3
    )
    
    return res

