import numpy as np
from flux import conflux_roe
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
    s4 = blocks['geo'][:, :, 9:11]         # 右边法向量
    ni, nj, _ = u.shape

    res = np.zeros_like(u)

    for i in range(0, ni):
        for j in range(0, nj):
            w_stat = re(blocks, [i, j], [2, -1], m, gamma)
            flux = conflux_roe(w_stat[:, 0], w_stat[:, 1], s1, gamma)



    return res
