import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from type_transform import trans_conservative2primitive
from type_transform import trans_primitive_dl2primitive_nondl


def output_tecplot(blocks, filename='solution.dat'):
    """
    将 blocks 中的 fluid 和 res 数据导出为 Tecplot 可读的 .dat 文件（结构网格）。
    每个 block 是一个 Zone。
    """
    with open(filename, 'w') as f:
        f.write("TITLE = \"CFD Solution\"\n")
        f.write("VARIABLES = \"X\", \"Y\", \"Rho\", \"U\", \"V\", \"P\",\"T\", \"Res1\", \"Res2\", \"Res3\", \"Res4\"\n")

        for idx, blk in enumerate(blocks):
            geo = blk['geo']
            fluid = blk['fluid']
            res = blk['res']

            x = geo[:, :, 0]
            y = geo[:, :, 1]
            rho = fluid[:, :, 0]
            u = fluid[:, :, 1] / rho
            v = fluid[:, :, 2] / rho
            e = fluid[:, :, 3] / rho
            p = (config.GAMMA - 1.0) * rho * (e - 0.5 * (u ** 2 + v ** 2))
            tem = p / (rho * config.R_GAS)
            u, v, p, rho, tem = trans_primitive_dl2primitive_nondl( u, v, p, rho, tem)

            ni, nj, _ = fluid.shape
            f.write(f"ZONE T=\"Block {idx}\", I={ni}, J={nj}, DATAPACKING=POINT\n")
            for j in range(nj):
                for i in range(ni):
                    vals = [x[i, j], y[i, j],
                            rho[i, j], u[i, j], v[i, j], p[i, j], tem[i, j],
                            res[i, j, 0], res[i, j, 1], res[i, j, 2], res[i, j, 3]]
                    f.write(" ".join(f"{v:.6e}" for v in vals) + "\n")
