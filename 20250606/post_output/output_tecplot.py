import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from type_transform import trans_numpy_conservative2primitive
from type_transform import trans_primitive_dl2primitive_nondl
from type_transform import identify_face


def output_tecplot(blocks, filename='solution.dat'):
    """
    将 blocks 中的 fluid 和 res 数据导出为 Tecplot 可读的 .dat 文件（结构网格）。
    每个 block 是一个 Zone。
    参数:
        blocks: list[BlockData]
        filename: 输出文件名
    """
    with open(filename, 'w') as f:
        f.write("TITLE = \"CFD Solution\"\n")
        f.write("VARIABLES = \"X\", \"Y\", \"Rho\", \"U\", \"V\", \"P\", \"T\", "
                "\"Res1\", \"Res2\", \"Res3\", \"Res4\"\n")

        for idx, blk in enumerate(blocks):
            geo = blk.geo
            fluid = blk.fluid
            res = blk.res  # 这里假设 blk.res 是每个 block 外部赋值的属性

            x = geo[:, :, 0]
            y = geo[:, :, 1]
            rho = fluid[:, :, 0]
            u = fluid[:, :, 1] / rho
            v = fluid[:, :, 2] / rho
            e = fluid[:, :, 3] / rho
            p = (config.GAMMA - 1.0) * rho * (e - 0.5 * (u ** 2 + v ** 2))
            tem = p / (rho * config.R_GAS)

            # 非量纲化转换（如有需要）
            u, v, p, rho, tem = trans_primitive_dl2primitive_nondl(u, v, p, rho, tem)

            ni, nj, _ = fluid.shape
            f.write(f"ZONE T=\"Block {idx}\", I={ni}, J={nj}, DATAPACKING=POINT\n")
            for j in range(nj):
                for i in range(ni):
                    vals = [
                        x[i, j], y[i, j],
                        rho[i, j], u[i, j], v[i, j], p[i, j], tem[i, j],
                        res[i, j, 0], res[i, j, 1], res[i, j, 2], res[i, j, 3]
                    ]
                    f.write(" ".join(f"{v:.6e}" for v in vals) + "\n")


def output_forces(blocks):
    """
    插值得到 blocks 中壁面边上的压力值并积分求 Fx 和 Fy。
    参数:
        blocks: List[BlockData]
    返回:
        fx, fy: 总体气动力（无量纲）
    """
    fx = 0.0
    fy = 0.0

    for blk in blocks:
        fluid = blk.fluid
        geo = blk.geo

        for bc in blk.bc:
            if bc['type'] != 2:
                continue  # 只处理壁面边界

            i1, i2, j1, j2 = bc['source']
            face_id = identify_face(i1, i2, j1, j2)

            # 下边界（S1）
            if face_id == 1:
                i_start = min(i1, i2)
                i_end = max(i1, i2)
                s_vec = geo[i_start - 1:i_end, j1 - 1, 3:5]
                fluid_slice = fluid[i_start - 1:i_end, j1 - 1:j1 + 1, :]
                prim = trans_numpy_conservative2primitive(fluid_slice)
                p2 = prim[:, 0, 3]
                p3 = prim[:, 1, 3]
                p_wall = 0.5 * (3 * p2 - p3)
                p_wall *= config.P_REF
                fx += np.sum(p_wall * s_vec[:, 0])
                fy += np.sum(p_wall * s_vec[:, 1])

            # 上边界（S3）
            elif face_id == 3:
                i_start = min(i1, i2)
                i_end = max(i1, i2)
                s_vec = geo[i_start - 1:i_end, j1 - 1, 7:9]
                fluid_slice = fluid[i_start - 1:i_end, j1 - 2:j1, :]
                prim = trans_numpy_conservative2primitive(fluid_slice)
                p2 = prim[:, 1, 3]
                p3 = prim[:, 0, 3]
                p_wall = 0.5 * (3 * p2 - p3)
                p_wall *= config.P_REF
                fx += np.sum(p_wall * s_vec[:, 0])
                fy += np.sum(p_wall * s_vec[:, 1])

            # 右边界（S2）
            elif face_id == 2:
                j_start = min(j1, j2)
                j_end = max(j1, j2)
                s_vec = geo[i1 - 1, j_start - 1:j_end, 5:7]
                fluid_slice = fluid[i1 - 2:i1, j_start - 1:j_end, :]
                prim = trans_numpy_conservative2primitive(fluid_slice)
                p2 = prim[1, :, 3]
                p3 = prim[0, :, 3]
                p_wall = 0.5 * (3 * p2 - p3)
                p_wall *= config.P_REF
                fx += np.sum(p_wall * s_vec[:, 0])
                fy += np.sum(p_wall * s_vec[:, 1])

            # 左边界（S4）
            elif face_id == 4:
                j_start = min(j1, j2)
                j_end = max(j1, j2)
                s_vec = geo[i1 - 1, j_start - 1:j_end, 9:11]
                fluid_slice = fluid[i1 - 1:i1 + 1, j_start - 1:j_end, :]
                prim = trans_numpy_conservative2primitive(fluid_slice)
                p2 = prim[0, :, 3]
                p3 = prim[1, :, 3]
                p_wall = 0.5 * (3 * p2 - p3)
                p_wall *= config.P_REF
                fx += np.sum(p_wall * s_vec[:, 0])
                fy += np.sum(p_wall * s_vec[:, 1])

    return fx, fy


def output_tecplot_series(blocks, iteration, filename=None):
    """
    将当前 blocks 写入一个 Tecplot .dat 文件，每个 ZONE 带有 SOLUTIONTIME=iteration。
    文件名默认为 solution_iter_{iteration}.dat
    """
    if filename is None:
        filename = f"solution_iter_{iteration:04d}.dat"

    with open(filename, 'w') as f:
        f.write("TITLE = \"CFD Solution\"\n")
        f.write("VARIABLES = \"X\", \"Y\", \"Rho\", \"U\", \"V\", \"P\", \"T\", \"Res1\", \"Res2\", \"Res3\", \"Res4\"\n")

        for idx, blk in enumerate(blocks):
            geo = blk.geo
            fluid = blk.fluid
            res = blk.res

            x = geo[:, :, 0]
            y = geo[:, :, 1]
            rho = fluid[:, :, 0]
            u = fluid[:, :, 1] / rho
            v = fluid[:, :, 2] / rho
            e = fluid[:, :, 3] / rho
            p = (config.GAMMA - 1.0) * rho * (e - 0.5 * (u ** 2 + v ** 2))
            tem = p / (rho * config.R_GAS)
            u, v, p, rho, tem = trans_primitive_dl2primitive_nondl(u, v, p, rho, tem)

            ni, nj, _ = fluid.shape
            f.write(f"ZONE T=\"Block {idx}\", I={ni}, J={nj}, DATAPACKING=POINT, SOLUTIONTIME={iteration}\n")

            for j in range(nj):
                for i in range(ni):
                    vals = [x[i, j], y[i, j],
                            rho[i, j], u[i, j], v[i, j], p[i, j], tem[i, j],
                            res[i, j, 0], res[i, j, 1], res[i, j, 2], res[i, j, 3]]
                    f.write(" ".join(f"{v:.6e}" for v in vals) + "\n")

