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


def output_forces(blocks):
    """
    插值得到 blocks 中壁面边上的压力值并积分求 Fx 和 Fy
    """
    fx = 0.0
    fy = 0.0

    for blk in blocks:
        fluid = blk['fluid']
        geo = blk['geo']  # 用当前块的 geo

        for bc in blk['bc']:
            if bc['type'] != 2:
                continue  # 只处理壁面边界

            i1, i2, j1, j2 = bc['source']
            face_id = identify_face(i1, i2, j1, j2)

            # 下边
            if face_id == 1:
                i_start = min(i1, i2)
                i_end = max(i1, i2)

                # 方向向量（面法向量），(ni, 2)
                s_vec = geo[i_start - 1:i_end, j1 - 1, 3:5]  # Sx, Sy
                # length = np.linalg.norm(s_vec, axis=-1)  # 每段边长    面矢量本身隐含面积/边长故注释

                # 输出压力
                fluid_slice = fluid[i_start - 1:i_end, j1 - 1:j1 + 1, :]  # 取两行用于壁面插值
                prim = trans_numpy_conservative2primitive(fluid_slice)  # (ni, 2, 4) -> primitive

                p2 = prim[:, 0, 3]  # 壁面内一层
                p3 = prim[:, 1, 3]  # 更里一层
                p_wall = 0.5 * (3 * p2 - p3)  # 二阶外推

                # 无量纲转换为有量纲
                p_wall = p_wall * config.P_REF

                fx += np.sum(p_wall * s_vec[:, 0])
                fy += np.sum(p_wall * s_vec[:, 1])

            # 上边
            if face_id == 3:
                i_start = min(i1, i2)
                i_end = max(i1, i2)

                # 方向向量（面法向量），(ni, 2)
                s_vec = geo[i_start - 1:i_end, j1 - 1, 7:9]  # Sx, Sy

                # 输出压力
                fluid_slice = fluid[i_start - 1:i_end, j1 - 2:j1, :]  # 取两行用于壁面插值
                prim = trans_numpy_conservative2primitive(fluid_slice)  # (ni, 2, 4) -> primitive

                p2 = prim[:, 1, 3]  # 壁面内一层
                p3 = prim[:, 0, 3]  # 更里一层
                p_wall = 0.5 * (3 * p2 - p3)  # 二阶外推

                # 无量纲转换为有量纲
                p_wall = p_wall * config.P_REF

                fx += np.sum(p_wall * s_vec[:, 0])
                fy += np.sum(p_wall * s_vec[:, 1])

            # 右边
            if face_id == 2:
                j_start = min(j1, j2)
                j_end = max(j1, j2)

                # 方向向量（面法向量），(ni, 2)
                s_vec = geo[i1 - 1, j_start - 1:j_end, 5:7]  # Sx, Sy

                # 输出压力
                fluid_slice = fluid[i1 - 2:i1, j_start - 1:j_end, :]  # 取两行用于壁面插值
                prim = trans_numpy_conservative2primitive(fluid_slice)  # (ni, 2, 4) -> primitive

                p2 = prim[1, :, 3]  # 壁面内一层
                p3 = prim[0, :, 3]  # 更里一层
                p_wall = 0.5 * (3 * p2 - p3)  # 二阶外推

                # 无量纲转换为有量纲
                p_wall = p_wall * config.P_REF

                fx += np.sum(p_wall * s_vec[:, 0])
                fy += np.sum(p_wall * s_vec[:, 1])

            # 左边
            if face_id == 4:
                j_start = min(j1, j2)
                j_end = max(j1, j2)

                # 方向向量（面法向量），(ni, 2)
                s_vec = geo[i1 - 1, j_start - 1:j_end, 9:11]  # Sx, Sy

                # 输出压力
                fluid_slice = fluid[i1 - 1:i1 + 1, j_start - 1:j_end, :]  # 取两行用于壁面插值
                prim = trans_numpy_conservative2primitive(fluid_slice)  # (ni, 2, 4) -> primitive

                p2 = prim[0, :, 3]  # 壁面内一层
                p3 = prim[1, :, 3]  # 更里一层
                p_wall = 0.5 * (3 * p2 - p3)  # 二阶外推

                # 无量纲转换为有量纲
                p_wall = p_wall * config.P_REF

                fx += np.sum(p_wall * s_vec[:, 0])
                fy += np.sum(p_wall * s_vec[:, 1])

    return fx, fy

