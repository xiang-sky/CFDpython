import numpy as np
from type_transform import identify_face
import config

def crate_ghost_cells(blocks, ghost_layer, N_C):
    """
    根据每个 block 的边界条件，为每条边添加 ghost_cell 数组
    参数：
        blocks (list of dict): 每个 block 是一个包含网格和边界条件的字典。
        ghost_layer (int): 每个方向添加的 ghost cell 层数。
        N_C:流场守恒量个数
    修改：
        在每个边界条件中添加键 'ghost_cell'，为一个 shape = (边长, ghost_layer, N_C) 的空数组。
    """
    for block in blocks:
        if 'bc' not in block:
            raise ValueError("block列表中没边界条件")

        for bc in block['bc']:
            i1, i2, j1, j2 = bc['source']
            if i1 == i2:
                # 垂直边界，变化的是 j，边长为 j2 - j1
                length = abs(j2 - j1)
                shape = (length, ghost_layer, N_C)
            elif j1 == j2:
                # 水平边界，变化的是 i，边长为 i2 - i1
                length = abs(i2 - i1)
                shape = (length, ghost_layer, N_C)
            else:
                raise ValueError(f"无法识别边界: source={bc['source']}")

            # 初始化 ghost_cell 为 float 类型的空数组
            bc['ghost_cell'] = np.zeros(shape, dtype=float)


def boundary_farfeild(blocks, alpha=0):
    """
    使用 config.U_FAR 更新远场边界条件 ghost cell：
    ghost = α * 内部实网格 + (1 - α) * config.U_FAR

    参数：
        blocks: 所有结构块列表
        alpha: 权重系数，默认 0.9（越大表示越靠近内部真实值）
    """
    for blk in blocks:
        fluid = blk.fluid
        ni, nj, _ = fluid.shape

        for bc in blk.bc:
            if bc['type'] != 4:
                continue  # 非远场边界跳过

            i1, i2, j1, j2 = bc['source']
            ghost = bc['ghost_cell']
            length, ghost_layer, _ = ghost.shape
            face_id = identify_face(i1, i2, j1, j2)

            for n in range(length):
                for layer in range(ghost_layer):
                    if face_id == 1:  # 下边界
                        U_inside = fluid[n, layer, :]
                    elif face_id == 2:  # 右边界
                        U_inside = fluid[ni - 1 - layer, n, :]
                    elif face_id == 3:  # 上边界
                        U_inside = fluid[n, nj - 1 - layer, :]
                    elif face_id == 4:  # 左边界
                        U_inside = fluid[layer, n, :]

                    ghost[n, layer, :] = alpha * U_inside + (1 - alpha) * config.U_FAR


def boundary_wall_inviscid(blocks):
    """
    无粘壁面边界条件（type=2）：速度沿法向反射，满足无穿透。
    使用 BlockData.geo 中的面向量计算局部法向方向。
    """
    for blk in blocks:
        fluid = blk.fluid
        geo = blk.geo
        s = blk.s  # (ni, nj, 4, 2)，四个方向的法向量
        ni, nj, _ = fluid.shape

        for bc in blk.bc:
            if bc['type'] != 2:
                continue

            i1, i2, j1, j2 = bc['source']
            ghost = bc['ghost_cell']
            length, ghost_layer, _ = ghost.shape
            face_id = identify_face(i1, i2, j1, j2)

            if face_id == 1:  # 下边界，对应 s[..., 0]
                for n in range(length):
                    for layer in range(ghost_layer):
                        vector_n = s[n, j1 - 1, 0, :]
                        n_unit = vector_n / np.linalg.norm(vector_n)
                        rho, rhou, rhov, E = fluid[n, layer, :]
                        u = rhou / rho
                        v = rhov / rho
                        un = u * n_unit[0] + v * n_unit[1]
                        u_new = u - 2 * un * n_unit[0]
                        v_new = v - 2 * un * n_unit[1]
                        ghost[n, layer, 0] = rho
                        ghost[n, layer, 1] = rho * u_new
                        ghost[n, layer, 2] = rho * v_new
                        ghost[n, layer, 3] = E

            elif face_id == 2:  # 右边界，对应 s[..., 1]
                for n in range(length):
                    for layer in range(ghost_layer):
                        vector_n = s[i2 - 1, n, 1, :]
                        n_unit = vector_n / np.linalg.norm(vector_n)
                        rho, rhou, rhov, E = fluid[ni - 1 - layer, n, :]
                        u = rhou / rho
                        v = rhov / rho
                        un = u * n_unit[0] + v * n_unit[1]
                        u_new = u - 2 * un * n_unit[0]
                        v_new = v - 2 * un * n_unit[1]
                        ghost[n, layer, 0] = rho
                        ghost[n, layer, 1] = rho * u_new
                        ghost[n, layer, 2] = rho * v_new
                        ghost[n, layer, 3] = E

            elif face_id == 3:  # 上边界，对应 s[..., 2]
                for n in range(length):
                    for layer in range(ghost_layer):
                        vector_n = s[n, j2 - 1, 2, :]
                        n_unit = vector_n / np.linalg.norm(vector_n)
                        rho, rhou, rhov, E = fluid[n, nj - 1 - layer, :]
                        u = rhou / rho
                        v = rhov / rho
                        un = u * n_unit[0] + v * n_unit[1]
                        u_new = u - 2 * un * n_unit[0]
                        v_new = v - 2 * un * n_unit[1]
                        ghost[n, layer, 0] = rho
                        ghost[n, layer, 1] = rho * u_new
                        ghost[n, layer, 2] = rho * v_new
                        ghost[n, layer, 3] = E

            elif face_id == 4:  # 左边界，对应 s[..., 3]
                for n in range(length):
                    for layer in range(ghost_layer):
                        vector_n = s[i1 - 1, n, 3, :]
                        n_unit = vector_n / np.linalg.norm(vector_n)
                        rho, rhou, rhov, E = fluid[layer, n, :]
                        u = rhou / rho
                        v = rhov / rho
                        un = u * n_unit[0] + v * n_unit[1]
                        u_new = u - 2 * un * n_unit[0]
                        v_new = v - 2 * un * n_unit[1]
                        ghost[n, layer, 0] = rho
                        ghost[n, layer, 1] = rho * u_new
                        ghost[n, layer, 2] = rho * v_new
                        ghost[n, layer, 3] = E



def boundary_interface(blocks):
    """
    更新所有 block 的 type=-1 内边界 ghost_cell，通过 transform 映射目标块实网格。
    参数:
        blocks: BlockData 的列表
    """
    for blk in blocks:
        for bc in blk.bc:
            if bc['type'] != -1:
                continue  # 跳过非内边界

            ghost = bc['ghost_cell']  # shape = (length, ghost_layer, N_C)
            length, ghost_layer, _ = ghost.shape

            # 获取目标 block 和流场数据
            tgt_blk = blocks[bc['target_block']]
            tgt_fluid = tgt_blk.fluid
            tgt_shape = tgt_fluid.shape[:2]

            # source 和 target 的边界信息
            i1, i2, j1, j2 = bc['source']
            ti1, ti2, tj1, tj2 = bc['target']
            transform = bc['transform']

            if i1 > i2:
                i1, i2 = i2, i1
                ti1, ti2 = ti2, ti1
            if j1 > j2:
                j1, j2 = j2, j1
                tj1, tj2 = tj2, tj1

            if i1 == i2:  # 垂直边界（沿 j 方向变）
                for n in range(length):
                    for layer in range(ghost_layer):

                        dr_i = - 1 if i1 == 1 else 1
                        transform_matrix = transform_matrix_cal(transform)
                        i_j_p = np.array([[i1 + dr_i * layer], [j1 + n]])
                        i_j_b1 = np.array([[i1], [j1]])
                        i_j_b2 = np.array([[ti1], [tj1]])
                        i_j_map = transform_matrix @ (i_j_p - i_j_b1) + i_j_b2
                        ghost[n, layer, :] = tgt_fluid[int(i_j_map[0]) - 1, int(i_j_map[1]) - 1, :]

            elif j1 == j2:  # 水平边界（沿 i 方向变）
                for n in range(length):
                    for layer in range(ghost_layer):

                        dr_j = -1 if j1 == 1 else 1
                        transform_matrix = transform_matrix_cal(transform)
                        i_j_p = np.array([[i1 + n], [j1 + dr_j * layer]])
                        i_j_b1 = np.array([[i1], [j1]])
                        i_j_b2 = np.array([[ti1], [tj1]])
                        i_j_map = transform_matrix @ (i_j_p - i_j_b1) + i_j_b2
                        ghost[n, layer, :] = tgt_fluid[int(i_j_map[0]) - 1, int(i_j_map[1]) - 1, :]


def transform_matrix_cal(transform):
    """
    将(i,j)坐标按照 transform 映射，返回新的(i,j)
    transform: (a, b)
    shape: 目标块的二维 shape，用于翻转索引
    """
    a, b = transform
    dela1 = 1 if abs(a) == 1 else 0
    dela2 = 1 if abs(a) == 2 else 0
    delb1 = 1 if abs(b) == 1 else 0
    delb2 = 1 if abs(b) == 2 else 0

    t = np.array([
        [np.sign(a) * dela1, np.sign(b) * delb1],
        [np.sign(a) * dela2, np.sign(b) * delb2]
    ])

    return t
