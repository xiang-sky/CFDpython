import numpy as np
from type_transform import identify_face


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


def boundary_farfeild(blocks):
    """
    更新远场边界条件
    参数：
        blocks (list of dict): 每个 block 是一个包含网格和边界条件的字典。
    """
    # 用远场条件初始化远场虚网格后后续迭代中不需要改变
    pass


def boundary_wall_inviscid(blocks):
    """
    无粘壁面边界条件（type=2）：速度沿法向反射，满足无穿透。
    使用 geo 中的面向量（S1~S4）计算局部法向方向。
    """
    for blk in blocks:
        fluid = blk['fluid']
        geo = blk['geo']  # (ni-1, nj-1, 7)，S1: [3:5]，S2: [5:7]，S3: [7:9]，S4: [9:11]
        ni, nj, _ = fluid.shape

        for bc in blk['bc']:
            if bc['type'] != 2:
                continue

            i1, i2, j1, j2 = bc['source']
            ghost = bc['ghost_cell']
            face_id = identify_face(i1, i2, j1, j2)

            if face_id == 1:  # 下边界 → 使用 geo[:, :, 3:5]
                for i in range(i1, i2 + 1):
                    n = geo[i, j1, 3:5]
                    n_unit = n / np.linalg.norm(n)
                    real = fluid[i, 1, :]  # j=1 是实网格第一层
                    rho, rhou, rhov, E = real[0:4]
                    u, v = rhou / rho, rhov / rho
                    un = u * n_unit[0] + v * n_unit[1]
                    u_new = u - 2 * un * n_unit[0]
                    v_new = v - 2 * un * n_unit[1]
                    ghost[0, i - i1, 0] = rho
                    ghost[0, i - i1, 1] = rho * u_new
                    ghost[0, i - i1, 2] = rho * v_new
                    ghost[0, i - i1, 3] = E

            elif face_id == 2:  # 右边界 → geo[i2-1, j, 5:7]
                for j in range(j1, j2 + 1):
                    n = geo[i2 - 1, j, 5:7]
                    n_unit = n / np.linalg.norm(n)
                    real = fluid[ni - 2, j, :]  # i=ni-2 是边界内层
                    rho, rhou, rhov, E = real[0:4]
                    u, v = rhou / rho, rhov / rho
                    un = u * n_unit[0] + v * n_unit[1]
                    u_new = u - 2 * un * n_unit[0]
                    v_new = v - 2 * un * n_unit[1]
                    ghost[0, j - j1, 0] = rho
                    ghost[0, j - j1, 1] = rho * u_new
                    ghost[0, j - j1, 2] = rho * v_new
                    ghost[0, j - j1, 3] = E

            elif face_id == 3:  # 上边界 → geo[i, j2-1, 7:9]
                for i in range(i1, i2 + 1):
                    n = geo[i, j2 - 1, 7:9]
                    n_unit = n / np.linalg.norm(n)
                    real = fluid[i, nj - 2, :]  # j=nj-2 是内层
                    rho, rhou, rhov, E = real[0:4]
                    u, v = rhou / rho, rhov / rho
                    un = u * n_unit[0] + v * n_unit[1]
                    u_new = u - 2 * un * n_unit[0]
                    v_new = v - 2 * un * n_unit[1]
                    ghost[0, i - i1, 0] = rho
                    ghost[0, i - i1, 1] = rho * u_new
                    ghost[0, i - i1, 2] = rho * v_new
                    ghost[0, i - i1, 3] = E

            elif face_id == 4:  # 左边界 → geo[i1, j, 9:11]
                for j in range(j1, j2 + 1):
                    n = geo[i1, j, 9:11]
                    n_unit = n / np.linalg.norm(n)
                    real = fluid[1, j, :]  # i=1 是内层
                    rho, rhou, rhov, E = real[0:4]
                    u, v = rhou / rho, rhov / rho
                    un = u * n_unit[0] + v * n_unit[1]
                    u_new = u - 2 * un * n_unit[0]
                    v_new = v - 2 * un * n_unit[1]
                    ghost[0, j - j1, 0] = rho
                    ghost[0, j - j1, 1] = rho * u_new
                    ghost[0, j - j1, 2] = rho * v_new
                    ghost[0, j - j1, 3] = E


def boundary_interface(blocks):
    """
    更新内边界（type=-1）的 ghost_cell，通过 transform 映射目标块实网格。
    参数：
        blocks: 所有 block 列表
    """
    for blk in blocks:
        for bc in blk['bc']:
            if bc['type'] != -1:
                continue  # 只处理内边界

            ghost = bc['ghost_cell']  # (length, ghost_layer, N_C)
            length, ghost_layer, _ = ghost.shape

            # 获取目标 block
            tgt_blk = blocks[bc['target_block']]
            tgt_fluid = tgt_blk['fluid']
            tgt_shape = tgt_fluid.shape[0:2]

            # source 和 target 的边界信息
            i1, i2, j1, j2 = bc['source']
            ti1, ti2, tj1, tj2 = bc['target']
            transform = bc['transform']

            if i1 == i2:  # 垂直边界
                for n in range(length):
                    for layer in range(ghost_layer):

                        if i1 == 1:
                            layer = -layer

                        i_delta, j_delta = apply_transform(layer, n, transform, tgt_shape)
                        i_map = ti1 + i_delta - 1
                        j_map = tj1 + j_delta - 1
                        # 复制守恒变量
                        ghost[n, layer, :] = tgt_fluid[i_map, j_map, :]
            elif j1 == j2:  # 水平边界
                for n in range(length):
                    for layer in range(ghost_layer):

                        if j1 == 1:
                            layer = -layer

                        i_delta, j_delta = apply_transform(n, layer, transform, tgt_shape)
                        i_map = ti1 + i_delta - 1
                        j_map = tj1 + j_delta - 1
                        # 复制守恒变量
                        ghost[n, layer, :] = tgt_fluid[i_map, j_map, :]


def apply_transform(i, j, transform, shape):
    """
    将(i,j)坐标按照 transform 映射，返回新的(i,j)
    transform: (a, b)
    shape: 目标块的二维 shape，用于翻转索引
    """
    a, b = transform
    ni, nj = shape
    i_new = i if abs(a) == 1 else j
    j_new = i if abs(b) == 1 else j

    if a < 0:
        i_new = ni - 1 - i_new
    if b < 0:
        j_new = nj - 1 - j_new

    return i_new, j_new
