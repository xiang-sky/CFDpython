import numpy as np

def crate_ghost_cells(blocks, ghost_layer, N_C):
    """
    根据每个 block 的边界条件，为每条边添加 ghost_cell 数组
    参数：
        blocks (list of dict): 每个 block 是一个包含网格和边界条件的字典。
        ghost_layer (int): 每个方向添加的 ghost cell 层数。
        N_C:流场守恒量个数
    修改：
        在每个边界条件中添加键 'ghost_cell'，为一个 shape = (ghost_layer, 边长) 的空数组。
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


