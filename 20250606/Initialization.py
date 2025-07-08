import config


def initialization_from_farfield(blocks):
    """
    使用远场条件进行流场和虚网格的初始化
    参数：
        blocks (list of BlockData): 每个 block 是一个 BlockData 对象。
    """
    for blk in blocks:
        # 赋值流场守恒量为远场状态
        blk.fluid[:, :, 0:4] = config.U_FAR

        # 赋值各边界的 ghost_cell 虚网格为远场状态
        for bc in blk.bc:
            bc['ghost_cell'][:, :, 0:4] = config.U_FAR

