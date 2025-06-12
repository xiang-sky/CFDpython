import config


def intialization_from_farfield(blocks):
    """
    使用远场条件进行流场和虚网格的初始化
    参数：
        blocks (list of dict): 每个 block 是一个包含网格和边界条件的字典。
    """
    for blk in blocks:
        blk['fluid'][:, :, 0:4] = config.U_FAR
        for bc in blk['bc']:
            bc['ghost_cell'][:, :, 0:4] = config.U_FAR

