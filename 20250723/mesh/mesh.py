import numpy as np


class MeshGeoCalculator2D:
    """
        计算输入blocks的格心坐标、控制体面积、面向量
    """

    def __init__(self, mesh):
        self.mesh = mesh  # mesh 应该有 mesh.blocks

    def compute_centroids(self):
        for blk_id, blk in enumerate(self.mesh.blocks):
            x, y = blk["x"], blk["y"]
            ni, nj, nk = x.shape

            # 计算格心坐标（2D 情况）
            xc = 0.25 * (x[:-1, :-1, 0] + x[1:, :-1, 0] + x[:-1, 1:, 0] + x[1:, 1:, 0])
            yc = 0.25 * (y[:-1, :-1, 0] + y[1:, :-1, 0] + y[:-1, 1:, 0] + y[1:, 1:, 0])

            blk["xc"] = xc
            blk["yc"] = yc

            # 输出格心坐标范围
            print(f"[Block {blk_id}] xc range: ({xc.min():.8f}, {xc.max():.8f}), "
                  f"yc range: ({yc.min():.8f}, {yc.max():.8f})")

    def compute_volumes(self):
        for blk_id, blk in enumerate(self.mesh.blocks):
            x, y = blk["x"], blk["y"]

            # 取 2D 网格 (ni, nj, 1)，去掉最后一个维度
            x = x[:, :, 0]
            y = y[:, :, 0]

            # 定义四个角点
            x1 = x[:-1, :-1]  # 左下
            y1 = y[:-1, :-1]

            x2 = x[1:, :-1]  # 右下
            y2 = y[1:, :-1]

            x3 = x[1:, 1:]  # 右上
            y3 = y[1:, 1:]

            x4 = x[:-1, 1:]  # 左上
            y4 = y[:-1, 1:]

            # 按给定公式计算控制体面积（2D 体积）
            vol = 0.5 * ((x1 - x3) * (y2 - y4) + (x4 - x2) * (y1 - y3))

            blk["volume"] = vol

            print(f"[Block {blk_id}] volume range: ({vol.min():.8e}, {vol.max():.8e})")

    def compute_face_vectors(self):
        for blk_id, blk in enumerate(self.mesh.blocks):
            x, y = blk["x"][:, :, 0], blk["y"][:, :, 0]

            # 点定义（逆时针）：1=左下，2=右下，3=右上，4=左上
            x1 = x[:-1, :-1]
            y1 = y[:-1, :-1]
            x2 = x[1:, :-1]
            y2 = y[1:, :-1]
            x3 = x[1:, 1:]
            y3 = y[1:, 1:]
            x4 = x[:-1, 1:]
            y4 = y[:-1, 1:]

            # 下边面法向量 S1: 从点1→2，法向量 [y2 - y1, x1 - x2]
            S1 = np.stack((y2 - y1, x1 - x2), axis=-1)

            # 右边面法向量 S2: 从点2→3，法向量 [y3 - y2, x2 - x3]
            S2 = np.stack((y3 - y2, x2 - x3), axis=-1)

            # 上边面法向量 S3: 从点3→4，法向量 [y4 - y3, x3 - x4]
            S3 = np.stack((y4 - y3, x3 - x4), axis=-1)

            # 左边面法向量 S4: 从点4→1，法向量 [y1 - y4, x4 - x1]
            S4 = np.stack((y1 - y4, x4 - x1), axis=-1)

            blk["S1"] = S1  # (ni-1, nj-1, 2)
            blk["S2"] = S2
            blk["S3"] = S3
            blk["S4"] = S4

