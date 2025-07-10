import struct
import numpy as np

class MeshReader:
    """
    MeshReader 读取Plot3D二进制网格文件。
    功能：
    - 输入网格文件名及地址
    - 得到blocks列表
    """
    def __init__(self):
        self.blocks = []
    def read_plot3d_binary(self, filename):
        with open(filename, 'rb') as f:
            # 读取 block 数量
            num_blocks = struct.unpack('<i', f.read(4))[0]

            # 读取每个 block 的尺寸 (ni, nj, nk)
            dims_list = [struct.unpack('<3i', f.read(12)) for _ in range(num_blocks)]
            self.blocks = []

            # 依次读取每个 block 的坐标数据
            for (ni, nj, nk) in dims_list:
                N = ni * nj * nk
                shape = (ni, nj, nk)

                x = np.frombuffer(f.read(8 * N), dtype='<f8').reshape(shape, order='F')
                y = np.frombuffer(f.read(8 * N), dtype='<f8').reshape(shape, order='F')
                z = np.frombuffer(f.read(8 * N), dtype='<f8').reshape(shape, order='F')

                self.blocks.append({'shape': shape, 'x': x, 'y': y, 'z': z})

    def info(self):
        for i, block in enumerate(self.blocks):
            x, y, z = block['x'], block['y'], block['z']
            print(f"[Block {i}] shape: {block['shape']}")
            print(f"  x range: {x.min():.6f} ~ {x.max():.6f}")
            print(f"  y range: {y.min():.6f} ~ {y.max():.6f}")
            print(f"  z range: {z.min():.6f} ~ {z.max():.6f}")

class BcReader:
    """
    BcReader 读取边界条件文件。
    功能：
    - 输入边界文件名及地址
    - 得到blocks_bc列表
    """
    def __init__(self):
        self.blocks_bc = []
        self.ndim = 2  # 默认二维

    def read_bc_file(self, filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        idx = 0
        flow_solver_id = int(lines[idx]); idx += 1
        num_blocks = int(lines[idx]); idx += 1

        for _ in range(num_blocks):
            # 求解维数
            dims = list(map(int, lines[idx].split())); idx += 1
            if len(dims) == 2:
                ni, nj = dims
                nk = 1
            elif len(dims) == 3:
                ni, nj, nk = dims
                self.ndim = 3
            else:
                raise ValueError(f"Invalid dimension line: {dims}")

            name = lines[idx]; idx += 1
            n_regions = int(lines[idx]); idx += 1
            regions = []

            for _ in range(n_regions):
                src_line = list(map(int, lines[idx].split())); idx += 1
                if self.ndim == 2:
                    si1, si2, sj1, sj2, bc_type = src_line
                    region = {'source': (si1, si2, sj1, sj2), 'type': bc_type}
                else:
                    si1, si2, sj1, sj2, sk1, sk2, bc_type = src_line
                    region = {'source': (si1, si2, sj1, sj2, sk1, sk2), 'type': bc_type}

                if bc_type < 0:
                    tgt_line = list(map(int, lines[idx].split())); idx += 1
                    if self.ndim == 2:
                        ti1, ti2, tj1, tj2, tgt_block = tgt_line
                        region['target'] = (ti1, ti2, tj1, tj2)
                        region['target_block'] = tgt_block - 1
                    else:
                        ti1, ti2, tj1, tj2, tk1, tk2, tgt_block = tgt_line
                        region['target'] = (ti1, ti2, tj1, tj2, tk1, tk2)
                        region['target_block'] = tgt_block - 1

                regions.append(region)

            self.blocks_bc.append({
                'name': name,
                'shape': (ni, nj, nk),
                'regions': regions
            })

    def info(self):
        print(f"Detected {'3D' if self.ndim == 3 else '2D'} boundary condition file.")
        for i, block in enumerate(self.blocks_bc):
            print(f"[Block {i}] {block['name']} shape: {block['shape']}")
            for r in block['regions']:
                print(f"  - Source: {r['source']}, Type: {r['type']}", end='')
                if r['type'] < 0:
                    print(f", Target: {r['target']}, Block: {r['target_block']}")
                else:
                    print()

class StructuredMeshInitialization2D:
    """
     读取网格和边界条件文件,将边界条件和网格列表绑定后合并网格,按照MPI并行核数进行划分
     功能：
     -mesh = StructuredMeshInitialization()
     -mesh.load_file("airfoil0012.grd", "airfoil0012.inp",0.001)
     -mesh.merge_blocks_2D()
     -mesh.interface_transform_cal()
     -mesh.print_block_info()
     """
    def __init__(self):
        self.mesh_reader = MeshReader()
        self.bc_reader = BcReader()
        self.blocks = []
        self.ndim = 2

    def scale_mesh(self, factor):
        for blk_id, blk in enumerate(self.blocks):
            blk["x"] = blk["x"] * factor
            blk["y"] = blk["y"] * factor
            blk["z"] = blk["z"] * factor

    def load_file(self, mesh_file, bc_file, factor):
        """
        读取.grd和.inp并缩放网格
        """
        # 读取.grd
        self.mesh_reader.read_plot3d_binary(mesh_file)
        self.blocks = self.mesh_reader.blocks
        # 缩放网格
        self.scale_mesh(factor)
        # 读取.inp
        self.bc_reader.read_bc_file(bc_file)
        self.ndim = self.bc_reader.ndim
        # 将边界条件合并到网格列表里面
        self.bind_bc_to_blocks()

    def merge_blocks_2D(self):
        """
        除了内边界以外的边界合并为一条线
        """
        for block in self.blocks:
            new_bc = []
            bc_by_type_and_edge = {}

            # 对所有边界按 type 和方向分组
            for bc in block.get("bc", []):
                key = (bc["type"], self.identify_face(*bc["source"]))
                bc_by_type_and_edge.setdefault(key, []).append(bc)

            for (bc_type, face_id), bcs in bc_by_type_and_edge.items():
                if len(bcs) == 1 or bc_type == -1:
                    new_bc.append(bcs[0])
                    continue

                merged = []
                current = bcs[0]["source"]

                # I 方向（下边或上边）
                if face_id == 1 or face_id == 3:
                    for i in range(len(bcs) - 1):
                        cur = bcs[i]["source"]
                        nxt = bcs[i + 1]["source"]
                        if cur[1] == nxt[0]:  # i2 == i1
                            current = (current[0], nxt[1], current[2], current[3])  # 合并段
                        elif cur[0] == nxt[1]:  # i1 == i2
                            current = (nxt[0], current[1], current[2], current[3])  # 合并段
                        else:
                            merged.append({
                                "type": bc_type,
                                "source": current
                            })
                            current = nxt
                    # 添加最后一个段
                    merged.append({
                        "type": bc_type,
                        "source": current
                    })

                # J 方向（右边或左边）
                elif face_id == 2 or face_id == 4:
                    for i in range(len(bcs) - 1):
                        cur = bcs[i]["source"]
                        nxt = bcs[i + 1]["source"]
                        if cur[3] == nxt[2]:  # j2 == j1
                            current = (current[0], current[1], current[2], nxt[3])  # 合并段
                        elif cur[2] == nxt[3]:  # j1 == j2
                            current = (current[0], current[1], nxt[2], current[3])  # 合并段
                        else:
                            merged.append({
                                "type": bc_type,
                                "source": current
                            })
                            current = nxt
                    # 添加最后一个段
                    merged.append({
                        "type": bc_type,
                        "source": current
                    })

                new_bc.extend(merged)

            block["bc"] = new_bc

    def interface_transform_cal(self):
        """
        给所有 type == -1 的边界条件计算 interface 转换信息 (a, b)，
        其中：
        a：source .I 方向对应 target 的哪个方向（1=i, 2=j），负号表示反方向；
        b：source .J 方向对应 target的哪个方向（1=i, 2=j），负号表示反方向；
        """
        for block in self.blocks:
            for bc in block.get("bc", []):
                if bc["type"] != -1:
                    continue

                s = bc["source"]
                t = bc["target"]

                i1_s, i2_s, j1_s, j2_s = s
                i1_t, i2_t, j1_t, j2_t = t

                # source vector in ij space
                ds_i = i2_s - i1_s
                ds_j = j2_s - j1_s

                # target vector in ij space
                dt_i = i2_t - i1_t
                dt_j = j2_t - j1_t

                # 构造 source 和 target 边的方向向量
                source_vec = (ds_i, ds_j)
                target_vec = (dt_i, dt_j)
                a = b = 0

                # J为内边界面
                if source_vec[0] == 0:
                    if target_vec[0] == 0:
                        if (i1_s == 1 and i1_t == 1) or (i1_s > 1 and i1_t > 1):
                            a = -1
                            b = 2 * np.sign(ds_j * dt_j)
                            b = int(b)
                        else:
                            a = 1
                            b = 2 * np.sign(ds_j * dt_j)
                            b = int(b)
                    elif target_vec[1] == 0:
                        if (i1_s == 1 and j1_t == 1) or (i1_s > 1 and j1_t > 1):
                            a = -2
                            b = np.sign(ds_j * dt_i)
                            b = int(b)
                        else:
                            a = 2
                            b = np.sign(ds_j * dt_i)
                            b = int(b)

                # I为内边界面
                if source_vec[1] == 0:
                    if target_vec[1] == 0:
                        if (j1_s == 1 and j1_t == 1) or (j1_s > 1 and j1_t > 1):
                            b = -2
                            a = np.sign(ds_i * dt_i)
                            a = int(a)
                        else:
                            b = 2
                            a = np.sign(ds_i * dt_i)
                            a = int(a)
                    elif target_vec[0] == 0:
                        if (j1_s == 1 and i1_t == 1) or (j1_s > 1 and i1_t > 1):
                            b = -1
                            a = 2 * np.sign(ds_i * dt_j)
                            a = int(a)
                        else:
                            b = 1
                            a = 2 * np.sign(ds_i * dt_j)
                            a = int(a)

                # 20250709 确认对应关系后得转换为正值
                bc["source"] = [abs(x) for x in bc["source"]]  # 20250709 确认对应关系后得转换为正值
                bc["target"] = [abs(x) for x in bc["target"]]
                bc["transform"] = (a, b)

    def bind_bc_to_blocks(self):
        """
        将边界条件（从 bc_reader 读取）绑定到每个 block 中，
        整合为 self.blocks[i]['bc']。
        """
        bc_blocks = self.bc_reader.blocks_bc
        if len(self.blocks) != len(bc_blocks):
            raise ValueError("Block 数量与边界条件数量不匹配")
        for i, bc_block in enumerate(bc_blocks):
            self.blocks[i]["bc"] = bc_block["regions"]

    def print_block_info(self):
        for i, block in enumerate(self.blocks):
            print(f"[Block {i}]  shape: {block['shape']}")
            for region in block.get("bc", []):
                info = (f"  - BC type {region['type']}, source {region['source']}, "
                        f"target_block {region.get('target_block', 'N/A')}, target {region.get('target', 'N/A')}")

                # 如果是内边界，并且包含 transform 信息，则打印 transform
                if region["type"] == -1 and "transform" in region:
                    info += f", transform {region['transform']}"

                print(info)

    def identify_face(self, i1, i2, j1, j2):
        if j1 == j2:
            return 1 if j1 == 1 else 3  # 下或上
        elif i1 == i2:
            return 4 if i1 == 1 else 2  # 左或右
        raise ValueError("不能识别边")