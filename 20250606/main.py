import numpy as np
from mesh.mesh_read_plot3D import StructuredMeshInitialization2D
from mesh.mesh import MeshGeoCalculator2D
import boundary.boundary as bd
import type_transform as tf
import config
import Initialization as initial
from solver.solver import RK4Solver
from post_output.output_tecplot import output_tecplot
import pickle


"""
读取网格和边界条件，预处理网格
"""
mesh_read = StructuredMeshInitialization2D()
mesh_read.load_file("airfoil0012extend.grd", "airfoil0012extend.inp", 0.001)
mesh_read.merge_blocks_2D()
mesh_read.interface_transform_cal()
mesh_read.print_block_info()


"""
计算网格几何参数
"""
mesh_geocal = MeshGeoCalculator2D(mesh_read)
mesh_geocal.compute_centroids()
mesh_geocal.compute_volumes()
mesh_geocal.compute_face_vectors()


"""
添加虚网格，整理为一个计算用的列表
"""
blocks = np.copy(mesh_geocal.mesh.blocks)
bd.crate_ghost_cells(blocks, config.GHOST_LAYER, config.N_C)
for block in blocks:
    for bc in block['bc']:
        if 'ghost_cell' in bc:
            print("ghost_cell shape:", bc['ghost_cell'].shape)
for i, block in enumerate(blocks):
    print(f"Block {i} keys:", list(block.keys()))

blocks_cal = tf.trans_list2numpy_2d(blocks, config.N_C)

"""
初始化流场和边界条件
"""
initial.initialization_from_farfield(blocks_cal)


"""
迭代计算
"""
slover = RK4Solver(blocks_cal, config.GAMMA, 0.5)
slover.run(20000, 1e-3)
blocks_result = slover.blocks


"""
输出
"""
with open('blocks_result.pkl', 'wb') as f:
    pickle.dump(blocks_result, f)
output_tecplot(blocks_result)
