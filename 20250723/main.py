import numpy as np
from mesh.mesh_read_plot3D import StructuredMeshInitialization2D
from mesh.mesh import MeshGeoCalculator2D
import boundary.boundary as bd
import type_transform as tf
import config
import Initialization as initial
from solver.solver import CFDSolver
from post_output.output_tecplot import output_tecplot
import pickle


"""
读取网格和边界条件，预处理网格
"""
mesh_read = StructuredMeshInitialization2D()
mesh_read.load_file("RAE2822.grd", "RAE2822.inp", 1)
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
slover = CFDSolver(blocks_cal, config.GAMMA, 3)

# 时间离散格式
slover.temporal_discrete = 2

# 是否当地时间步长
slover.if_localdt = 1

# 是否生成一个沿时间序列的大矩阵
slover.if_output_npy = 0

slover.run(60000, 1e-3)
blocks_result = slover.blocks
blocks_result_seriesnpy = slover.results_series_npy

"""
输出
"""
np.save('CFDoutput.npy', blocks_result_seriesnpy)

with open('blocks_result.pkl', 'wb') as f:
    pickle.dump(blocks_result, f)
output_tecplot(blocks_result)


