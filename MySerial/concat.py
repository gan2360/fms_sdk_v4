"""
@Project ：new_fms_test
@File    ：concat.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/8/7 16:44
@Des     ：
"""
import numpy as np
from .matrix_tools import rearrange_pressure_1, rearrange_pressure_2,interpolation, normalize_matrix, interpolation


def concat_pressure(data):
    pressure1 = data['1']
    pressure2 = data['2']
    pressure3 = data['3']
    pressure4 = data['4']
    pressure1_rearrange = rearrange_pressure_1(pressure1)
    pressure2_rearrange = rearrange_pressure_1(pressure2)
    pressure3_rearrange = rearrange_pressure_2(pressure3)
    pressure4_rearrange = rearrange_pressure_2(pressure4)
    pressure12 = np.concatenate((pressure1_rearrange, pressure2_rearrange), axis=1)
    pressure34 = np.concatenate((pressure4_rearrange, pressure3_rearrange), axis=1)
    full_pressure = np.concatenate((pressure34, pressure12), axis=2)
    target_pressure = interpolation(full_pressure,(120,120))
    normalized_matrix = normalize_matrix(target_pressure)
    # print(normalized_matrix.shape)
    return normalized_matrix

