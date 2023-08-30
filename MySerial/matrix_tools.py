"""
@Project ：new_fms_test
@File    ：matrix_tools.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/8/7 14:09
@Des     ：
"""
import struct

import numpy as np
from scipy import interpolate



def parse_press_60_32_rt(data):
    bin_str = bytes.fromhex(data)
    press_matrix = np.zeros((60, 32))
    for row in range(60):
        for col in range(32):
            hex_data = bin_str[col*64 + row+4]
            value = hex_data
            press_matrix[row][col] = value
    return press_matrix


def parse_press_fulll(data):
    bin_str = bytes.fromhex(data)
    matrix = np.zeros((64,32))
    for row in range(64):
        for col in range(32):
            byte_data = bin_str[col*64 + row:col*64 + row+1].hex()
            value = struct.unpack('<B',bytes.fromhex(byte_data))[0]
            matrix[row][col] = value
    return matrix


def interpolation(pressure_matrix, target_shape=(60,60)):
    # 原始矩阵的形状
    pressure_matrix = pressure_matrix[0]
    original_shape = pressure_matrix.shape

    # 创建插值函数
    x = np.linspace(0, original_shape[1] - 1, original_shape[1])
    y = np.linspace(0, original_shape[0] - 1, original_shape[0])
    f = interpolate.interp2d(x, y, pressure_matrix, kind='linear')

    # 在目标形状上进行插值
    new_x = np.linspace(0, original_shape[1] - 1, target_shape[1])
    new_y = np.linspace(0, original_shape[0] - 1, target_shape[0])
    interpolated_matrix = f(new_x, new_y)
    interpolated_matrix = interpolated_matrix.reshape(-1, *target_shape)
    return interpolated_matrix


def normalize_matrix(matrix):
    # 计算矩阵的最大值和最小值
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    if min_value == max_value:
        normalized_matrix = np.zeros_like(matrix)
    # 归一化矩阵
    else:
        normalized_matrix = (matrix - min_value) / (max_value - min_value)
    # 限制小数位数不超过5位
    rounded_matrix = np.round(normalized_matrix, decimals=5)

    return rounded_matrix


def flip180_right(arr):
    # 逆序操作
    arr = rotate_matrix_90(arr)
    arr = rotate_matrix_90(arr)

    # 转置操作
    return arr

def rotate_matrix_90(matrix):
    # 转置操作
    transposed_matrix = np.transpose(matrix)

    # 逆序操作
    rotated_matrix = np.flip(transposed_matrix, axis=1)

    return rotated_matrix


def rearrange_pressure_2(pressure):
    pressure = pressure.reshape((-1, 64, 32))
    output = []
    for frame in pressure:
        frame = flip180_right(frame)
        output.append(frame)
    output = np.array(output)
    return output


def rearrange_pressure_1(pressure):
    pressure = pressure.reshape((-1, 64, 32))
    return pressure


