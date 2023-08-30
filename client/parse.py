"""
@Project ：fms_predict_sdk_v2
@File    ：parse.py
@IDE     ：PyCharm
@Author  ：Tristen
@Date    ：2023/05/05 19:20
@Des     ：解析传感器SDK相关的接收数据
"""
import numpy as np
import struct


def parse_handle_serial(data):
    count = int(data[0:2])
    handle_serials = []
    for index in range(count):
        handle_serials.append(data[2 + index * 32:2 + index * 32 + 32])
    handle_serials.sort()
    return handle_serials


def parse_calibration_serial(data):
    count = int(len(data) / 32)
    calibration_serials = []
    for index in range(count):
        calibration_serials.append(data[index * 32:index * 32 + 32])
    calibration_serials.sort()
    return calibration_serials


def parse_all_press_data(data):
    count = int(len(data) / 51232)
    handle_press = {}
    for index in range(count):
        temp_data = data[index * 51232: index * 51232 + 51232]
        handle_serial = temp_data[0:32]
        press_data = temp_data[32:]
        handle_press[handle_serial] = parse_press_data(press_data)
    return handle_press


def parse_press_data(data):
    data = bytes.fromhex(data)
    data_matrix = zip(*(iter(data),) * 320)
    data_matrix = [list(i) for i in data_matrix]
    press_matrix = np.zeros((60, 60))
    for i in range(20, 80):
        col = 0
        for j in range(80, 320, 4):
            point_bytes = '{:02x}'.format(data_matrix[i][j]) + '{:02x}'.format(data_matrix[i][j + 1]) + '{:02x}'.format(data_matrix[i][j + 2]) + '{:02x}'.format(data_matrix[i][j + 3])
            [point] = struct.unpack('<f', bytes.fromhex(point_bytes))
            press_matrix[i-20][col] = point / 4
            col += 1
    return press_matrix
