"""
@Project ：fms_predict_sdk_v2
@File    ：parse.py
@IDE     ：PyCharm
@Author  ：Tristen
@Date    ：2023/05/05 19:20
@Des     ：压力传感数据校准处理
"""
import numpy as np


def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr



def flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr


def rearrange_pressure1(pressure):
    pressure = pressure.reshape((-1, 60, 60))
    output = []
    for frame in pressure:
        frame = flip90_left(frame)
        output.append(frame)
    output = np.array(output)
    return output


def rearrange_pressure2(pressure):
    pressure = pressure.reshape((-1, 60, 60))
    output = []
    for frame in pressure:
        frame = flip90_right(frame)
        output.append(frame)
    output = np.array(output)
    return output


def align(pressure_data):
    pressure1 = pressure_data['30313030323231313239413031303031']
    pressure2 = pressure_data['30313030323231313239413031303032']
    pressure3 = pressure_data['30313030323231313239413031303033']
    pressure4 = pressure_data['30313030323231313239413031303034']
    # pressure1代表序号为1的传感器数据
    # 参数和返回值均为np.array

    pressure1_rearrange = rearrange_pressure1(pressure1)
    pressure2_rearrange = rearrange_pressure2(pressure2)
    pressure3_rearrange = rearrange_pressure1(pressure3)
    pressure4_rearrange = rearrange_pressure2(pressure4)

    pressure12 = np.concatenate((pressure2_rearrange, pressure1_rearrange), axis=2)
    pressure34 = np.concatenate((pressure4_rearrange, pressure3_rearrange), axis=2)
    pressure = np.concatenate((pressure12, pressure34), axis=1)

    pressure_normalized = []
    for frame_pressure in pressure:
        if np.max(frame_pressure) == 0:
            pressure_normalized.append(frame_pressure)
        else:
            frame_pressure = (frame_pressure - np.min(frame_pressure)) / (
                        np.max(frame_pressure) - np.min(frame_pressure))
            pressure_normalized.append(frame_pressure)
    pressure_normalized = np.array(pressure_normalized)
    return pressure_normalized
