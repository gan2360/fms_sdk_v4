"""
@Project ：fms_predict_sdk_v2
@File    ：tLokiSDK.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/14 12:28
@Des     ：
"""
import sys

import keyboard
import numpy as np

from client import parse
from client.baseClient import SocketClient
from client.crc import get_crc


def set_stop_flag():
    global blStop
    blStop = True


# 1. 查询当前handle数量与序列号合集
def get_handle_serial(socket_client):
    print("socket_client send to get handle serials")
    socket_client.send('AABB0100000000C1FF')


# 2. 获取标定文件序列号合集
def get_calibration_serial(socket_client):
    socket_client.send('AABB0900000000FFFF')


# 3. 设置标定文件与采集器对应,data部分长度为32,数据为标定文件(16byte)+采集器序列号(16byte)
def set_calibration_handle(socket_client, calibration_serial, handle_serial):
    send_data = 'AABB0720000000' + calibration_serial + handle_serial
    crc_code = get_crc(send_data)
    print(send_data + crc_code + 'FF')
    socket_client.send(send_data + crc_code + 'FF')


# 查询指定序列号采集器当前压力帧数据
def get_handle_data(socket_client, handle_serial):
    send_data = 'AABB0210000000' + handle_serial
    crc_code = get_crc(send_data)
    socket_client.send(send_data + crc_code + 'FF')


# 查询所有handle的当前压力帧数据
def get_all_handle_data(socket_client):
    socket_client.send('AABB030000000042FF')


if __name__ == '__main__':
    blStop = False
    socket_client = SocketClient('127.0.0.1', 20000)
    socket_client.connect()

    # 获取传感器Handle 序列号
    get_handle_serial(socket_client)

    # 会阻塞，只有获取到了传感器序列号才会进行下面的代码
    handle_dict = socket_client.receive()

    # 解析传感器序列号
    handle_serials = parse.parse_handle_serial(handle_dict["data"])
    print("传感器解析成功")

    # 获取标定文件 序列号
    get_calibration_serial(socket_client)

    # 会阻塞，只有获取到了传感器序列号才会进行下面的代码
    calibration_dict = socket_client.receive()

    # 解析标定文件序列号
    calibration_serials = parse.parse_calibration_serial(calibration_dict["data"])
    print("标定文件解析成功")

    # 进行映射
    for index in range(len(handle_serials)):
        set_calibration_handle(socket_client, calibration_serials[index], handle_serials[index])
        map_receive = socket_client.receive()
        if map_receive["data"] == '01':
            print("映射失败")
            sys.exit()
        else:
            print("序列号为" + handle_serials[index] + '的传感器映射成功')

    keyboard.add_hotkey('ctrl+shift+q', set_stop_flag)
    keyboard.add_hotkey('ctrl+shift+z', get_all_handle_data, args={socket_client})
    keyboard.add_hotkey('ctrl+shift+p', get_handle_data, args={socket_client, handle_serials[0]})
    index = 0
    while not blStop and index<20:
        index += 1
        get_handle_data(socket_client, handle_serials[1])
        # get_all_handle_data(socket_client)

        current_press_data = ""
        press_data = socket_client.receive()
        current_press_data += press_data["data"]
        while not press_data["finish"]:
            press_data = socket_client.receive()
            current_press_data += press_data["data"]
        matrix = parse.parse_press_data(current_press_data)
        # matrix = parse.parse_all_press_data(current_press_data)
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=4)  # 设精度
        np.savetxt( "./data/{}.csv".format(str(index)), matrix, delimiter="," , fmt='%.04f')

        # time.sleep(0.02)

    socket_client.close()
