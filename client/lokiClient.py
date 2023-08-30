"""
@Project ：fms_predict_sdk
@File    ：lokiClient.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/6 10:34
@Des     ：
"""
from client.parse import *
from client.baseClient import SocketClient
from client.crc import get_crc


class LokiSDKClient:
    def __init__(self):
        # 连接SDK传感器数据
        self.socket_client = SocketClient('127.0.0.1', 20000)
        self.socket_client.connect()

        # 获取传感器Handle 序列号
        self.get_handle_serial()

        # 会阻塞，只有获取到了传感器序列号才会进行下面的代码
        handle_dict = self.socket_client.receive()

        # 解析传感器序列号
        handle_serials = parse_handle_serial(handle_dict["data"])
        print("传感器解析成功")

        # 获取标定文件 序列号
        self.get_calibration_serial()

        # 会阻塞，只有获取到了传感器序列号才会进行下面的代码
        calibration_dict = self.socket_client.receive()

        # 解析标定文件序列号
        calibration_serials = parse_calibration_serial(calibration_dict["data"])
        print("标定文件解析成功")

        # 进行映射
        for index in range(len(handle_serials)):
            self.set_calibration_handle(calibration_serials[index], handle_serials[index])
            map_receive = self.socket_client.receive()
            if map_receive["data"] == '01':
                print("映射失败")
            else:
                print("序列号为" + handle_serials[index] + '的传感器映射成功')



    def get_pressure_data(self):
        self.get_all_handle_data()
        current_press_data = ""
        press_data = self.socket_client.receive()
        current_press_data += press_data["data"]
        while not press_data["finish"]:
            press_data = self.socket_client.receive()
            current_press_data += press_data["data"]
        matrix = parse_all_press_data(current_press_data)
        return matrix

    def close(self):
        self.socket_client.close()

    # 1. 查询当前handle数量与序列号合集
    def get_handle_serial(self):
        print("socket_client send to get handle serials")
        self.socket_client.send('AABB0100000000C1FF')

    # 2. 获取标定文件序列号合集
    def get_calibration_serial(self):
        self.socket_client.send('AABB0900000000FFFF')

    # 3. 设置标定文件与采集器对应,data部分长度为32,数据为标定文件(16byte)+采集器序列号(16byte)
    def set_calibration_handle(self, calibration_serial, handle_serial):
        send_data = 'AABB0720000000' + calibration_serial + handle_serial
        crc_code = get_crc(send_data)
        print(send_data + crc_code + 'FF')
        self.socket_client.send(send_data + crc_code + 'FF')

    # 查询指定序列号采集器当前压力帧数据
    def get_handle_data(self, handle_serial):
        send_data = 'AABB0210000000' + handle_serial
        crc_code = get_crc(send_data)
        self.socket_client.send(send_data + crc_code + 'FF')

    # 查询所有handle的当前压力帧数据
    def get_all_handle_data(self):
        self.socket_client.send('AABB030000000042FF')
