"""
@Project ：new_fms_test
@File    ：ComSerial.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/8/6 11:18
@Des     ：
"""
import serial
from .matrix_tools import parse_press_fulll


class ComSerial():
    com_ser = None
    is_ready = False
    def __init__(self, name):
        self.com_ser = serial.Serial()  #port = name,baudrate = 460800,bytesize = 8,stopbits = 1,parity = "N"
        self.com_ser.port = name
        self.com_ser.baudrate = 460800
        self.com_ser.bytesize = 8
        self.com_ser.stopbits = 1
        self.com_ser.parity = "N"
        self.com_ser.open()

    def recv(self, size):
        if (self.com_ser.isOpen()):
            data = self.com_ser.read(size)
            # print('收取成功，{}'.format(data.hex()))
            return data.hex()

    def get_press_matrix(self):
        pre_head = '00'
        tail_head = '00'
        is_ready = False #找到帧头
        while not is_ready:
            pre_head = self.recv(1)
            if pre_head == 'a5':
                tail_head = self.recv(1)
                if tail_head == '5a':
                    is_ready = True
        other = self.recv(4)    # 获取帧长和帧类型
        data = self.recv(2048)  # 数据帧总长度为2056 = 2048 + 2 + 2 + 1 + 1 + 2
        cal_sum = self.recv(2) # 获取检验和
        press_matrix = parse_press_fulll(data)
        return press_matrix




