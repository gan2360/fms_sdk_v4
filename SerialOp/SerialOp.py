"""
@Project ：new_fms_test
@File    ：SerialOp.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/8/7 16:59
@Des     ：
"""

from MySerial.ComSerial import ComSerial
from serial.tools import list_ports
class SerialeOp:
    ser_list = []
    def __init__(self):
        port_list = list(list_ports.comports())
        port_names = [port.name for port in port_list if 'USB-SERIAL' in port.description]
        if len(port_names) <= 0:
            raise Exception("USB串口未连接")
        for name in port_names:
            ser = ComSerial(name)
            self.ser_list.append(ser)
    def get_all_pressure(self):
        data = {}
        for i in range(len(self.ser_list)):
            data[f'{i+1}'] = self.ser_list[i].get_press_matrix()
        return data
