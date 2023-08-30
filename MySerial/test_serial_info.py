"""
@Project ：new_fms_test
@File    ：test_serial_info.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/8/7 21:04
@Des     ：
"""
from serial.tools import list_ports


def test_serial_ports():
    port_lists = list(list_ports.comports())
    ports_name = [port.name for port in port_lists if 'USB-SERIAL' in port.description]
    if len(ports_name) <= 0:
        raise Exception('无相关串口信息')
    print(ports_name)

test_serial_ports()