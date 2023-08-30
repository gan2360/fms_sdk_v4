"""
@Project ：fms_predict_sdk_v2
@File    ：crc.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/6 10:29
@Des     ：计算crc冗余码
"""
import crcmod.predefined


def get_crc(send_data):
    crc8 = crcmod.predefined.Crc('crc-8-maxim')
    crc8.update(bytes().fromhex(send_data))
    return hex(crc8.crcValue)[2:]
