"""
@Project ：fms_predict_sdk_v2
@File    ：dataDecodeException.py
@IDE     ：PyCharm
@Author  ：Tristen
@Date    ：2023/05/05 19:20
@Des     ：接收数据不是json，解析异常
"""
from utils import statusCode


class DataDecodeException(Exception):
    """ 接收数据不是json，解析异常 """
    def __init__(self, msg):
        self.msg = msg
        self.status_code = statusCode.DATA_DECODE_ERROR

    def __str__(self):
        return "[{}]: data decode error. {}".format(self.status_code, self.msg)
