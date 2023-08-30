"""
@Project ：fms_predict_sdk_v2
@File    ：parameterMissingException.py
@IDE     ：PyCharm
@Author  ：Tristen
@Date    ：2023/05/05 19:20
@Des     ：参数缺失异常
"""
from utils import statusCode


class ParameterMissingException(Exception):
    """ 参数缺失 """
    def __init__(self, msg):
        self.msg = msg
        self.status_code = statusCode.PARAMETER_MISSING

    def __str__(self):
        return "[{}]: parameter is missing. {}".format(self.status_code, self.msg)