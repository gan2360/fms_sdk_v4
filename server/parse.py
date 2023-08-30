"""
@Project ：fms_predict_sdk_v2
@File    ：parse.py
@IDE     ：PyCharm
@Author  ：Tristen
@Date    ：2023/05/05 19:20
@Des     ：解析从socket client接收到的数据
"""
import json
from utils.exceptions.parameterMissingException import ParameterMissingException


def parse_data_from_socket(receive_data: json):
    if 'code' in receive_data:
        # 获取code
        code = receive_data['code']
        return code
    else:
        raise ParameterMissingException("code in request data is missing")
