"""
@Project ：fms_predict_sdk_v2
@File    ：server.py
@IDE     ：PyCharm
@Author  ：Tristen
@Date    ：2023/05/05 19:20
@Des     ：作为传输的服务，一个Socket实例
"""
import base64
import datetime
import hashlib
import json
import logging
import socket
import websockets
import struct

import cv2
import numpy as np

from rawModel.lib.hrnet.hrnetModel import HRnetModelPrediction
from rawModel.lib.yolov3.yoloModel import YoloModelPrediction
from rawModel.predictor import getPose3dRawModel
from rawModel.rawModel import RawModelPrediction
from utils import statusCode
from utils.exceptions.dataDecodeException import DataDecodeException


class SocketStatus:
    READY = 0
    RUNNING = 1
    CLOSED = 2


class FMSMovement:
    DEEP_SQUAT = 1
    HURDLE_STEP_LEFT = 2
    HURDLE_STEP_RIGHT = 2
    INLINE_LUNGE_LEFT = 3
    INLINE_LUNGE_RIGHT = 4
    SHOULDER_MOBILITY_LEFT = 5
    SHOULDER_MOBILITY_RIGHT = 6
    ACTIVE_STRAIGHT_LEG_RAISE_LEFT = 7
    ACTIVE_STRAIGHT_LEG_RAISE_RIGHT = 8
    TRUNK_STABILITY_PUSH_UP = 9
    ROTARY_STABILITY_QUADRUPED_LEFT = 10
    ROTARY_STABILITY_QUADRUPED_RIGHT = 11


header_size = struct.calcsize("!I")


class SocketServer:
    """
    用于向Unity传送预测完成后的数据
    """

    def __init__(self, host="127.0.0.1", port=5001):

        self.status = SocketStatus.READY
        # Socket端口开在5001端口上面
        self.address = (host, port)
        # 创建客户端套接字
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #############################
        self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #############################
        # 设置成非阻塞
        self.socket_server.bind(self.address)
        self.socket_server.listen(5)
        logging.info("========================= WAITING FOR A CONNECTION =========================")
        print("========================= 等待一个连接 =========================")
        self.conn, address = self.socket_server.accept()
        self.status = SocketStatus.RUNNING
        self.data = b''
        self.is_loaded = False
        self.start_predict = False
        self.onnx = None
        self.raw_model_prediction = None

        # 当前正在预测的动作
        self.current_prediction_movement = None
        # 当前正在预测的动作的预测次数
        self.current_prediction_times = 0
        data = self.conn.recv(1024)  # 获取客户端发送的消息
        # 想将http协议的数据处理成字典的形式方便后续取值
        header_dict = self.get_headers(data)  # 将一大堆请求头转换成字典数据  类似于wsgiref模块
        client_random_string = header_dict['Sec-WebSocket-Key']  # 获取浏览器发送过来的随机字符串
        # magic string拼接
        magic_string = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'  # 全球共用的随机字符串 一个都不能写错
        # 确认握手Sec-WebSocket-Key固定格式：headers头部的Sec-WebSocket-Key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        # 确认握手的秘钥值为 传入的秘钥+magic_string，使用sha1算法加密，然后base64转码
        value = client_random_string + magic_string  # 拼接

        # 算法加密 对请求头中的sec-websocket-key进行加密
        ac = base64.b64encode(hashlib.sha1(value.encode('utf-8')).digest())  # 加密处理

        # 将处理好的结果再发送给客户端校验
        tpl = "HTTP/1.1 101 Switching Protocols\r\n" \
              "Upgrade:websocket\r\n" \
              "Connection: Upgrade\r\n" \
              "Sec-WebSocket-Accept: %s\r\n" \
              "WebSocket-Location: ws://127.0.0.1:8080\r\n\r\n"
        response_str = tpl % ac.decode('utf-8')  # 处理到响应头中

        # 将随机字符串给浏览器返回回去
        print(f"建立连接,加密验证key{ac}")
        logging.info("========================= SOCKET CONNECTION IS CREATED =========================")
        print("========================= 连接已经建立 =========================")
        print((bytes(response_str, encoding='utf-8')))
        self.conn.send(bytes(response_str, encoding='utf-8'))
        self.conn.setblocking(False)


    def get_headers(self, data):

        """
        将请求头格式化成字典
        :param data:
        :return:
        """

        """
        请求头格式：
        GET / HTTP/1.1\r\n  # 请求首行，握手阶段还是使用http协议
        Host: 127.0.0.1:8080\r\n # 请求头
        Connection: Upgrade\r\n  # 表示要升级协议
        Pragma: no-cache\r\n
        Cache-Control: no-cache\r\n
        User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.72 Safari/537.36\r\n
        Upgrade: websocket\r\n  # 要升级协议到websocket协议
        Origin: http://localhost:63342\r\n
        Sec-WebSocket-Version: 13\r\n  # 表示websocket的版本。如果服务端不支持该版本，需要返回一个Sec-WebSocket-Versionheader，里面包含服务端支持的版本号
        Accept-Encoding: gzip, deflate, br\r\n
        Accept-Language: zh-CN,zh;q=0.9,en;q=0.8\r\n
        Sec-WebSocket-Key: 07EWNDBSpegw1vfsIBJtkg==\r\n # 对应服务端响应头的Sec-WebSocket-Accept，由于没有同源限制，websocket客户端可任意连接支持websocket的服务。这个就相当于一个钥匙一把锁，避免多余的，无意义的连接
        Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits\r\n\r\n
        """
        header_dict = {}
        data = str(data, encoding='utf-8')

        header, body = data.split('\r\n\r\n', 1)  # 因为请求头信息结尾都是\r\n,并且最后末尾部分是\r\n\r\n；
        header_list = header.split('\r\n')
        for i in range(0, len(header_list)):
            if i == 0:
                if len(header_list[i].split(' ')) == 3:
                    header_dict['method'], header_dict['url'], header_dict['protocol'] = header_list[i].split(' ')
            else:
                k, v = header_list[i].split(':', 1)
                header_dict[k] = v.strip()
        return header_dict

    def get_data(self, info):
        """
        前后端进行通信,对前端发生消息进行解密
        对返回消息进行解码比较复杂，详见数据帧格式解析
        """
        payload_len = info[1] & 127
        if payload_len == 126:
            extend_payload_len = info[2:4]
            mask = info[4:8]
            decoded = info[8:]
        elif payload_len == 127:
            extend_payload_len = info[2:10]
            mask = info[10:14]
            decoded = info[14:]
        else:
            extend_payload_len = None
            mask = info[2:6]
            decoded = info[6:]
        bytes_list = bytearray()  # 使用字节将数据全部收集，再去字符串编码，这样不会导致中文乱码
        for i in range(len(decoded)):
            chunk = decoded[i] ^ mask[i % 4]  # 异或运算
            bytes_list.append(chunk)
        try:
            body = str(bytes_list, encoding='utf-8')
        except:
            return None
        return body

    def reInit(self):
        self.socket_server.listen(5)
        logging.info("========================= WAITING FOR A CONNECTION =========================")
        print("========================= 等待一个连接 =========================")
        self.conn, address = self.socket_server.accept()
        self.conn.setblocking(False)
        self.status = SocketStatus.RUNNING
        self.data = b''
        self.is_loaded = False
        self.start_predict = False
        self.onnx = None
        self.raw_model_prediction = None

        # 当前正在预测的动作
        self.current_prediction_movement = None
        # 当前正在预测的动作的预测次数
        self.current_prediction_times = 0
        data = self.conn.recv(1024)  # 获取客户端发送的消息
        # 想将http协议的数据处理成字典的形式方便后续取值
        header_dict = self.get_headers(data)  # 将一大堆请求头转换成字典数据  类似于wsgiref模块
        client_random_string = header_dict['Sec-WebSocket-Key']  # 获取浏览器发送过来的随机字符串
        # magic string拼接
        magic_string = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'  # 全球共用的随机字符串 一个都不能写错
        # 确认握手Sec-WebSocket-Key固定格式：headers头部的Sec-WebSocket-Key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        # 确认握手的秘钥值为 传入的秘钥+magic_string，使用sha1算法加密，然后base64转码
        value = client_random_string + magic_string  # 拼接

        # 算法加密 对请求头中的sec-websocket-key进行加密
        ac = base64.b64encode(hashlib.sha1(value.encode('utf-8')).digest())  # 加密处理

        # 将处理好的结果再发送给客户端校验
        tpl = "HTTP/1.1 101 Switching Protocols\r\n" \
              "Upgrade:websocket\r\n" \
              "Connection: Upgrade\r\n" \
              "Sec-WebSocket-Accept: %s\r\n" \
              "WebSocket-Location: ws://127.0.0.1:8080\r\n\r\n"
        response_str = tpl % ac.decode('utf-8')  # 处理到响应头中

        # 将随机字符串给浏览器返回回去
        print(f"建立连接,加密验证key{ac}")
        logging.info("========================= SOCKET CONNECTION IS CREATED =========================")
        print("========================= 连接已经建立 =========================")
        print((bytes(response_str, encoding='utf-8')))
        self.conn.send(bytes(response_str, encoding='utf-8'))


    def send(self, code, msg, data=None):
        try:
            token = b'\x81'
            data = {"code": code, "data": data, "msg": msg}
            head_bytes = bytes(json.dumps(data), encoding='utf-8')  # 序列化并转成bytes,用于传输
            length = len(head_bytes)
            if length < 126:
                token += struct.pack('B', length)
            elif length <= 0xFFFF:
                token += struct.pack('!BH', 126, length)
            else:
                token += struct.pack('!BQ', 127, length)
            send_msg = token + head_bytes
            self.conn.sendall(send_msg)
        except socket.error as msg:
            print('Failed to send ' + json.dumps(data) + 'to client' + '. Error: ' + str(msg))

    def receive(self):
        try:
            data = self.conn.recv(1024)
            packet = self.get_data(data)
            if data:
                if packet is not None:
                    try:
                        request = json.loads(packet)
                        return request
                    except json.decoder.JSONDecodeError:
                        pass
            else:
                print("========================= CLIENT CONNECTION IS CLOSED =========================")
                self.conn.close()
        except socket.error as e:
            pass
        return None




    def close(self):
        self.conn.close()
        self.status = SocketStatus.CLOSED
        logging.info("========================= SOCKET CONNECTION IS CLOSED =========================")
        print("========================= 连接已经关闭 =========================")

    def parse_code_run_command(self, code: int):
        args = {}
        if code == statusCode.TEST_PREDICTION_CONNECT_TO_START:
            # 1. 加载ONNX模型
            self.raw_model_prediction = RawModelPrediction()
            self.hrnetModel = HRnetModelPrediction()
            self.yoloModel = YoloModelPrediction()
            first_img = cv2.imread('./1.jpg')
            first_press = np.zeros((120, 120))
            key_points = getPose3dRawModel(rawModel=self.raw_model_prediction,
                                           hrnetModel=self.hrnetModel,
                                           yoloModel=self.yoloModel,
                                           image_feature=first_img,
                                           pressure_feature=first_press)
            self.is_loaded = True

            self.send(code=statusCode.TEST_PREDICTION_CONNECT_IS_STARTED,
                      msg=statusCode.TEST_PREDICTION_CONNECT_IS_STARTED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_START_PREDICTION:
            # 开始进行预测
            self.start_predict = True
        elif code == statusCode.TEST_PREDICTION_CONNECT_TO_CLOSE:
            # 客户端主动断开链接
            args['stop_connect'] = True
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_IS_CLOSED,
                      msg=statusCode.TEST_PREDICTION_CONNECT_IS_CLOSED_MSG)
            self.close()
            self.reInit()

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_DEEP_SQUAT:
            # TODO：预测深蹲
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.DEEP_SQUAT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_DEEP_SQUAT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_HURDLE_STEP_LEFT:
            # TODO：预测跨栏架步-左
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.HURDLE_STEP_LEFT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_HURDLE_STEP_LEFT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_HURDLE_STEP_RIGHT:
            # TODO：预测跨栏架步-右
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.HURDLE_STEP_RIGHT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_HURDLE_STEP_RIGHT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_INLINE_LUNGE_LEFT:
            # TODO：预测前后分腿蹲-左
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.INLINE_LUNGE_LEFT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_INLINE_LUNGE_LEFT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_INLINE_LUNGE_RIGHT:
            # TODO：预测前后分腿蹲-右
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.INLINE_LUNGE_RIGHT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_INLINE_LUNGE_RIGHT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SHOULDER_MOBILITY_LEFT:
            # TODO：预测肩部灵活性-左
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.SHOULDER_MOBILITY_LEFT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SHOULDER_MOBILITY_LEFT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SHOULDER_MOBILITY_RIGHT:
            # TODO：预测肩部灵活性-右
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.SHOULDER_MOBILITY_RIGHT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SHOULDER_MOBILITY_RIGHT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ACTIVE_STRAIGHT_LEG_RAISE_LEFT:
            # TODO：预测直腿主动上抬-左
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.ACTIVE_STRAIGHT_LEG_RAISE_LEFT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ACTIVE_STRAIGHT_LEG_RAISE_LEFT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ACTIVE_STRAIGHT_LEG_RAISE_RIGHT:
            # TODO：预测直腿主动上抬-右
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.ACTIVE_STRAIGHT_LEG_RAISE_RIGHT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ACTIVE_STRAIGHT_LEG_RAISE_RIGHT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_TRUNK_STABILITY_PUSH_UP:
            # TODO：预测躯干稳定俯卧撑
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.TRUNK_STABILITY_PUSH_UP
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_TRUNK_STABILITY_PUSH_UP,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ROTARY_STABILITY_QUADRUPED_LEFT:
            # TODO：预测躯干旋转稳定性-左
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.ROTARY_STABILITY_QUADRUPED_LEFT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ROTARY_STABILITY_QUADRUPED_LEFT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ROTARY_STABILITY_QUADRUPED_RIGHT:
            # TODO：预测躯干旋转稳定性-右
            self.current_prediction_times = 0
            self.current_prediction_movement = FMSMovement.ROTARY_STABILITY_QUADRUPED_RIGHT
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED,
                      data=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ROTARY_STABILITY_QUADRUPED_RIGHT,
                      msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SWITCHED_MSG)

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_NEXT_TIMES:
            # 接收到 进行下一次预测：
            # 如果当前预测的动作为None
            if self.current_prediction_movement is None:
                self.send(code=statusCode.PREDICTION_MOVEMENT_MISSING_ERROR,
                          msg=statusCode.PREDICTION_MOVEMENT_MISSING_ERROR_MSG)
            elif self.current_prediction_times >= 3:
                self.send(code=statusCode.PREDICTION_MOVEMENT_TIMES_OUT_ERROR,
                          msg=statusCode.PREDICTION_MOVEMENT_TIMES_OUT_ERROR_MSG)
            else:
                self.current_prediction_times = self.current_prediction_times + 1
                # 开始新的一轮预测
                self.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_IS_NEXT_TIMES,
                          msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_IS_NEXT_TIMES_MSG)
                # 当前开始预测的时间
                current_prediction_times_start_time = datetime.datetime.now()
                args["current_prediction_score_times_start_time"] = current_prediction_times_start_time
                args[
                    "current_prediction_score_times_end_time"] = current_prediction_times_start_time + datetime.timedelta(
                    seconds=1)

        return args
