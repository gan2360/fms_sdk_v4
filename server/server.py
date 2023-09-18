"""
@Project ：fms_predict_sdk_v2
@File    ：server.py
@IDE     ：PyCharm
@Author  ：Tristen
@Date    ：2023/05/05 19:20
@Des     ：作为传输的服务，一个Socket实例
"""
import datetime
import json
import logging
import socket
import struct
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
        # 设置成非阻塞
        self.socket_server.bind(self.address)
        self.socket_server.listen()
        logging.info("========================= WAITING FOR A CONNECTION =========================")
        print("========================= 等待一个连接 =========================")
        self.conn, address = self.socket_server.accept()
        self.conn.setblocking(False)
        logging.info("========================= SOCKET CONNECTION IS CREATED =========================")
        print("========================= 连接已经建立 =========================")
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

    def reInit(self):
        self.socket_server.listen()
        logging.info("========================= WAITING FOR A CONNECTION =========================")
        print("========================= 等待一个连接 =========================")
        self.conn, address = self.socket_server.accept()
        self.conn.setblocking(False)
        logging.info("========================= SOCKET CONNECTION IS CREATED =========================")
        print("========================= 连接已经建立 =========================")
        self.status = SocketStatus.RUNNING
        self.data = b''

        self.is_loaded = False
        self.start_predict = False

        # 当前正在预测的动作
        self.current_prediction_movement = None
        # 当前正在预测的动作的预测次数
        self.current_prediction_times = 0

    def send(self, code, msg, data=None):
        try:
            data = {"code": code, "data": data, "msg": msg}
            head_bytes = bytes(json.dumps(data), encoding='utf-8')  # 序列化并转成bytes,用于传输
            head_len_bytes = struct.pack('!I', len(head_bytes))  # 这4个字节里只包含了一个数字,该数字是报头的长度
            self.conn.sendall(head_len_bytes + head_bytes)
        except socket.error as msg:
            logging.error('Failed to send ' + json.dumps(data) + 'to client' + '. Error: ' + str(msg))

    def receive(self):
        try:
            data = self.conn.recv(1024)
            if data:
                self.data += data
                # 如果已经接收到了固定长度的帧头，那么就解析，获取消息体的长度
                if len(self.data) >= header_size:
                    length = struct.unpack("!I", self.data[:header_size])[0]
                    # 如果已经收到了完整的消息体，那么就去处理
                    if len(self.data[header_size:]) >= length:
                        packet = self.data[header_size:header_size + length]
                        self.data = self.data[header_size + length:]
                        try:
                            request = json.loads(packet)
                            return request
                        except json.decoder.JSONDecodeError:
                            raise DataDecodeException("'{}' can not be parsed".format(str(request.decode("utf-8"))))
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
            # self.onnx = ONNXModel()

            self.raw_model_prediction = RawModelPrediction()

            self.is_loaded = True
            self.send(code=statusCode.TEST_PREDICTION_CONNECT_IS_STARTED,
                      msg=statusCode.TEST_PREDICTION_CONNECT_IS_STARTED_MSG)
        elif code == statusCode.TEST_PREDICTION_CONNECT_START_PREDICTION:
            # 开始进行预测
            self.start_predict = True
        elif code == statusCode.TEST_PREDICTION_CONNECT_TO_CLOSE:
            # 客户端主动断开链接
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
