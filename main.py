"""
@Project ：fms_predict_sdk_v2
@File    ：main.py
@IDE     ：PyCharm
@Author  ：Tristen
@Date    ：2023/05/05 19:20
@Des     ：启动程序，主程序
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import multiprocessing
import time
from datetime import datetime
from multiprocessing import Process, Event
import random
import cv2
from camera.camera import CameraClient
from client.lokiClient import LokiSDKClient
from client.pressurAlign import align
from rawModel.predictor import getPose3dRawModel
from rawModel.lib.yolov3.yoloModel import YoloModelPrediction
from rawModel.lib.hrnet.hrnetModel import HRnetModelPrediction
from server.parse import parse_data_from_socket
from server.server import SocketServer, SocketStatus
from utils import statusCode
from utils.exceptions.dataDecodeException import DataDecodeException
from utils.exceptions.parameterMissingException import ParameterMissingException
import numpy as np
from MySerial.concat import concat_pressure
from SerialOp.SerialOp import SerialeOp


# -----------------------------test_data_start--------------------------

def getData(video_path, pressure_path):
    pressure = np.load(pressure_path)['pressure']
    video = cv2.VideoCapture(video_path)
    return pressure, video


def getPressure(queue, e, e2):
    pressure, video = getData('./fmsTestData/video.mp4', './fmsTestData/pressure.npz')
    e2.set()  # 压力垫启动成功
    e.wait()  # 等摄像头启动
    index = 0
    while 1:
        p_data = pressure[index]
        index += 1
        # pressure = np.delete(pressure, 0 ,axis=0)
        queue.put({"t": time.time(), "d": p_data})
        # print('2---', time.time())
        time.sleep(0.0001)


def getVideo(queue, e, e2):
    pressure, video = getData('./fmsTestData/video.mp4', './fmsTestData/pressure.npz')
    e.set()  # 摄像头启动成功
    e2.wait()  # 等待压力垫启动
    while 1:
        ret, img = video.read()
        # original_height, original_width = img.shape[:2]
        # target_width = int(original_width * 0.5)
        # target_height = int(original_height * 0.5)
        # resized_image = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        if ret:
            queue.put({'t': time.time(), 'd': img})
        time.sleep(0.0001)
        # print('1---', time.time())
    video.release()


class LocalSerialOp:
    def __init__(self):
        self.pressure = np.load('./fmsTestData/pressure.npz')['pressure']
        self.index = -1

    def get_all_pressure(self):
        self.index += 1
        return self.pressure[self.index]


class LocalCamera:
    def __init__(self):
        self.camera = cv2.VideoCapture('./fmsTestData/video.mp4')

    def get_frame(self):
        return self.camera.read()[1]


# -----------------------------test_data_end----------------------------


def mapData(camera, ser_op):
    img = camera.get_frame()
    pressure = ser_op.get_all_pressure()
    # pressure = concat_pressure(pressure)
    return [img, pressure]


# 预测分数的开始时间和预测分数的结束时间
currentPredictionScoreTimesStartTime = None
currentPredictionScoreTimesEndTime = None

if __name__ == '__main__':
    cur_action_code = None
    key_points = None
    cur_frames = []
    processInstancesAlive = False
    hrnetModel = HRnetModelPrediction()
    yoloModel = YoloModelPrediction()
    socket_server = SocketServer("127.0.0.1")
    while socket_server.status != SocketStatus.CLOSED:
        # 如果获取数据的进程没有启动，并且已经要求开始预测了
        # 并修改状态
        if not processInstancesAlive and socket_server.start_predict:
            # 实时
            # camera = CameraClient()
            # ser_op = SerialeOp()
            # 本地
            camera = LocalCamera()
            ser_op = LocalSerialOp()
            processInstancesAlive = True
        # 如果状态为T，两个进程都已经启动，将两个存储了数据的队列进行对齐
        # 返回对齐的数据或者None
        if processInstancesAlive:
            map_data = mapData(camera, ser_op)
            if len(map_data) > 0:
                # print("predict")
                # 将对齐的数据进行输入，开始进行预测可用于可视化的人体骨骼关节点
                key_points = getPose3dRawModel(rawModel=socket_server.raw_model_prediction,
                                               hrnetModel=hrnetModel,
                                               yoloModel=yoloModel,
                                               image_feature=map_data[0],
                                               pressure_feature=map_data[1])
                socket_server.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_PREDICTION,
                                   msg=datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                   data=key_points)
        # 这一部分是用来发送预测结果分数的，当客户端发来开始预测动作命令时，会将开始时间和结束时间设置，如果结束，则返回预测的分值发送到客户端
        if currentPredictionScoreTimesStartTime is not None and currentPredictionScoreTimesEndTime is not None:
            current_time = datetime.now()
            cur_frames.append(np.array(key_points))
            if current_time >= currentPredictionScoreTimesEndTime:
                # 发送一个预测的结果code
                # score = analayse_frames(curframes)
                print(cur_action_code, cur_frames)
                cur_frames = []
                socket_server.send(code=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_PREDICTION_SCORE,
                                   msg=statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_PREDICTION_SCORE_MSG,
                                   data=random.randint(1, 3))
                currentPredictionScoreTimesStartTime = None
                currentPredictionScoreTimesEndTime = None
        # 客户端读取数据，并根据code进行执行
        try:
            receive = socket_server.receive()
            if receive is None:
                continue
            code = parse_data_from_socket(receive)
            print(code)
            # 根据code执行
            args = socket_server.parse_code_run_command(code)
            print(args)
            if "current_prediction_score_times_start_time" in args and "current_prediction_score_times_end_time" in args:
                currentPredictionScoreTimesStartTime = args["current_prediction_score_times_start_time"]
                currentPredictionScoreTimesEndTime = args["current_prediction_score_times_end_time"]
            else:
                cur_action_code = code
        except DataDecodeException as e:
            socket_server.send(code=statusCode.DATA_DECODE_ERROR, msg=str(e))
            continue
        except ParameterMissingException as e:
            socket_server.send(code=statusCode.PARAMETER_MISSING, msg=str(e))
            continue



"""

1.预测开始的时间
2.收集接下来2秒的数据帧存储
3.分析数据帧，得出评分
4.返回评分

"""
