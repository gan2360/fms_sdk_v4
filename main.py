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



def processCamera(queue,e, e2, wait_pressure=None):
    # 相机进程
    # 创建一个相机的实例
    camera = CameraClient()
    e.set() # 摄像头启动成功
    e2.wait() # 等待压力垫启动
    print('c is ok')
    while 1:
        # 循环获取视频相机帧，将当前的视频帧的时间和数据形成json数据包，放入队列当中
        # wait_pressure.wait()
        img = camera.get_frame()
        queue.put({'t': time.time(), 'd': img})
        # print(1, time.time())
    cv2.destroyAllWindows()
    cap.release()


def processLoki(queue, e, e2, wait_camera=None):
    # Loki SDK 实例， 创建一个获取压力垫的实例
    ser_op = SerialeOp()
    e2.set() #压力垫启动成功
    e.wait() #等摄像头线启动
    print('p is ok')
    while 1:
        # 循环获取所有传感器压力垫的数据
        # wait_camera.wait()
        pressure = ser_op.get_all_pressure()
        if pressure:
            # 如果压力点存在的话，对压力点数据进行预处理
            processedPressure = concat_pressure(pressure)
            # 将处理后的压力点数据形成json包，放入对应的队列当中
            queue.put({'t': time.time(), 'd': processedPressure})
        # print(2, time.time())
    loki.close()

# -----------------------------test_data_start--------------------------


def getData(video_path, pressure_path):
    pressure = np.load(pressure_path)['pressure']
    video = cv2.VideoCapture(video_path)
    return pressure, video


def getPressure(queue, e, e2):
    pressure, video = getData('./fmsTestData/video.mp4', './fmsTestData/pressure.npz')
    e2.set() #压力垫启动成功
    e.wait() #等摄像头启动
    index = 0
    while 1:
        p_data = pressure[index]
        index += 1
        # pressure = np.delete(pressure, 0 ,axis=0)
        queue.put({"t": time.time(), "d":p_data})
        # print('2---', time.time())
        time.sleep(0.0001)


def getVideo(queue, e, e2):
    pressure, video = getData('./fmsTestData/video.mp4', './fmsTestData/pressure.npz')
    e.set() # 摄像头启动成功
    e2.wait() # 等待压力垫启动
    while 1:
        ret, img = video.read()
        # original_height, original_width = img.shape[:2]
        # target_width = int(original_width * 0.5)
        # target_height = int(original_height * 0.5)
        # resized_image = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        if ret:
            queue.put({'t':time.time(), 'd':img})
        time.sleep(0.0001)
        # print('1---', time.time())
    video.release()


#-----------------------------test_data_end----------------------------


def mapVideoPressure(cQueue, lQueue):
    """
    视频帧和压力帧对齐算法
    :param cQueue: 存储相机帧的队列
    :param lQueue: 存储Loki压力垫的队列
    :return:
    """
    mPressureData = None
    if not cQueue.empty():
        # 如果相机帧的队列为空，则出队，获取当前视频帧
        frameDict = cQueue.get()
        # 当前图像帧的时间戳
        frameDictTimestamp = frameDict["t"]
        # 当前图像的数据
        frameDictData = frameDict["d"]
        # 如果压力垫的队列不为空，则不断读取，直到匹配或实在无法匹配了
        while not lQueue.empty():
            pressureDict = lQueue.get()
            pressureDictTimestamp = pressureDict["t"]
            pressureDictData = pressureDict["d"]
            # if frameDictTimestamp - pressureDictTimestamp >= 1:
            #     wait_camera.clear()
            #     break
            # if frameDictTimestamp - pressureDictTimestamp <= -1:
            #     wait_pressure.clear()
            #     break
            # 时间戳的对比，如果两个时间戳小于阈值则表示当前压力垫的数据能和图像对对齐，如果超过一个阈值，则表示无法对齐
            if abs(frameDictTimestamp - pressureDictTimestamp) <= 0.06:
                mPressureData = pressureDictData
                break
            elif abs(frameDictTimestamp - pressureDictTimestamp) > 0.4:
                break

        if frameDictData is not None and mPressureData is not None:
            return [frameDictData, mPressureData]
    return []

# 预测分数的开始时间和预测分数的结束时间
currentPredictionScoreTimesStartTime = None
currentPredictionScoreTimesEndTime = None

if __name__ == '__main__':
    e = Event() #摄像头是否准备好
    e2 = Event() #压力垫是否准备好
    wait_camera = Event() # 是否要等待摄像头
    wait_camera.set()
    wait_pressure = Event() # 是否要等待压力垫
    wait_pressure.set()
    processInstancesAlive = False
    hrnetModel = HRnetModelPrediction()
    yoloModel = YoloModelPrediction()
    # 创建一个和客户端通信的实例
    # socket_server = SocketServer("192.168.10.110")
    socket_server = SocketServer("127.0.0.1")
    # 用于获取数据。通过的是输入输出队列
    cameraManager = multiprocessing.Manager()
    lokiManager = multiprocessing.Manager()
    cameraQueue = cameraManager.Queue()
    lokiQueue = lokiManager.Queue()
    # 创建相机，Loki的进程
    cameraProcessInstance = Process(target=processCamera, args=(cameraQueue, e, e2))
    lokiProcessInstance = Process(target=processLoki, args=(lokiQueue, e, e2))
    # 不断循环，刚开始socket 的状态是running，这个状态表示已经关掉了
    while socket_server.status != SocketStatus.CLOSED:
        # 如果获取数据的进程没有启动，并且已经要求开始预测了
        # 则启动两个进程
        # 并修改状态
        if not processInstancesAlive and socket_server.start_predict:
            cameraProcessInstance.start()
            lokiProcessInstance.start()
            processInstancesAlive = True
        # 如果状态为T，两个进程都已经启动，将两个存储了数据的队列进行对齐
        # 返回对齐的数据或者None
        if processInstancesAlive:
            map_data = mapVideoPressure(cameraQueue, lokiQueue)
            if len(map_data) > 0:
                print("predict")
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
            if current_time >= currentPredictionScoreTimesEndTime:
                # 发送一个预测的结果code
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
            # 根据code执行
            args = socket_server.parse_code_run_command(code)
            if "current_prediction_score_times_start_time" in args and "current_prediction_score_times_end_time" in args:
                currentPredictionScoreTimesStartTime = args["current_prediction_score_times_start_time"]
                currentPredictionScoreTimesEndTime = args["current_prediction_score_times_end_time"]
        except DataDecodeException as e:
            socket_server.send(code=statusCode.DATA_DECODE_ERROR, msg=str(e))
            continue
        except ParameterMissingException as e:
            socket_server.send(code=statusCode.PARAMETER_MISSING, msg=str(e))
            continue
    # 关闭所有进程
    cameraProcessInstance.terminate()
    lokiProcessInstance.terminate()
