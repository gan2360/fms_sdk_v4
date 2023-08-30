"""
@Project ：fms_predict_sdk_v2
@File    ：testScript.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/8/3 15:13
@Des     ：
"""

import numpy as np  # test
from multiprocessing import Process
import multiprocessing
import cv2
import time

def getData(video_path, pressure_path):
    pressure = np.load(pressure_path)['pressure']
    video = cv2.VideoCapture(video_path)
    return pressure, video


def getPressure(queue):
    pressure, video = getData('./video.mp4', './pressure.npz')
    index = 0
    while 1:
        p_data = pressure[index]
        index += 1
        # pressure = np.delete(pressure, 0 ,axis=0)
        queue.put({"t": time.time(), "d":p_data})
        print('2---', time.time())
        time.sleep(0.01)


def getVideo(queue):
    pressure, video = getData('./video.mp4', './pressure.npz')
    while 1:
        ret, img = video.read()
        if ret:
            queue.put({'t':time.time(), 'd':img})
        print('1---', time.time())
        time.sleep(0.02)
    video.release()


#-----------------------------test_data_end----------------------------

if __name__ == '__main__':
    cameraManager = multiprocessing.Manager()
    lokiManager = multiprocessing.Manager()
    cameraQueue = cameraManager.Queue()
    lokiQueue = lokiManager.Queue()
    # 创建相机，Loki的进程
    cameraProcessInstance = Process(target=getVideo, args=(cameraQueue,))
    lokiProcessInstance = Process(target=getPressure, args=(lokiQueue,))
    lokiProcessInstance.start()
    cameraProcessInstance.start()
    while 1:
        pass
