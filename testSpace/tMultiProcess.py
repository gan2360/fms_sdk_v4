"""
@Project ：fms_predict_sdk_v2
@File    ：tMultiProcess.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/14 12:52
@Des     ：
"""
import time
import multiprocessing
from multiprocessing import Process
import cv2
from camera.camera import CameraClient
from client.lokiClient import LokiSDKClient

def run1(que):
    camera = CameraClient()

    while 1:
        img = camera.get_frame()
        length = que.qsize()

        if length > 2:
            for i in range(length - 2):
                frame = que.get()  # 清除缓存
        que.put(img)
        # pipe.send(img_q)

    cv2.destroyAllWindows()
    cap.release()


def run2(que):
    loki = LokiSDKClient()

    while 1:
        pressure = loki.get_all_handle_data()
        que.put(pressure)


def test_2(que,que2):
    while 1:
        img = que.get()
        # img = pipe.recv()
        # img = cv2.resize(img,(360,240))
        print(time.time(), img.shape)
        pressure = que2.get()
        # img = pipe.recv()
        # img = cv2.resize(img,(360,240))
        print(time.time(), "p")
    cv2.destroyAllWindows()


def test_1(que):
    while 1:
        pressure = que.get()
        # img = pipe.recv()
        # img = cv2.resize(img,(360,240))
        print(time.time(), len(pressure))


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    manager2 = multiprocessing.Manager()
    que = manager.Queue()
    que2 = manager.Queue()
    test1_start = 1
    test2_start = 1

    t1 = Process(target=run1, args=(que,))
    t2 = Process(target=run2, args=(que2,))
    t1.start()
    t2.start()
    test_2(que, que2)
    t1.terminate()
    t2.terminate()
