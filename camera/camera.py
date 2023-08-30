"""
@Project ：fms_predict_sdk_v2
@File    ：camera.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/6 11:21
@Des     ：相机实例
"""
import cv2
import time


class CameraClient:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(self.cap.isOpened())

    def get_frame(self):
        if not self.cap.isOpened():
            print("device error")
            return None
        else:
            ret, frame = self.cap.read()
            # print(frame.shape)
            if ret:
                return frame

    def close(self):
        self.cap.release()
