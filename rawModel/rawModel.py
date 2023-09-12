"""
@Project ：fms_predict_sdk_v2
@File    ：rawModel.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/14 15:00
@Des     ：模型实例，记得修改下面的地址
"""
import numpy as np
import onnxruntime
import torch

from rawModel.vaTPose import VaTPose

class RawModelPrediction:
    def __init__(self):
        use_gpu = torch.cuda.is_available()
        self.device = 'cuda:0' if use_gpu else 'cpu'
        np.random.seed(0)
        torch.manual_seed(0)
        # self.model = VaTPose(0.5)  # model
        self.model = onnxruntime.InferenceSession('D:\\GuoJ\\fms_predict_sdk_v4\\onnx_file\\vat_pose.onnx', providers=['CUDAExecutionProvider'])


    def get_pose3d(self, tactile, visual): # (120, 120), (22,2)
        with torch.set_grad_enabled(False):
            inputs = {'input.1': tactile, 'input':visual}
            keypoint_out = self.model.run(['output'], inputs)[0]
            b = [-800, -800, 0]
            resolution = 100
            scale = 19
            keypoint_out = keypoint_out * scale
            keypoint_out = keypoint_out * resolution + b
            keypoint_out = keypoint_out / 10
        return keypoint_out
