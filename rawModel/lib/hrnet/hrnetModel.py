"""
@Project ：fms_predict_sdk_v2
@File    ：rawModel.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/14 15:00
@Des     ：模型实例，记得修改下面的地址
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import os
import os.path as osp
import argparse
import time
import numpy as np
import onnxruntime
from tqdm import tqdm
import json
import torch
import torch.backends.cudnn as cudnn
import cv2
import copy

from rawModel.lib.hrnet.lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from rawModel.lib.hrnet.lib.config import cfg, update_config
from rawModel.lib.hrnet.lib.utils.transforms import *
from rawModel.lib.hrnet.lib.utils.inference import get_final_preds
from rawModel.lib.hrnet.lib.models import pose_hrnet

cur_path = os.path.abspath(__file__)
cfg_dir = os.path.join(cur_path, '../experiments/')
model_dir = 'D:\\GuoJ\\fms_predict_sdk_v4\\rawModel\\lib\\hrnet\\'

# Loading human detector model
from rawModel.lib.yolov3.human_detector import load_model as yolo_model
from rawModel.lib.yolov3.human_detector import yolo_human_det as yolo_det
from rawModel.lib.sort.sort import Sort

from rawModel.spatialSoftmax3D import SpatialSoftmax3D
from rawModel.vaTPose import VaTPose

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_adam_lr1e-3.yaml',
                        help='experiment configure file name')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help="Modify config options using the command-line")
    parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_384x288.pth',
                        help='The model directory')
    parser.add_argument('--det-dim', type=int, default=416,
                        help='The input dimension of the detected image')
    parser.add_argument('--thred-score', type=float, default=0.30,
                        help='The threshold of object Confidence')
    parser.add_argument('-a', '--animation', action='store_true',
                        help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1,
                        help='The maximum number of estimated poses')
    parser.add_argument("-v", "--video", type=str, default='camera',
                        help="input video file name")
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    return args


def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
     
# load model
def model_load(config):
        model = pose_hrnet.get_pose_net(config, is_train=False)
        if torch.cuda.is_available():
            model = model.cuda()

        state_dict = torch.load(config.OUTPUT_DIR)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k  # remove module.
            #  print(name,'\t')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        # print('HRNet network successfully loaded')
        
        return model
    
class HRnetModelPrediction:
    
    def __init__(self):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # args = parse_args()
        # reset_config(args)
        # self.pose_model = model_load(cfg)
        self.pose_model = onnxruntime.InferenceSession('D:\\GuoJ\\fms_predict_sdk_v4\\rawModel\\lib\\hrnet\\hrnet_model.onnx',providers=['CUDAExecutionProvider'])

        
    def gen_video_kpts(self, yoloModel, frame, det_dim=416, num_peroson=1, gen_output=False):
        """
        yolo:0.04
        hr:0.03
        """
        # Updating configuration
        # args = self.parse_args()
        # self.reset_config(args)

        # Loading detector and pose model, initialize sort for track
        # human_model = yolo_model(inp_dim=det_dim)
        # human_model = yoloModel
        # pose_model = self.model_load(cfg)
        people_sort = Sort(min_hits=0)
        bboxs, scores = yoloModel.yolo_human_det(frame)  # bobxs 代表边框的四个角的坐标，frame是原始的1280*720的图像
        if bboxs is None or not bboxs.any():  # any(), Python 中用于判断可迭代对象（如列表、元组）中的元素是否满足某个条件的方法。它返回一个布尔值
            print('No person detected!')
            bboxs = bboxs_pre
            scores = scores_pre
        else:
            bboxs_pre = copy.deepcopy(bboxs) 
            scores_pre = copy.deepcopy(scores) 

        # Using Sort to track people
        people_track = people_sort.update(bboxs)

        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
            people_track_ = people_track_[::-1]

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        with torch.no_grad():
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)  # frame是原始的1280*720的图像，inputs是处理完之后的

            inputs = inputs[:, [2, 1, 0]]
            inputs = inputs.cpu().numpy()
            inputs = {'input':inputs}
            output = self.pose_model.run(['output'], inputs)[0]
            preds, maxvals = get_final_preds(cfg, output, np.asarray(center), np.asarray(scale))


        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt
        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        return kpts.reshape(1,1,17,2), scores.reshape(1,1,17)

