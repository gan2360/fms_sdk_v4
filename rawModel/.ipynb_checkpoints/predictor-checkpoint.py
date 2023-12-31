"""
@Project ：fms_predict_sdk_v2
@File    ：predictor.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/14 15:16
@Des     ：
"""
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from lib.preprocess import h36m_coco_format
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose

# def visualPreprocess(visual):
#     preprocess = transforms.Compose([
#         transforms.CenterCrop((640, 720)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     visual = Image.fromarray(np.uint8(visual))
#     visual = preprocess(visual)
#     visual = visual.numpy()
#     return visual

def rearrange_17to22(kp2d):
    kp2d_new = []
    temp = [0,0]
    kp2d_new.append(kp2d[0])
    kp2d_new.append(kp2d[1])
    kp2d_new.append(kp2d[2])
    kp2d_new.append(kp2d[3])
    kp2d_new.append(kp2d[3])
    kp2d_new.append(kp2d[3])
    kp2d_new.append(kp2d[3])
    # kp2d_new.append(temp)
    # kp2d_new.append(temp)
    # kp2d_new.append(temp)
    
    kp2d_new.append(kp2d[4])
    kp2d_new.append(kp2d[5])
    kp2d_new.append(kp2d[6])
    kp2d_new.append(kp2d[6])
    kp2d_new.append(kp2d[6])
    kp2d_new.append(kp2d[6])
    # kp2d_new.append(temp)
    # kp2d_new.append(temp)
    # kp2d_new.append(temp)
    
    kp2d_new.append(kp2d[7])
    kp2d_new.append(kp2d[8])
    kp2d_new.append(kp2d[10])
    
    kp2d_new.append(kp2d[11])
    kp2d_new.append(kp2d[12])
    kp2d_new.append(kp2d[13])
    
    kp2d_new.append(kp2d[14])
    kp2d_new.append(kp2d[15])
    kp2d_new.append(kp2d[16])
    
    return np.array(kp2d_new)

def normalize_screen_coordinates(X, w, h):
    if X.shape[-1] != 2:
        print('X.shape: ', X.shape)
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]

def get_pose2D(hrnetModel, yoloModel, frame):
    # print('\nGenerating 2D pose...')
    keypoints, scores = hrnetModel.gen_video_kpts(yoloModel, frame, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    # print('Generating 2D pose successful!')

    keypoints = keypoints.reshape(17,2)
    keypoints = normalize_screen_coordinates(keypoints, 1280, 720)
    keypoints = rearrange_17to22(keypoints)

    return keypoints.reshape(22,2)

def getPose3dRawModel(rawModel, hrnetModel, yoloModel, image_feature, pressure_feature):
    image_feature = get_pose2D(hrnetModel, yoloModel, image_feature) # (22,2)

    visual = torch.tensor(image_feature.reshape((1, 1, 22, 2)), dtype=torch.float, device="cuda:0")

    tactile = torch.tensor(pressure_feature.reshape((1, 1, 120, 120)), dtype=torch.float, device="cuda:0")

    key_points = rawModel.get_pose3d(tactile, visual)
    return key_points.tolist()
