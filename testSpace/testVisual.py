"""
@Project ：fms_predict_sdk_v2
@File    ：testVisual.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/14 17:00
@Des     ：将压力数据和图像数据可视化
"""
import numpy as np
from matplotlib import pyplot as plt


def plotTouch(touch_frame):
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111)
    pos_x, pos_y = np.meshgrid(
        np.linspace(0, 120, 120, endpoint=False),
        np.linspace(0, 120, 120, endpoint=False)
    )
    touch_frame = touch_frame * 15
    print(touch_frame.shape)
    ax.scatter(pos_x, pos_y, linewidths=.2, edgecolors='k', c='k', s=touch_frame)
    ax.set_aspect('equal')
    plt.axis('off')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # img = data.reshape((720, 720, 3))[..., ::-1]
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]
    return img

