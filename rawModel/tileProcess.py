"""
@Project ：fms_predict_sdk_v2
@File    ：tileProcess.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/14 15:23
@Des     ：
"""
import torch.nn as nn


class tileProcess(nn.Module):
    def __init__(self, windowSize):
        super(tileProcess, self).__init__()  # tactile 120*120
        if windowSize == 0:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        else:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(2 * windowSize, 32, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))  # output = 60 * 60

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2))  # output = 30 * 30

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(7, 7)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))  # output = 24 * 24

        self.conv_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=3))  # 8 * 8

    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.conv_5(output)
        output = self.conv_6(output)

        return output
