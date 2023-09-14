"""
@Project ：fms_predict_sdk_v4
@File    ：score.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/9/12 9:15
@Des     ：
"""
from server.server import FMSMovement
from utils import statusCode
import numpy as np


class ScoreOp:
    def __init__(self):
        pass

    def score(self, movement_code, movement_frames):
        pass

    def parse_code_get_score(self, code, movements):
        score = 1
        args = {}
        if code == statusCode.TEST_PREDICTION_CONNECT_TO_START:
            pass
        elif code == statusCode.TEST_PREDICTION_CONNECT_START_PREDICTION:
            # 开始进行预测
            pass
        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_DEEP_SQUAT:
            # TODO：预测深蹲
            i = np.argmin(movements[:, 0, 2])
            key_frame = movements[i]
            if (key_frame[0][2]) > (key_frame[2][2] + key_frame[8][2]) / 2:
                score = 1
            elif abs(key_frame[21][0] - key_frame[18][0]) < 2 and key_frame[2][1] <= key_frame[3][1] and key_frame[8][1] >= key_frame[9][1]:
                score = 3
            else:
                score = 2


        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_HURDLE_STEP_LEFT:
            # TODO：预测跨栏架步-左
            len = movements.shape()[0]
            p_18_x = movements[:, 18, 0]
            p_21_x = movements[:, 21, 0]
            p_18_z = movements[:, 18, 2]
            p_21_z = movements[:, 21, 2]
            x_dif = np.abs(p_21_x - p_18_x)
            z_dif = np.abs(p_21_z - p_18_z)
            hand_unsatisfy = max(np.sum(z_dif > 2), np.sum(x_dif > 2))
            hand_unsatisfy_rate = hand_unsatisfy / len
            p_13 = movements[:, 13, :]
            p_13_distances = np.linalg.norm(p_13[1:] - p_13[:-1], axis=1)
            change_sum = np.sum(p_13_distances > 2)
            change_rate = change_sum / len
            p_1 = movements[:, 1, :]
            p_2 = movements[:, 2, :]
            p_3 = movements[:, 3, :]
            p_123 = np.stack((p_3, p_2, p_1), axis=1)
            normal_vectors = np.cross(p_123[:, 1] - p_123[:, 0], p_123[:, 2] - p_123[:, 0])
            target_vector = np.array([0, 1, 0])
            target_vector /= np.linalg.norm(target_vector)
            angles = np.degrees(np.arccos(np.dot(normal_vectors, target_vector)))
            offset_sum = np.sum(angles > 5)
            offset_rate = offset_sum / len
            if hand_unsatisfy_rate <= 0.2 and change_rate <=0.2 and offset_rate <= 0.2:
                score = 3
            elif hand_unsatisfy_rate > 0.6 or offset_rate > 0.6 or change_rate > 0.6:
                score = 1
            else:
                score = 2

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_HURDLE_STEP_RIGHT:
            # TODO：预测跨栏架步-右
            len = movements.shape()[0]
            p_18_x = movements[:, 18, 0]
            p_21_x = movements[:, 21, 0]
            p_18_z = movements[:, 18, 2]
            p_21_z = movements[:, 21, 2]
            x_dif = np.abs(p_21_x - p_18_x)
            z_dif = np.abs(p_21_z - p_18_z)
            hand_unsatisfy = max(np.sum(z_dif > 2), np.sum(x_dif > 2))
            hand_unsatisfy_rate = hand_unsatisfy / len
            p_13 = movements[:, 13, :]
            p_13_distances = np.linalg.norm(p_13[1:] - p_13[:-1], axis=1)
            change_sum = np.sum(p_13_distances > 2)
            change_rate = change_sum / len
            p_1 = movements[:, 7, :]
            p_2 = movements[:, 8, :]
            p_3 = movements[:, 9, :]
            p_123 = np.stack((p_3, p_2, p_1), axis=1)
            normal_vectors = np.cross(p_123[:, 1] - p_123[:, 0], p_123[:, 2]-p_123[:, 0])
            target_vector = np.array([0, 1, 0])
            target_vector /= np.linalg.norm(target_vector)
            angles = np.degrees(np.arccos(np.dot(normal_vectors, target_vector)))
            offset_sum = np.sum(angles > 5)
            offset_rate = offset_sum / len
            if hand_unsatisfy_rate <= 0.2 and change_rate <=0.2 and offset_rate <= 0.2:
                score = 3
            elif hand_unsatisfy_rate > 0.6 or offset_rate > 0.6 or change_rate > 0.6:
                score = 1
            else:
                score = 2

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_INLINE_LUNGE_LEFT:
            # TODO：预测前后分腿蹲-左
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_INLINE_LUNGE_RIGHT:
            # TODO：预测前后分腿蹲-右
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SHOULDER_MOBILITY_LEFT:
            # TODO：预测肩部灵活性-左
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SHOULDER_MOBILITY_RIGHT:
            # TODO：预测肩部灵活性-右
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ACTIVE_STRAIGHT_LEG_RAISE_LEFT:
            # TODO：预测直腿主动上抬-左
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ACTIVE_STRAIGHT_LEG_RAISE_RIGHT:
            # TODO：预测直腿主动上抬-右
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_TRUNK_STABILITY_PUSH_UP:
            # TODO：预测躯干稳定俯卧撑
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ROTARY_STABILITY_QUADRUPED_LEFT:
            # TODO：预测躯干旋转稳定性-左
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ROTARY_STABILITY_QUADRUPED_RIGHT:
            # TODO：预测躯干旋转稳定性-右
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_NEXT_TIMES:
            # 接收到 进行下一次预测：
            # 如果当前预测的动作为None
            pass
        return score

