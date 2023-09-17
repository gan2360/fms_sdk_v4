"""
@Project ：fms_predict_sdk_v4
@File    ：score.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/9/12 9:15
@Des     ：
"""
import random

# from server.server import FMSMovement
from utils import statusCode
import numpy as np


class ScoreOp:
    def __init__(self):
        gender = None
        palm = None
        pass



    def parse_code_get_score(self, code, movements):
        len = movements.shape[0]
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
            elif abs(key_frame[21][0] - key_frame[18][0]) < 2 and key_frame[21][2] - key_frame[18][2]<2 and key_frame[2][1] <= key_frame[3][1] and key_frame[8][1] >= key_frame[9][1]:
                score = 3
            else:
                score = 2


        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_HURDLE_STEP_LEFT:
            # TODO：预测跨栏架步-左 1
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
            target_vector = np.divide(target_vector, np.linalg.norm(target_vector))
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
            # TODO：预测跨栏架步-右 2
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
            normal_vectors = np.cross(p_123[:, 1] - p_123[:, 0], p_123[:, 2]-p_123[:, 0]) # 法向量
            target_vector = np.array([0, 1, 0])
            target_vector = np.divide(target_vector, np.linalg.norm(target_vector))
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
            # TODO：预测前后分腿蹲-左 3
            p_12 = movements[:, 12] # 前脚掌
            p_12_0 = p_12[0]
            p_12_half = p_12[len // 2]
            p_12_last = p_12[-1]
            p_12_dis_1 = np.linalg.norm(p_12_0 - p_12_half)
            p_12_dis_2 = np.linalg.norm(p_12_last - p_12_half)
            p_12_dis_min= min(p_12_dis_1, p_12_dis_2)
            p_2_y = movements[:, 2, 1]
            # p_2_x = movements[:, 2, 0]
            p_9_y = movements[:, 9, 1]
            # p_9_x = movements[:, 9, 0]
            y_dist = np.abs(p_9_y - p_2_y)
            # z_dist = np.abs(p_9_x - p_2_x)
            below_offset_sum = max(np.sum(y_dist > 3))
            below_offset_rate = below_offset_sum / len
            p_21 = movements[:, 21]
            p_18 = movements[:, 18]
            hand_vectors = p_21 - p_18 # 手臂向量
            hand_norm_vectors = np.linalg.norm(hand_vectors, axis=1)
            target_vector = np.array([0, 0, 1])
            dot_product = np.sum(hand_vectors * target_vector, axis=1)
            angles_deg = np.degrees(np.arccos(dot_product / hand_norm_vectors))
            angles_deg_offset_sum = np.sum(angles_deg > 5)
            angles_deg_offset_rate = angles_deg_offset_sum / len
            if below_offset_rate <= 0.2 and angles_deg_offset_rate <= 0.2 and p_12_dis_min < 3:
                score = 3
            elif below_offset_rate > 0.5 or angles_deg_offset_rate > 0.5 or p_12_dis_min > 5:
                score = 1
            else:
                score = 2

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_INLINE_LUNGE_RIGHT:
            # TODO：预测前后分腿蹲-右 4
            p_6 = movements[:, 6] # 前脚掌
            p_6_0 = p_6[0]
            p_6_half = p_6[len // 2]
            p_6_last = p_6[-1]
            p_6_dis_1 = np.linalg.norm(p_6_0 - p_6_half)
            p_6_dis_2 = np.linalg.norm(p_6_last - p_6_half)
            p_6_dis_min= min(p_6_dis_1, p_6_dis_2)
            p_3_y = movements[:, 3, 1]
            # p_3_x = movements[:, 3, 0]
            p_8_y = movements[:, 8, 1]
            # p_8_x = movements[:, 8, 0]
            y_dist = np.abs(p_8_y - p_3_y)
            # z_dist = np.abs(p_8_x - p_3_x)
            below_offset_sum = max(np.sum(y_dist > 3))
            below_offset_rate = below_offset_sum / len
            p_21 = movements[:, 21]
            p_18 = movements[:, 18]
            hand_vectors = p_21 - p_18 # 手臂向量
            hand_norm_vectors = np.linalg.norm(hand_vectors, axis=1)
            target_vector = np.array([0, 0, 1])
            dot_product = np.sum(hand_vectors * target_vector, axis=1)
            angles_deg = np.degrees(np.arccos(dot_product / hand_norm_vectors))
            angles_deg_offset_sum = np.sum(angles_deg > 5)
            angles_deg_offset_rate = angles_deg_offset_sum / len
            if below_offset_rate <= 0.2 and angles_deg_offset_rate <= 0.2 and p_6_dis_min < 3:
                score = 3
            elif below_offset_rate > 0.5 or angles_deg_offset_rate > 0.5 or p_6_dis_min > 8:
                score = 1
            else:
                score = 2
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SHOULDER_MOBILITY_LEFT:
            # TODO：预测肩部灵活性-左 5
            p_21_z = movements[:, 21, 2]
            p_18_z = movements[:, 18, 2]
            min_distance = np.min(np.abs(p_21_z - p_18_z))
            if min_distance < 8:
                score = 3
            elif min_distance < 12:
                score = 2
            else:
                score = 1

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SHOULDER_MOBILITY_RIGHT:
            # TODO：预测肩部灵活性-右 6
            p_21_z = movements[:, 21, 2]
            p_18_z = movements[:, 18, 2]
            min_distance = np.min(np.abs(p_21_z - p_18_z))
            if min_distance < 8:
                score = 3
            elif min_distance < 12:
                score = 2
            else:
                score = 1


        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ACTIVE_STRAIGHT_LEG_RAISE_LEFT:
            # TODO：预测直腿主动上抬-左 7
            key_frame_i = np.argmax(movements[:, 9, 2])
            key_frame = movements[key_frame_i]
            p_7 = key_frame[7]
            p_9 = key_frame[9]
            p_1 = key_frame[1]
            p_3 = key_frame[3]
            vector_leg_1 = p_9 - p_7
            vector_leg_2 = p_1 - p_3
            dot_product = vector_leg_2.dot(vector_leg_1)
            magnitude_v1 = np.linalg.norm(vector_leg_1)
            magnitude_v2 = np.linalg.norm(vector_leg_2)
            angle = np.degrees(np.arccos(dot_product / (magnitude_v1 * magnitude_v2)))
            if abs(angle) > 83:
                score = 3
            elif abs(angle) > 75:
                score = 2
            else:
                score = 1

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ACTIVE_STRAIGHT_LEG_RAISE_RIGHT:
            # TODO：预测直腿主动上抬-右 8
            key_frame_i = np.argmax(movements[:, 9, 2])
            key_frame = movements[key_frame_i]
            p_7 = None
            p_7 = key_frame[7]
            p_9 = key_frame[9]
            p_1 = key_frame[1]
            p_3 = key_frame[3]
            vector_leg_1 = p_9 - p_7
            vector_leg_2 = p_1 - p_3
            dot_product = vector_leg_2.dot(vector_leg_1)
            magnitude_v1 = np.linalg.norm(vector_leg_1)
            magnitude_v2 = np.linalg.norm(vector_leg_2)
            angle = np.degrees(np.arccos(dot_product / (magnitude_v1 * magnitude_v2)))
            if abs(angle) > 83:
                score = 3
            elif abs(angle) > 75:
                score = 2
            else:
                score = 1

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_TRUNK_STABILITY_PUSH_UP:
            # TODO：预测躯干稳定俯卧撑 9
            key_frame_i = np.argmax(movements[:, 13, 2])
            key_frame = movements[key_frame_i]
            p_13 = key_frame[13]
            p_3 = key_frame[3]
            p_9 = key_frame[9]
            p_39 = np.vstack((p_3, p_9))
            p_mean = np.mean(p_39, axis=0)
            v_13_mean = p_13 - p_mean
            normal_vector = np.array([0, 0, 1])
            dot_product = np.dot(v_13_mean, normal_vector)
            magnitude_1 = np.linalg.norm(v_13_mean)
            magnitude_2 = np.linalg.norm(dot_product)
            angle = 90 - np.degrees(np.arccos(dot_product / (magnitude_2 * magnitude_1)))
            if angle < 15:
                score = 1
            else:
                score = random.randint(2,3)
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ROTARY_STABILITY_QUADRUPED_LEFT:
            # TODO：预测躯干旋转稳定性-左 10
            p_7 = None
            p_7 = movements[:, 7]
            p_8 = movements[:, 8]
            p_16 = movements[:, 16]
            p_17 = movements[:, 17]
            p_14 = movements[:, 14]
            p_0 = movements[:, 0]
            v_16_17= p_16 - p_17
            v_7_8 = p_7 - p_8
            v_14_0 = p_14 - p_0
            magnitude_1_10 = np.linalg.norm(v_16_17, axis=1)
            magnitude_2_10 = np.linalg.norm(v_7_8, axis=1)
            magnitude_3_10 = np.linalg.norm(v_14_0, axis=1)
            dot_product_1 = np.sum(v_14_0 * v_16_17, axis=1)
            dot_product_2 = np.sum(v_7_8 * v_14_0,  axis=1)
            angles_1 = np.degrees(np.arccos(dot_product_1 / (magnitude_3_10 * magnitude_1_10)))
            angles_2 = np.degrees(np.arccos(dot_product_2 / (magnitude_2_10 * magnitude_3_10)))
            angles_2_abs = np.abs(angles_1)
            angles_1_abs = np.abs(angles_2)
            unsatisfy_sum = max(np.sum(angles_1_abs > 15), np.sum(angles_2_abs > 15))
            unsatisfy_rate = unsatisfy_sum / len
            if unsatisfy_rate < 0.15:
                score = 3
            elif unsatisfy_rate < 0.3:
                score = 2
            else:
                score = 1
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ROTARY_STABILITY_QUADRUPED_RIGHT:
            # TODO：预测躯干旋转稳定性-右 11
            p_1 = movements[:, 1]
            p_2 = movements[:, 2]
            p_19 = movements[:, 19]
            p_20= movements[:, 20]
            p_14 = movements[:, 14]
            p_0 = movements[:, 0]
            v_19_20= p_19 - p_20
            v_1_2 = p_1 - p_2
            v_14_0 = p_14 - p_0
            magnitude_1_11 = np.linalg.norm(v_19_20, axis=1)
            magnitude_2_11 = np.linalg.norm(v_1_2, axis=1)
            magnitude_3_11 = np.linalg.norm(v_14_0, axis=1)
            dot_product_1 = np.sum(v_14_0 * v_19_20, axis=1)
            dot_product_2 = np.sum(v_1_2 * v_14_0,  axis=1)
            angles_1 = np.degrees(np.arccos(dot_product_1 / (magnitude_3_11 * magnitude_1_11)))
            angles_2 = np.degrees(np.arccos(dot_product_2 / (magnitude_2_11 * magnitude_3_11)))
            angles_2_abs = np.abs(angles_1)
            angles_1_abs = np.abs(angles_2)
            unsatisfy_sum = max(np.sum(angles_1_abs > 15), np.sum(angles_2_abs > 15))
            unsatisfy_rate = unsatisfy_sum / len
            if unsatisfy_rate < 0.15:
                score = 3
            elif unsatisfy_rate < 0.3:
                score = 2
            else:
                score = 1


        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_NEXT_TIMES:
            # 接收到 进行下一次预测：
            # 如果当前预测的动作为None
            pass
        return score


if __name__ == '__main__':
    import numpy

    x = numpy.array([[[1, 2], [3, 4]],
                     [[4, 3], [2, 1]]])
    print(numpy.max(x, axis=0))


