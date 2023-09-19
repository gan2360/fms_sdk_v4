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
            i = np.argmin(movements[:, 0, 2]) #关键帧的下标，屁股最低点
            key_frame = movements[i]
            if key_frame[0][2] > key_frame[2][2] or key_frame[0][2] > key_frame[8][2]: # 屁股比膝盖左右 任何一个膝盖高
                score = 1
            # elif key_frame[21][2] - key_frame[18][2] < 10 and key_frame[2][1] <= key_frame[3][1] and key_frame[8][1] >= key_frame[9][1]:
            elif key_frame[21][2] - key_frame[18][2] < 10 : # 屁股比膝盖低，而且双手的z坐标相差不超过10
                score = 3
            else:
                score = 2


        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_HURDLE_STEP_LEFT:
            # TODO：预测跨栏架步-左 1
            # p_18_x = movements[:, 18, 0]
            # p_21_x = movements[:, 21, 0]
            p_18_z = movements[:, 18, 2] # 手的z坐标
            p_21_z = movements[:, 21, 2] # 手的z坐标
            # x_dif = np.abs(p_21_x - p_18_x)
            z_dif = np.abs(p_21_z - p_18_z)
            hand_unsatisfy = np.sum(z_dif > 10) # 两手z坐标超过10定位不合格
            hand_unsatisfy_rate = hand_unsatisfy / len # 手臂不合格的比例
            p_13 = movements[:, 13, :] # 腰部位置
            p_13_distances = np.linalg.norm(p_13[1:] - p_13[:-1], axis=1) #判断腰部位置有没有发生很大的移动
            change_sum = np.sum(p_13_distances > 10) # 腰的位置移动超过10为不合格
            change_rate = change_sum / len  # 腰部位置不合格的比例
            p_1 = movements[:, 1, :] #左盆骨
            p_2 = movements[:, 2, :] #左膝盖
            p_3 = movements[:, 3, :] #左脚踝
            p_123 = np.stack((p_3, p_2, p_1), axis=1)
            normal_vectors = np.cross(p_123[:, 1] - p_123[:, 0], p_123[:, 2] - p_123[:, 0]) # 左边小腿和大腿形成平面构成的法向量
            magnitude_normal_vectors = np.linalg.norm(normal_vectors, axis=1) # 所有法向量的模长
            p_1 = movements[:, 1, :] # 左盆骨
            p_7 = movements[:, 7, :] # 右盆骨
            v_1_7 = p_1 - p_7 # 左右盆骨构成的向量
            magnitude_1_7 = np.linalg.norm(v_1_7, axis=1) # 盆骨向量的模长
            dot_product = np.sum(normal_vectors * v_1_7, axis=1) # 法向量和盆骨向量构成的点积
            angles = np.degrees(np.arccos(dot_product / (magnitude_1_7 * magnitude_normal_vectors))) # 法向量和盆骨向量的夹角
            offset_sum = np.sum(angles > 15) #夹角大于15度为不合格
            offset_rate = offset_sum / len # 臀部膝盖脚踝在矢状面不平齐的比例
            if hand_unsatisfy_rate <= 0.2 and change_rate <=0.2 and offset_rate <= 0.2: # 腰，手，矢状面 合格的比例都高于0.8，为3分
                score = 3
            elif hand_unsatisfy_rate > 0.6 or offset_rate > 0.6 or change_rate > 0.6: # 任意一个不合格的比例右0.6，给1
                score = 1
            else:
                score = 2

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_HURDLE_STEP_RIGHT:
            # TODO：预测跨栏架步-右 2
            # p_18_x = movements[:, 18, 0]
            # p_21_x = movements[:, 21, 0]
            p_18_z = movements[:, 18, 2] # 右手
            p_21_z = movements[:, 21, 2] # 左手
            # x_dif = np.abs(p_21_x - p_18_x)
            z_dif = np.abs(p_21_z - p_18_z) # 两手的z的差距
            hand_unsatisfy = np.sum(z_dif > 10) # 超过10为不合格
            hand_unsatisfy_rate = hand_unsatisfy / len # 手不合格的比例
            p_13 = movements[:, 13, :] # 腰
            p_13_distances = np.linalg.norm(p_13[1:] - p_13[:-1], axis=1)
            change_sum = np.sum(p_13_distances > 10)
            change_rate = change_sum / len # 腰变化过大的比例
            p_7 = movements[:, 7, :] # 右盆骨
            p_8 = movements[:, 8, :] # 右膝盖
            p_9 = movements[:, 9, :] # 右脚踝
            p_789 = np.stack((p_7, p_8, p_9), axis=1)
            normal_vectors = np.cross(p_789[:, 1] - p_789[:, 0], p_789[:, 2]-p_789[:, 0]) # 法向量
            magnitude_normal_vectors = np.linalg.norm(normal_vectors, axis=1) # 法向量的模长
            p_1 = movements[:, 1, :]
            p_7 = movements[:, 7, :]
            v_1_7 = p_1 - p_7
            magnitude_1_7 = np.linalg.norm(v_1_7, axis=1)
            dot_product = np.sum(normal_vectors * v_1_7, axis=1)
            angles = np.degrees(np.arccos(dot_product / (magnitude_1_7 * magnitude_normal_vectors)))
            offset_sum = np.sum(angles > 15)
            offset_rate = offset_sum / len # 法向量和盆骨向量夹角不合格的比例
            if hand_unsatisfy_rate <= 0.2 and change_rate <=0.2 and offset_rate <= 0.2:
                score = 3
            elif hand_unsatisfy_rate > 0.6 or offset_rate > 0.6 or change_rate > 0.6:
                score = 1
            else:
                score = 2

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_INLINE_LUNGE_RIGHT:
            # TODO：预测前后分腿蹲-右 3
            p_12 = movements[:, 12] # 前脚掌
            p_12_0 = p_12[0]
            p_12_half = p_12[len // 2]
            p_12_last = p_12[-1]
            p_12_dis_1 = np.linalg.norm(p_12_0 - p_12_half)
            p_12_dis_2 = np.linalg.norm(p_12_last - p_12_half)
            p_12_dis_min= min(p_12_dis_1, p_12_dis_2) # 序列前，中，后三个时刻右前脚掌的距离变化
            p_8_x = movements[:, 8, 0] # 右膝盖的x
            # p_2_x = movements[:, 2, 0]
            p_3_x = movements[:, 3, 0] #左脚踝的x
            # p_9_x = movements[:, 9, 0]
            x_dist = np.abs(p_8_x - p_3_x) # 右膝盖和左脚踝的x的差
            # z_dist = np.abs(p_9_x - p_2_x)
            below_offset_sum = np.sum(x_dist > 10) # 右膝盖和左脚踝距离差距过大的数量
            below_offset_rate = below_offset_sum / len # 膝盖和脚踝不合要求的占比
            p_21 = movements[:, 21] # 左手
            p_18 = movements[:, 18] # 右手
            hand_vectors = p_21 - p_18 # 手臂向量
            hand_norm_vectors = np.linalg.norm(hand_vectors, axis=1)
            target_vector = np.array([0, 0, 1]) # z轴单位向量
            target_magnitude = np.linalg.norm(target_vector)
            dot_product = np.sum(hand_vectors * target_vector, axis=1)
            angles_deg = np.degrees(np.arccos(dot_product / (hand_norm_vectors * target_magnitude))) # 手臂向量与z轴单位向量形成的夹角
            angles_deg_offset_sum = np.sum(angles_deg > 15)
            angles_deg_offset_rate = angles_deg_offset_sum / len # 手臂向量与z轴单位向量大于15度的比例
            if below_offset_rate <= 0.2 and angles_deg_offset_rate <= 0.2 and p_12_dis_min < 10:
                score = 3
            elif below_offset_rate > 0.6 or angles_deg_offset_rate > 0.6 or p_12_dis_min > 0.6:
                score = 1
            else:
                score = 2

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_INLINE_LUNGE_LEFT:
            # TODO：预测前后分腿蹲-左 4
            p_6 = movements[:, 6] # 左脚掌
            p_6_0 = p_6[0]
            p_6_half = p_6[len // 2]
            p_6_last = p_6[-1]
            p_6_dis_1 = np.linalg.norm(p_6_0 - p_6_half)
            p_6_dis_2 = np.linalg.norm(p_6_last - p_6_half)
            p_6_dis_min= min(p_6_dis_1, p_6_dis_2) # 序列中前中后，脚掌的距离
            p_2_x = movements[:, 2, 0] # 左膝盖
            # p_3_x = movements[:, 3, 0]
            p_9_x = movements[:, 9, 0] # 右脚踝
            # p_8_x = movements[:, 8, 0]
            x_dist = np.abs(p_2_x - p_9_x) #左膝盖和右脚踝x的差距
            # z_dist = np.abs(p_8_x - p_3_x)
            below_offset_sum = np.sum(x_dist > 10)
            below_offset_rate = below_offset_sum / len # 左膝盖和右脚踝x距离超过10的占比
            p_21 = movements[:, 21] # 左手
            p_18 = movements[:, 18] # 右手
            hand_vectors = p_21 - p_18 # 手臂向量
            hand_norm_vectors = np.linalg.norm(hand_vectors, axis=1)
            target_vector = np.array([0, 0, 1])
            target_magnitude = np.linalg.norm(target_vector)
            dot_product = np.sum(hand_vectors * target_vector, axis=1)
            angles_deg = np.degrees(np.arccos(dot_product / (hand_norm_vectors * target_magnitude)))
            angles_deg_offset_sum = np.sum(angles_deg > 15)
            angles_deg_offset_rate = angles_deg_offset_sum / len # 手臂向量与z轴单位向量夹角超过15的占比
            if below_offset_rate <= 0.2 and angles_deg_offset_rate <= 0.2 and p_6_dis_min < 10:
                score = 3
            elif below_offset_rate > 0.6 or angles_deg_offset_rate > 0.6 or p_6_dis_min > 15:
                score = 1
            else:
                score = 2
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SHOULDER_MOBILITY_LEFT:
            # TODO：预测肩部灵活性-左 5
            p_21_z = movements[:, 21, 2] #左手的z
            p_18_z = movements[:, 18, 2] #右手的z
            min_distance = np.min(np.abs(p_21_z - p_18_z)) #序列中，左手和右手z差距的最小值
            if min_distance < 10:
                score = 3
            elif min_distance < 15:
                score = 2
            else:
                score = 1

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_SHOULDER_MOBILITY_RIGHT:
            # TODO：预测肩部灵活性-右 6
            p_21_z = movements[:, 21, 2]
            p_18_z = movements[:, 18, 2]
            min_distance = np.min(np.abs(p_21_z - p_18_z))
            if min_distance < 10:
                score = 3
            elif min_distance < 15:
                score = 2
            else:
                score = 1


        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ACTIVE_STRAIGHT_LEG_RAISE_RIGHT:
            # TODO：预测直腿主动上抬-右 7
            key_frame_i = np.argmax(movements[:, 9, 2]) # 右脚踝最高时的下标
            key_frame = movements[key_frame_i] # 关键帧
            p_7 = key_frame[7] # 右盆骨
            p_9 = key_frame[9] # 右脚踝
            p_1 = key_frame[1] # 左盆骨
            p_3 = key_frame[3] # 左脚踝
            vector_leg_1 = p_9 - p_7 # 左腿形成的向量
            vector_leg_2 = p_3 - p_1 # 右腿形成的向量
            dot_product = np.dot(vector_leg_1, vector_leg_2) # 两腿向量的点积
            magnitude_v1 = np.linalg.norm(vector_leg_1)
            magnitude_v2 = np.linalg.norm(vector_leg_2)
            angle = np.degrees(np.arccos(dot_product / (magnitude_v1 * magnitude_v2))) # 两腿向量的夹角
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
            p_7 = key_frame[7] #
            p_9 = key_frame[9]
            p_1 = key_frame[1]
            p_3 = key_frame[3]
            vector_leg_1 = p_9 - p_7
            vector_leg_2 = p_3 - p_1
            dot_product = np.dot(vector_leg_1, vector_leg_2) # 两腿向量的点积
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
            key_frame_i = np.argmax(movements[:, 13, 2]) # 关键帧的索引，腰部位置最高处
            key_frame = movements[key_frame_i]
            p_13 = key_frame[13] # 腰
            p_3 = key_frame[3] #左脚踝
            p_9 = key_frame[9] #右脚踝
            p_39 = np.vstack((p_3, p_9)) # 堆叠3， 9
            p_mean = np.mean(p_39, axis=0) # 3，9点的平均值
            v_13_mean = p_13 - p_mean # 腰与两脚中间形成的向量
            normal_vector = np.array([0, 0, 1])
            dot_product = np.dot(v_13_mean, normal_vector)
            magnitude_1 = np.linalg.norm(v_13_mean)
            magnitude_2 = np.linalg.norm(dot_product)
            angle = 90 - np.degrees(np.arccos(dot_product / (magnitude_2 * magnitude_1))) # 腰与两脚形成的向量与x，y面形成的夹角
            if angle < 15:
                score = 1
            else:
                score = random.randint(2,3)
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ROTARY_STABILITY_QUADRUPED_RIGHT:
            # TODO：预测躯干旋转稳定性-右 10
            p_7 = None
            p_7 = movements[:, 7] # 右盆骨
            p_8 = movements[:, 8] # 右膝盖
            p_16 = movements[:, 16] # 右肩膀
            p_17 = movements[:, 17] # 右肘关节
            p_14 = movements[:, 14] # 脖子
            p_0 = movements[:, 0] # 屁股
            v_16_17= p_16 - p_17 # 右手臂向量
            v_7_8 = p_7 - p_8 # 右腿向量
            v_14_0 = p_14 - p_0 # 躯干向量
            magnitude_1_10 = np.linalg.norm(v_16_17, axis=1) # 右手臂模长
            magnitude_2_10 = np.linalg.norm(v_7_8, axis=1) # 右腿模长
            magnitude_3_10 = np.linalg.norm(v_14_0, axis=1) # 躯干模长
            dot_product_1 = np.sum(v_14_0 * v_16_17, axis=1) # 躯干和手的点积
            dot_product_2 = np.sum(v_7_8 * v_14_0,  axis=1) # 躯干和腿的点积
            angles_1 = np.degrees(np.arccos(dot_product_1 / (magnitude_3_10 * magnitude_1_10))) # 躯干和手形成的角度
            angles_2 = np.degrees(np.arccos(dot_product_2 / (magnitude_2_10 * magnitude_3_10))) # 躯干和腿形成的角度
            angles_2_abs = np.abs(angles_1 - 90)
            angles_1_abs = np.abs(angles_2 - 90)
            unsatisfy_sum = max(np.sum(angles_1_abs > 20), np.sum(angles_2_abs > 20)) # 手臂或角与躯干的夹角与90°的差距超过20的数量
            unsatisfy_rate = unsatisfy_sum / len
            if unsatisfy_rate < 0.15:
                score = 3
            elif unsatisfy_rate < 0.3:
                score = 2
            else:
                score = 1
            pass

        elif code == statusCode.TEST_PREDICTION_CONNECT_MOVEMENT_ROTARY_STABILITY_QUADRUPED_LEFT:
            # TODO：预测躯干旋转稳定性-左 11
            p_1 = movements[:, 1] # 左盆骨
            p_2 = movements[:, 2] # 左膝盖
            p_19 = movements[:, 19] # 左肩膀
            p_20= movements[:, 20] # 左手肘
            p_14 = movements[:, 14] # 脖子
            p_0 = movements[:, 0] # 屁股
            v_19_20= p_19 - p_20 # 手臂向量
            v_1_2 = p_1 - p_2 # 腿向量
            v_14_0 = p_14 - p_0 # 躯干向量
            magnitude_1_11 = np.linalg.norm(v_19_20, axis=1) # 手臂模长
            magnitude_2_11 = np.linalg.norm(v_1_2, axis=1) # 腿模长
            magnitude_3_11 = np.linalg.norm(v_14_0, axis=1) # 躯干模长
            dot_product_1 = np.sum(v_14_0 * v_19_20, axis=1) # 手臂躯干点积
            dot_product_2 = np.sum(v_1_2 * v_14_0,  axis=1) # 腿，躯干点积
            angles_1 = np.degrees(np.arccos(dot_product_1 / (magnitude_3_11 * magnitude_1_11))) # 躯干手臂夹角
            angles_2 = np.degrees(np.arccos(dot_product_2 / (magnitude_2_11 * magnitude_3_11))) # 躯干腿夹角
            angles_2_abs = np.abs(angles_1 - 90)
            angles_1_abs = np.abs(angles_2 - 90)
            unsatisfy_sum = max(np.sum(angles_1_abs > 20 ), np.sum(angles_2_abs > 20)) # 同上
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
    pass


