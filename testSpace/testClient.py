"""
@Project ：fms_predict_sdk
@File    ：testClient.py
@IDE     ：PyCharm
@Author  ：Tristen
@Date    ：2023/05/05 19:20
@Des     ：本地测试main文件进行通信的测试代码，在开始之前确定下面的IP部分
"""
from datetime import datetime
import json
import socket
import struct
import cv2
import numpy as np
import matplotlib.pyplot as plt

BODY_22_color = np.array(
    [[239, 43, 3], [3, 216, 239], [3, 216, 239], [3, 216, 239], [239, 219, 3], [239, 219, 3], [239, 219, 3]
        , [239, 3, 217], [239, 3, 217], [239, 3, 217], [37, 239, 3], [37, 239, 3], [37, 239, 3], [239, 43, 3]
        , [239, 43, 3], [239, 118, 3], [3, 239, 173], [3, 239, 173], [3, 239, 173], [125, 3, 239], [125, 3, 239]
        , [125, 3, 239]])

BODY_22_pairs = np.array(
    [[14, 13], [13, 0], [14, 19], [14, 16], [19, 20], [20, 21], [16, 17], [17, 18], [0, 1], [1, 2], [2, 3], [0, 7],
     [7, 8], [8, 9], [14, 15], [9, 11], [11, 12], [9, 10],
     [3, 5], [5, 6], [3, 4]])


def plotKeypoint(fig, keypoints_pred):
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_zlim(0, 200)
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)
    ax.view_init(20, -70)
    xs = keypoints_pred[:, 0]
    ys = keypoints_pred[:, 1]
    zs = keypoints_pred[:, 2]

    for i in range(BODY_22_pairs.shape[0]):
        index_1 = BODY_22_pairs[i, 0]
        index_2 = BODY_22_pairs[i, 1]
        xs_line = [xs[index_1], xs[index_2]]
        ys_line = [ys[index_1], ys[index_2]]
        zs_line = [zs[index_1], zs[index_2]]
        ax.plot(xs_line, ys_line, zs_line, color=BODY_22_color[i] / 255.0)

    ax.scatter(xs, ys, zs, s=50, c=BODY_22_color[:22] / 255.0)
    # for x, y, z in zip(xs, ys, zs):
    #     label = f'({z:.2f})'
    #     ax.text(x, y, z, label, color='black', fontsize=5, )
    fig.canvas.draw()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]
    return img


def build_header(data_len):
    header = {'length': data_len}
    return header


header_size = struct.calcsize("!I")

if __name__ == '__main__':
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 检查这里~
    # conn.connect(("192.168.10.110", 5001))
    conn.connect(("127.0.0.1", 5001))
    conn.setblocking(False)
    index = 0
    data = {"code": 2100001}
    head_bytes = bytes(json.dumps(data), encoding='utf-8')  # 序列化并转成bytes,用于传输
    head_len_bytes = struct.pack('!I', len(head_bytes))  # 这4个字节里只包含了一个数字,该数字是报头的长度
    conn.sendall(head_len_bytes + head_bytes)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], data)
    aaa = True
    receive = b''
    video_output_path = './vis'
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(video_output_path + '.avi', fourcc, 10, (640, 480))
    fig = plt.figure()
    while True:
        try:
            data = conn.recv(1024)
            if data:
                receive += data
                # 如果已经接收到了固定长度的帧头，那么就解析，获取消息体的长度
                if len(receive) >= header_size:
                    length = struct.unpack("!I", receive[:header_size])[0]
                    # 如果已经收到了完整的消息体，那么就去处理
                    if len(receive[header_size:]) >= length:
                        packet = receive[header_size:header_size + length]
                        receive = receive[header_size + length:]
                        try:
                            request = json.loads(packet)
                            if request['code'] == 2000001:
                                p = {"code": 2100002}
                                head_bytes = bytes(json.dumps(p), encoding='utf-8')  # 序列化并转成bytes,用于传输
                                head_len_bytes = struct.pack('!I', len(head_bytes))  # 这4个字节里只包含了一个数字,该数字是报头的长度
                                conn.sendall(head_len_bytes + head_bytes)
                                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], p)
                            if index >= 10000 and aaa:
                                aaa = False
                                p = {"code": 2100010}
                                head_bytes = bytes(json.dumps(p), encoding='utf-8')  # 序列化并转成bytes,用于传输
                                head_len_bytes = struct.pack('!I', len(head_bytes))  # 这4个字节里只包含了一个数字,该数字是报头的长度
                                conn.sendall(head_len_bytes + head_bytes)
                                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], p)
                            if request['code'] == 2000002:
                                p = {"code": 21000100}
                                head_bytes = bytes(json.dumps(p), encoding='utf-8')  # 序列化并转成bytes,用于传输
                                head_len_bytes = struct.pack('!I', len(head_bytes))  # 这4个字节里只包含了一个数字,该数字是报头的长度
                                conn.sendall(head_len_bytes + head_bytes)
                                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], p)
                            if request['code'] == 2000003:
                                index += 1
                                keypoint11111 = np.array(request["data"]).reshape((22, 3))
                                print(keypoint11111.shape)
                                img = plotKeypoint(fig, keypoint11111)
                                # cv2.imshow('Local Camera', img)
                                out.write(img)
                                print(index)
                                if index == 30:
                                    out.release()
                                    print("finish")
                                    break
                            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], request)
                        except json.decoder.JSONDecodeError:
                            print("json error")
            else:
                print("========================= CLIENT CONNECTION IS CLOSED =========================")
                conn.close()
        except socket.error as e:
            pass
        except Exception as e:
            print(e)
            pass
