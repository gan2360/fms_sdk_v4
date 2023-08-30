"""
@Project ：fms_predict_sdk_v2
@File    ：baseClient.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/14 12:24
@Des     ：
"""
import logging
import socket
import sys


class SocketClient:
    socket_client = None
    host = None
    port = None

    def __init__(self, host, port):
        self.host = host
        self.port = port
        # AF_UNIX： 用于同一台机器进程间通信
        try:
            self.socket_client = socket.socket()
        except socket.error as msg:
            logging.error('Failed to create socket. Error code: ' + str(msg[0]) + ' , Error message : ' + msg[1])
            sys.exit()
        logging.info("Client is created")
        print("Client is created")

    def connect(self):
        self.socket_client.connect((self.host, self.port))
        # 线程阻塞被取消
        logging.info('Socket Connected to ' + str(self.port) + ' on ip ' + str(self.host))
        print('Socket Connected to ' + str(self.port) + ' on ip ' + str(self.host))

    def send(self, data):
        try:
            self.socket_client.sendall(bytes.fromhex(data))
        except socket.error as msg:
            logging.error(
                'Failed to send ' + data + 'to server2' + '. Error: ' + str(msg)
            )

    def receive(self):
        try:
            reply = self.socket_client.recv(4096)
            if reply.hex()[0:4] == 'aabb' and reply.hex()[-2:] == 'ff':
                data = reply.hex()[14: -4]
                code = reply[2]
                return {"code": code, "data": data, "reply": reply, 'finish': True}
            elif reply.hex()[0:4] == 'aabb' and reply.hex()[-2:] != 'ff':
                data = reply.hex()[14:]
                code = reply[2]
                return {"code": code, "data": data, "reply": reply, 'finish': False}
            elif reply.hex()[0:4] != 'aabb' and reply.hex()[-2:] != 'ff':
                data = reply.hex()
                return {"code": -1, "data": data, "reply": reply, 'finish': False}
            elif reply.hex()[0:4] != 'aabb' and reply.hex()[-2:] == 'ff':
                data = reply.hex()[:-4]
                return {"code": -1, "data": data, "reply": reply, 'finish': True}
        except socket.error:
            pass

    def close(self):
        self.socket_client.close()
        logging.info('Client is closed')
