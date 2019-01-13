# -*- coding:utf-8 -*-
import importlib
import sys
import socket

importlib.reload(sys)
sys.path.append("../")

hostname = socket.gethostname()
if hostname == "precision":  # linux 服务器
    CWD = "/home/dejian/my/end-to-end"
    data_path = '/home/dejian/my/end-to-end/data/exp/'
else:  # windows
    CWD = "D:/PY/Pycode/project/learn/learn_tensorflow/end-to-end"
    data_path = 'D:/PY/Pycode/project/learn/learn_tensorflow/end-to-end/data/exp/'

DATA_PATH = CWD + "/data"
RECORD_PATH = CWD + "/record"