# -*- coding: utf-8 -*-
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from keras.preprocessing import image

# 获取音频数据
data, samplerate = sf.read('cr_20181025-161858.wav')
data = data.reshape((-1, 2))
data = data.mean(axis=1)
len_data = len(data)
# Plugin initialization for specified device and load extensions library if specified



# assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
# assert len(net.outputs) == 1, "Sample supports only single output topologies"

# print("Preparing input blobs ...")
# input_blob = next(iter(net.inputs))
# out_blob = next(iter(net.outputs))
#
# # Read and pre-process input images
# # n, c, h, w = net.inputs[input_blob].shape
# image = np.ndarray(shape=(n, c, h, w))
l = 0
t0 = time()

l_length = 1*2000
while l < len_data-30000:
        batch_data = data[l:l+29999]
        p, q, j, k = plt.specgram(batch_data, NFFT=512, Fs=2000., noverlap=380, cmap='jet')  # 绘制时谱图
        p = 10 * np.log10(p)
        p = p[:224][::-1]
        begin_time_min = int(np.floor(l / 2000 / 60))
        begin_time_sec = int(l / 2000 % 60)
        end_time_min = int(np.floor((l + 30000) / 2000 / 60))
        end_time_sec = int((l + 30000) / 2000 % 60)
        # plt.imsave('img/{}_{}to{}_{}.png'.format(begin_time_min, begin_time_sec, end_time_min, end_time_sec), p, cmap='jet')
        plt.imsave('imgs/'+str(l)+'.png', p, cmap='jet')
        l = l + 30000



