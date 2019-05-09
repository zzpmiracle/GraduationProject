# -*- coding: utf-8 -*-
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from keras.preprocessing import image
from keras.models import load_model
# 获取音频数据
data, samplerate = sf.read('./cr_20181025-161858.wav')
data = data.reshape((-1, 2))
data = data.mean(axis=1)
len_data = len(data)

l = 0
Fs = 2000

DenseNet_file_path = './trained_models/denseNet96.2.hdf5'
img_width, img_height = 32, 32
DenseNet_model = load_model(DenseNet_file_path)
temp_file = 'temp.jpg'

#15s
l_length = 15*Fs
begin_time_min ,begin_time_sec = 0 , 0
conti_flag = False
while l < len_data-l_length:
    batch_data = data[l:l+l_length]
    p, q, j, k = plt.specgram(batch_data, NFFT=512, Fs=Fs, noverlap=380, cmap='jet')  # 绘制时谱图
    p = 10 * np.log10(p)
    p = p[:224][::-1]
    # plt.imsave('img/{}_{}to{}_{}.png'.format(begin_time_min, begin_time_sec, end_time_min, end_time_sec), p, cmap='jet')
    plt.imsave(temp_file, p, cmap='jet')
    img = image.load_img(temp_file, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x *= 1. / 255
    x = x.reshape([-1, 32, 32, 3])
    result = DenseNet_model.predict(x)
    if result<0.5:
        if conti_flag:
                pass
        else:
            begin_time_min = int(np.floor(l / Fs / 60))
            begin_time_sec = int(l / Fs % 60)
            conti_flag = True
    else:
        if conti_flag:
            end_time_min = int(np.floor((l) / Fs / 60))
            end_time_sec = int((l) / Fs % 60)
            conti_flag = False
            print('{}:{} to {}:{} happened an event!'.format(begin_time_min, begin_time_sec, end_time_min, end_time_sec))
        else:
            pass
    l = l + l_length
    # begin_time_min = int(np.floor(l / Fs / 60))
    # begin_time_sec = int(l / Fs % 60)
    # end_time_min = int(np.floor((l + l_length) / Fs / 60))
    # end_time_sec = int((l + l_length) / Fs % 60)



