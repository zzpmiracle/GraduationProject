# -*- coding: utf-8 -*-
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


model_xml = "/home/haoxuewu/PycharmProjects/classifier_new/audio_test/model/model.xml"
model_bin = "/home/haoxuewu/PycharmProjects/classifier_new/audio_test/model/model.bin"

# 获取音频数据
data, samplerate = sf.read('D:\Event&NoEvent\cr_20181025-161858.wav')
data = data.reshape((-1, 2))
data = data.mean(axis=1)
len_data = len(data)
# Plugin initialization for specified device and load extensions library if specified

# Read IR
print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
net = IENetwork(model=model_xml, weights=model_bin)

# Loading model to the plugin
print("Loading model to the plugin ...")
exec_net = plugin.load(network=net)

# assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
# assert len(net.outputs) == 1, "Sample supports only single output topologies"

print("Preparing input blobs ...")
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

# Read and pre-process input images
n, c, h, w = net.inputs[input_blob].shape
image = np.ndarray(shape=(n, c, h, w))
l = 0
t0 = time()
while l < len_data-30000:
        batch_data = data[l:l+29999]
        p, q, j, k = plt.specgram(batch_data, NFFT=512, Fs=2000., noverlap=380, cmap='jet')  # 绘制时谱图
        p = 10 * np.log10(p)
        p = p[:224][::-1]
        plt.imsave('/home/haoxuewu/PycharmProjects/classifier_new/audio_test/temp.png', p, cmap='jet')
        image = cv2.imread('/home/haoxuewu/PycharmProjects/classifier_new/audio_test/temp.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if image.shape[:-1] != (h, w):
        #     print("Image {} is resized from {} to {}".format(image_path, image.shape[:-1], (h, w)))
        #     image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

        # Start sync inference
        res = exec_net.infer(inputs={input_blob: image})
        if res['ArgMax/Squeeze'] == [1.0]:
            begin_time_min = int(np.floor(l/2000/60))
            begin_time_sec = int(l/2000%60)
            end_time_min = int(np.floor((l+30000)/2000/60))
            end_time_sec = int((l+30000)/2000%60)
            l = l + 30000
            print('{}:{} to {}:{} happened an event!'.format(begin_time_min, begin_time_sec, end_time_min, end_time_sec))
        else:
            l = l + 2000
print('Done with the running time: {}s'.format(time()-t0))



