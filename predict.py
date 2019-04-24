import re

import numpy as np
from keras.models import load_model
import os
from keras.utils import plot_model
from matplotlib import pyplot as plt
import math
from keras.preprocessing.image import ImageDataGenerator
DenseNet_file_path = './trained_models/denseNet96.2.hdf5'
test_image_path = '.\wav'
nb_train_samples = 4000
nb_val_samples = 500
nb_test_samples = 151

img_width, img_height = 32,32
image_dim = (img_width,img_height, 3)
name_list=list(name[:-4] for name in os.listdir('.\wav\imgs'))
name_list.sort(key=lambda x:int(x))
print(name_list)
batch_size = 64
test_data_generator = ImageDataGenerator(rescale=1./255).\
    flow_from_directory(directory=test_image_path,
                        target_size=(img_width,img_height),
                        class_mode='binary',
                        batch_size=batch_size,
                        shuffle=False)
DenseNet_model = load_model(DenseNet_file_path)
DenseNet_model.compile(loss='binary_crossentropy',
                       optimizer='Adadelta',
                       metrics=['accuracy'])
result = DenseNet_model.predict_generator(test_data_generator,steps=math.ceil(nb_test_samples/batch_size))


for i in range(len(result)):
    if result[i] < 0.5:
        l = int(name_list[i])
        begin_time_min = int(np.floor(l / 2000 / 60))
        begin_time_sec = int(l / 2000 % 60)
        end_time_min = int(np.floor((l + 30000) / 2000 / 60))
        end_time_sec = int((l + 30000) / 2000 % 60)
        l = l + 30000
        print('{}:{} to {}:{} happened an event!'.format(begin_time_min, begin_time_sec, end_time_min, end_time_sec))
result = result[:] < 0.5
print(result)