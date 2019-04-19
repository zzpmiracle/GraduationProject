import numpy as np
from keras.models import load_model

from keras.utils import plot_model
from matplotlib import pyplot as plt
import math
from keras.preprocessing.image import ImageDataGenerator
DenseNet_file_path = './trained_models/denseNet96.2.hdf5'
test_image_path = 'D:\\Event&NoEvent\\test'
nb_train_samples = 4000
nb_val_samples = 500
nb_test_samples = 500

img_width, img_height = 32,32
image_dim = (img_width,img_height, 3)

batch_size = 64
epochs = 100
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

result = result[:]<0.5

answer = [[True] for i in range(250)]
answer.extend([[False] for i in range(250)])

acc = float(np.sum(result==answer)/500)
print('%.1f%%' % (acc * 100))
