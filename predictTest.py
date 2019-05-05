import numpy as np
from keras.models import load_model
import math
from keras.preprocessing import image
DenseNet_file_path = './trained_models/denseNet96.2.hdf5'
img_width, img_height = 32, 32
image_dim = (img_width, img_height, 3)
DenseNet_model = load_model(DenseNet_file_path)
DenseNet_model.compile(loss='binary_crossentropy',
                       optimizer='Adadelta',
                       metrics=['accuracy'])
file = 'D:\Event&NoEvent\\test\Even_spec_224\\13_1405874704.0.png'

img = image.load_img(file,target_size=(img_width, img_height))
x = image.img_to_array(img)
x *= 1./255
result = DenseNet_model.predict(x,batch_size=1)
print(result)