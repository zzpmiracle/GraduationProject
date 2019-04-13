import os
from DenseNet.densenet import DenseNet
from ResNet import resnet_v2

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.utils import plot_model
from matplotlib import pyplot as plt
import math
from keras.preprocessing.image import ImageDataGenerator

from numpy.random import seed
seed(2)

from tensorflow import set_random_seed
set_random_seed(2)

train_image_path = 'D:\\Event&NoEvent\\train'
val_image_path = 'D:\\Event&NoEvent\\validation'
test_image_path = 'D:\\Event&NoEvent\\test'

nb_train_samples = 4000
nb_val_samples = 500
nb_test_samples = 500

img_width, img_height = 32, 32
image_dim = (img_width,img_height, 3)

batch_size = 64
epochs = 10

# train_data_gen = ImageDataGenerator(rescale=1./255)
train_data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=train_image_path,
                                                          target_size=(img_width,img_height),
                                                          class_mode='binary',
                                                          batch_size=batch_size)
val_data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=val_image_path,
                            target_size=(img_width,img_height),
                            class_mode='binary',
                            batch_size=batch_size)
test_data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=test_image_path,
                            target_size=(img_width,img_height),
                            class_mode='binary',
                            batch_size=batch_size)

DenseNet_file_path = 'denseNet.hdf5'
ResNet_file_path = 'ResNet.hdf5'


if os.path.exists(DenseNet_file_path):
    DenseNet_model = load_model(DenseNet_file_path)
else:
    DenseNet_model =DenseNet(image_dim,
                             classes=1,
                             depth=28,
                             activation='sigmoid',
                             )
DenseNet_model.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

# plot_model(DenseNet_model, to_file='denseNet_model.png')
DenseNet_check_point = ModelCheckpoint(filepath=DenseNet_file_path,
                              monitor='val_acc',
                              verbose=1,
                              save_best_only='True',
                              mode='max')
DenseNet_history = DenseNet_model.fit_generator(train_data_generator,
                                                steps_per_epoch=math.ceil(nb_train_samples/batch_size),
                                                epochs=epochs,
                                                verbose=2,
                                                callbacks=[DenseNet_check_point],
                                                validation_data=val_data_generator,
                                                validation_steps=math.ceil(nb_val_samples / batch_size)
                                                )


if os.path.exists(ResNet_file_path):
    ResNet_model = load_model(ResNet_file_path)
else:
    ResNet_model = resnet_v2(depth=20,
                             num_classes=1,
                             input_shape=image_dim)
ResNet_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
# plot_model(ResNet_model, to_file='ResNet_model.png')
ResNet_check_point = ModelCheckpoint(filepath=ResNet_file_path,
                              monitor='val_acc',
                              verbose=1,
                              save_best_only='True',
                              mode='max')
ResNet_history = ResNet_model.fit_generator(train_data_generator,
                                            steps_per_epoch=math.ceil(nb_train_samples/batch_size),
                                            epochs=epochs,
                                            verbose=2,
                                            callbacks=[ResNet_check_point],
                                            validation_data=val_data_generator,
                                            validation_steps=math.ceil(nb_val_samples/batch_size))



# 绘制训练 & 验证的准确率值
plt.figure()
plt.plot(DenseNet_history.history['acc'],label='DenseNet',color='g')
plt.plot(ResNet_history.history['acc'],label='ResNet',color='r')

# plt.plot(history.history['loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
