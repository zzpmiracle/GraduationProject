import os
from DenseNet.densenet import DenseNet
from ResNet.ResNet import resnet_v2

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.utils import plot_model
from matplotlib import pyplot as plt
import math
from keras.preprocessing.image import ImageDataGenerator

from numpy.random import seed

img_width, img_height = 64,64
image_dim = (img_width,img_height, 3)

DenseNet_model =DenseNet(image_dim,
                             classes=1,
                         depth=28,
                             activation='sigmoid',
                             )
DenseNet_model.compile(loss='binary_crossentropy',
                       optimizer='Adadelta',
                       metrics=['accuracy'])


DenseNet_model.summary()
DenseNet_model.get_config()

# if os.path.exists(ResNet_file_path):
#     ResNet_model = load_model(ResNet_file_path)
# else:
#     ResNet_model = resnet_v2(depth=29,
#                              num_classes=1,
#                              input_shape=image_dim)
# ResNet_model.compile(loss='binary_crossentropy',
#                      optimizer='Adadelta',
#                      metrics=['accuracy'])
# # plot_model(ResNet_model, to_file='ResNet_model.png',show_layer_names=False,show_shapes=True)
# ResNet_check_point = ModelCheckpoint(filepath=ResNet_file_path,
#                               monitor='val_acc',
#                               verbose=1,
#                               save_best_only='True',
#                               mode='max')
# ResNet_history = ResNet_model.fit_generator(train_data_generator,
#                                             steps_per_epoch=math.ceil(nb_train_samples/batch_size),
#                                             epochs=epochs,
#                                             verbose=2,
#                                             callbacks=[ResNet_check_point],
#                                             validation_data=val_data_generator,
#                                             validation_steps=math.ceil(nb_val_samples/batch_size))
#
#
#
# plt.figure()
# plt.plot(ResNet_history.history['acc'],label='ResNet_acc',color='g')
# plt.plot(ResNet_history.history['val_acc'],label='ResNet_val_acc',color='r')
# plt.plot(ResNet_history.history['loss'],label='ResNet_loss',color='g', linestyle='--')
# plt.plot(ResNet_history.history['val_loss'],label='ResNet_val_loss',color='r',linestyle='--')
# plt.title('ResNet Model')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='best')
# plt.show()
#
# ResNet_score = ResNet_model.evaluate_generator(test_data_generator,steps=math.ceil(nb_test_samples/batch_size))
# print(ResNet_score[-1])