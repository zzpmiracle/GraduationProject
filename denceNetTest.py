import os

from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.models import save_model,load_model

import densenet
from keras.preprocessing.image import ImageDataGenerator

from densenet_fast import create_dense_net

imagePath = 'D:\Event&NoEvent'
img_width,img_height = 32,32
batch_size = 32
nb_train_samples = 2000
epochs = 1


datagen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.3
                             )

data_generator = datagen.flow_from_directory(directory=imagePath,
                            target_size=(img_width,img_height),
                            class_mode='binary',
                            batch_size=batch_size)

filepath='weights.best.hdf5'
check_point = ModelCheckpoint(filepath=filepath,
                              monitor='val_acc',
                              verbose=1,
                              save_best_only='True',
                              mode='max')

image_dim = (img_width,img_height, 3)
if os.path.exists(filepath):
    model = load_model(filepath)
else:
    # model = create_dense_net(nb_classes=1,
    #                          img_dim=image_dim,
    #                          dropout_rate=0.5)
    model = densenet.DenseNet(image_dim,classes=1,activation='sigmoid')

filepath='weights.best.hdf5'
check_point = ModelCheckpoint(filepath=filepath,
                              monitor='val_acc',
                              verbose=1,
                              save_best_only='True',
                              mode='max')
callbacks_list = [check_point]


model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


model.fit_generator(data_generator,
                    steps_per_epoch=nb_train_samples//batch_size,
                    epochs=epochs,
                    callbacks=callbacks_list)
