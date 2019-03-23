import os

from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model

import densenet
from keras.preprocessing.image import ImageDataGenerator

from densenet_fast import create_dense_net

imagePath = 'D:\Event&NoEvent'
img_width,img_height = 224,224
batch_size = 16
nb_train_samples = 2000
epochs = 50


datagen = ImageDataGenerator(rescale=1./255,
                             featurewise_center=True,
                             featurewise_std_normalization=True,
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

image_dim = (224, 224, 3)
# model = densenet.DenseNet(include_top=False,
#                           input_shape=image_dim,
#                           classes=2,
#                           depth=40,
#                           growth_rate=12,
#                           bottleneck=True,
#                           dropout_rate=0.5,
#                           )
if os.path.exists(filepath):
    model = load_model(filepath)
else:
    model = create_dense_net(nb_classes=1,
                             img_dim=image_dim,
                             dropout_rate=0.5)
model.summary()
model.compile(loss='binarycrossentry',
              optimizer='adadelta',
              metrics=['accuracy'])


model.fit_generator(data_generator,
                    steps_per_epoch=nb_train_samples//batch_size,
                    epochs=epochs)
model.save_model('first_try.h5')
