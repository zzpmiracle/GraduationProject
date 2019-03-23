import os

from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import save_model,load_model

import densenet

batch_size = 32
nb_classes = 240
nb_epoch = 10

img_rows , img_cols = 28,28
nb_filters = 32
pool_size = (2,2)
kernel_size = (3,3)


(x_train,y_train),(x_test,y_test) = mnist.load_data()

X_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
X_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
input_shape = (img_rows,img_cols,1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /=255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)

filepath='weights.best.hdf5'
check_point = ModelCheckpoint(filepath=filepath,
                              monitor='val_acc',
                              verbose=1,
                              save_best_only='True',
                              mode='max')
callbacks_list = [check_point]
if os.path.exists(filepath):
    model = load_model(filepath)
    print(1)
else:
    model = densenet.DenseNet(include_top=False,
                          input_shape=input_shape,
                          classes=nb_classes,
                          depth=40,
                          growth_rate=12,
                          bottleneck=True,
                          dropout_rate=0.3,
                          )
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.fit(X_train,Y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          verbose=0,
          validation_split=0.33,
          callbacks=callbacks_list)

score = model.evaluate(X_test,Y_test,verbose=0)
print('Test score ',score[0])
print('Test accuracy ',score[1])