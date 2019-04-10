import os

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.utils import plot_model
from matplotlib import pyplot as plt

import densenet
from keras.preprocessing.image import ImageDataGenerator

train_image_path = 'D:\\Event&NoEvent\\train'
test_image_path = 'D:\\Event&NoEvent\\test'
nb_train_samples = 4000
nb_tests_samples = 1000

img_width, img_height = 32, 32
image_dim = (img_width,img_height, 3)

batch_size = 32
epochs = 20

train_data_gen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.3
                             )

train_data_generator = train_data_gen.flow_from_directory(directory=train_image_path,
                            target_size=(img_width,img_height),
                            class_mode='binary',
                            batch_size=batch_size)

filepath='weights.best.hdf5'

if os.path.exists(filepath):
    model = load_model(filepath)
else:
    model = densenet.DenseNet(image_dim,classes=1,activation='sigmoid')

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
plot_model(model, to_file='model.png')

filepath='weights.best.hdf5'
check_point = ModelCheckpoint(filepath=filepath,
                              monitor='acc',
                              verbose=1,
                              save_best_only='True',
                              mode='max')

history = model.fit_generator(train_data_generator,
                    steps_per_epoch=nb_train_samples//batch_size+1,
                    epochs=epochs,
                    callbacks=[check_point])
print(history.history)
# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


test_data_gen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.3
                             )

test_data_generator = train_data_gen.flow_from_directory(directory=train_image_path,
                            target_size=(img_width,img_height),
                            class_mode='binary',
                            batch_size=batch_size)
score = model.evaluate_generator(test_data_generator,
                                 steps=nb_tests_samples//batch_size+1)
print(score[-1])