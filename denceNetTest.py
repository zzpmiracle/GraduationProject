import densenet
from keras.preprocessing.image import ImageDataGenerator

imagePath = 'D:\Event&NoEvent'

datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             validation_split=0.3
                             )

datagen.flow_from_directory(directory=imagePath,
                                       target_size=(224,224),
                                       class_mode='binary'
                                       )
image_dim = (224, 224, 3)
model = densenet.DenseNet(include_top=False,
                          input_shape=image_dim,
                          classes=2,
                          depth=40,
                          growth_rate=12,
                          bottleneck=True,
                          dropout_rate=0.5,
                          )

model.fit_generator(datagen)
