from keras.preprocessing import image
img = image.load_img('0.png',target_size=(32,32))
img = image.img_to_array(img)
img *= (1./255)
print(img)