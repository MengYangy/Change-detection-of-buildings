import os
import tensorflow as tf
import matplotlib.pyplot as plt


model_path = './change.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError('找不到模型权重')
model = tf.keras.models.load_model(model_path)


path1 = './image/A/1.png'
path2 = './image/B/1.png'

img1=tf.io.read_file(path1)
img1=tf.image.decode_png(img1,channels=3)
img1=tf.image.resize(img1,[512,512])
img1=tf.cast(img1,tf.float32)/127.5-1
img1=tf.expand_dims(img1,axis=0)

img2=tf.io.read_file(path2)
img2=tf.image.decode_png(img2,channels=3)
img2=tf.image.resize(img2,[512,512])
img2=tf.cast(img2,tf.float32)/127.5-1
img2=tf.expand_dims(img2,axis=0)


result=model([img1,img2],training=False)


test_output=tf.argmax(result,axis=-1)
test_output=test_output[...,tf.newaxis]
test_output=tf.squeeze(test_output)
plt.imshow(test_output)