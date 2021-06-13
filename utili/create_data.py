# -*- coding:UTF-8 -*-
"""
文件说明：
    tf.data.Dataset 创建数据
"""
import tensorflow as tf
import glob


def read_img_png(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img / 127.5 -1
    return img


def read_lab_png(lab_path):
    lab = tf.io.read_file(lab_path)
    lab = tf.image.decode_png(lab, channels=1)
    lab = lab / 255
    return lab


def load_data(img_path_1, img_path_2, lab_path):
    img1 = read_img_png(img_path_1)
    img2 = read_img_png(img_path_2)
    lab = read_lab_png(lab_path)
    img1 = tf.image.resize(img1, [512,512])
    img2 = tf.image.resize(img2, [512,512])
    lab = tf.image.resize(lab, [512,512])
    return (img1, img2), lab


imgs1 = glob.glob(r'E:\CDdata\LEVIR-CD\train\new\one\*')
imgs2 = glob.glob(r'E:\CDdata\LEVIR-CD\train\new\two\*')
labs = glob.glob(r'E:\CDdata\LEVIR-CD\train\new\lab\*')

train_dataset = tf.data.Dataset.from_tensor_slices((imgs1, imgs2, labs))
AUTO = tf.data.experimental.AUTOTUNE
trains = train_dataset.map(load_data, num_parallel_calls=AUTO)
print(trains)
BATCH_SIZE = 8
BUFFER_SIZE = 1000

trains = trains.cache().repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)
print(trains)



