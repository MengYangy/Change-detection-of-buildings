import tensorflow as tf
from Nets.CDModel import get_model, ce_dice_loss, Recall, F1_score, Precision
from utili.generate_data import get_data, load_data, generate_train_val_data
import numpy as np
import cv2 as cv


train_list_x, train_list_y, train_list_lab, val_list_x, val_list_y, val_list_lab = get_data(
        base_path='E:/CDdata/LEVIR-CD/train/new')
batch_size = 4
# 训练集迭代生成器
iter_train_data = generate_train_val_data(train_path_list_x=train_list_x,
                               train_path_list_y=train_list_y,
                               train_path_list_lab=train_list_lab,
                               batch_size=batch_size)
# 验证集迭代生成器
iter_val_data = generate_train_val_data(train_path_list_x=val_list_x,
                               train_path_list_y=val_list_y,
                               train_path_list_lab=val_list_lab,
                               batch_size=batch_size)

learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.01,
                                                                      decay_steps=5,
                                                                      decay_rate=0.5)
model = get_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn),
              loss=ce_dice_loss,
              metrics=['accuracy', Recall, Precision, F1_score])
model.summary()


EPOCHS = 10
L_EP = 0
STEPS_PER_EPOCH = len(train_list_x) // batch_size
val_STEPS_PER_EPOCH = len(val_list_x) // batch_size
model.fit(iter_train_data,
          steps_per_epoch=STEPS_PER_EPOCH,
          epochs=EPOCHS,
          initial_epoch=L_EP,
          validation_data=iter_val_data,
          validation_steps=val_STEPS_PER_EPOCH,
          verbose=1)

model.save_weights('./net_weight_epoch{}_{}.h5'.format(L_EP, EPOCHS))


# >>>>>>>>>>>>>>>>>> pred start ******************

path1='./img1/1.png'
path2='./img2/1.png'

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

cv.imwrite('./pred/1.png', test_output)
# ****************** pred  end  <<<<<<<<<<<<<<<<<<