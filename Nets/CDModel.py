# -*- coding:UTF-8 -*-

"""
文件说明：

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Subtract, Conv2DTranspose, Input, \
    AveragePooling2D, BatchNormalization, Activation, UpSampling2D, Concatenate, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy


def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (possible_negatives + K.epsilon())


def F1_score(y_true, y_pred):
    R = Recall(y_true, y_pred)
    P = Precision(y_true, y_pred)
    return 2 * P * R / (R + P)


def PA(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)

    TP = tf.reduce_sum(tf.cast(y_true * y_pred, tf.int32))
    TN = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.int32))
    FP = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.int32))
    FN = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.int32))

    TP = tf.cast(TP, tf.float32)
    TN = tf.cast(TN, tf.float32)
    FP = tf.cast(FP, tf.float32)
    FN = tf.cast(FN, tf.float32)

    PA = (TP + TN) / (TP + TN + FP + FN + K.epsilon())
    return PA


def ce_dice_loss(y_true, y_pred):
    # binary_crossentropy L=-[y_true * log(y_pred)+(1-y_true)*log(1 - y_pred)]
    ce_loss = binary_crossentropy(y_true, y_pred)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))  # 平方和
    dice_loss = - tf.math.log((intersection + K.epsilon()) / (union + K.epsilon()))
    # K.epsilon() = 1e-07 防止分母为零
    loss = ce_loss + dice_loss
    return loss


def psp_modele(input_x, filters_nums=512, k_size=1):
    # 金字塔池化 Pyramid Scene Parsing Network
    _, _, c, _ = input_x.shape

    poolsize = [8, 4, 2, 1]
    # print(input_x.shape)
    # 6
    x_c1 = AveragePooling2D(pool_size=c // poolsize[0], name='ave_c1')(input_x)
    x_c1 = Conv2D(filters=filters_nums, kernel_size=k_size, strides=1, padding='same', name='conv_c1')(x_c1)
    x_c1 = BatchNormalization(momentum=0.95, axis=-1)(x_c1)
    x_c1 = Activation(activation='relu')(x_c1)
    x_c1 = UpSampling2D(size=(c // poolsize[0], c // poolsize[0]), name='up_c1')(x_c1)
    # print(x_c1.shape)

    # 3
    x_c2 = AveragePooling2D(pool_size=c // poolsize[1], name='ave_c2')(input_x)
    x_c2 = Conv2D(filters=filters_nums, kernel_size=k_size, strides=1, padding='same', name='conv_c2')(x_c2)
    x_c2 = BatchNormalization(momentum=0.95, axis=-1)(x_c2)
    x_c2 = Activation(activation='relu')(x_c2)
    x_c2 = UpSampling2D(size=(c // poolsize[1], c // poolsize[1]), name='up_c2')(x_c2)
    # print(x_c2.shape)

    # 2
    x_c3 = AveragePooling2D(pool_size=c // poolsize[2], name='ave_c3')(input_x)
    x_c3 = Conv2D(filters=filters_nums, kernel_size=k_size, strides=1, padding='same', name='conv_c3')(x_c3)
    x_c3 = BatchNormalization(momentum=0.95, axis=-1)(x_c3)
    x_c3 = Activation(activation='relu')(x_c3)
    x_c3 = UpSampling2D(size=(c // poolsize[2], c // poolsize[2]), name='up_c3')(x_c3)
    # print(x_c3.shape)

    # 1
    x_c4 = GlobalAveragePooling2D(name='glob1')(input_x)
    x_c4 = tf.reshape(x_c4, (-1, 1, 1, filters_nums))
    x_c4 = Conv2D(filters=filters_nums, kernel_size=k_size, strides=1, padding='same', name='conv_c4')(x_c4)
    x_c4 = BatchNormalization(momentum=0.95, axis=-1)(x_c4)
    x_c4 = Activation(activation='relu')(x_c4)
    x_c4 = UpSampling2D(size=(c, c), name='up_c4')(x_c4)
    # print(x_c4.shape)

    x = Concatenate(axis=-1, name='concat')([input_x, x_c1, x_c2, x_c3, x_c4])
    x = Conv2D(filters=filters_nums, kernel_size=3, name='conv_c6', padding='same')(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)
    return x


def res_conv_block(input_tensor, kernel_num, name_num):
    '''
    残差模块：
    input_tensor：输入特征
    kernel_num：卷积核个数
    name_num当前卷积块是第几个
    '''
    x = input_tensor
    x = Conv2D(kernel_num, (1, 1))(x)
    x1 = Conv2D(kernel_num, (3, 3), padding='same', kernel_initializer='he_normal',
                name='con{}_1'.format(name_num))(input_tensor)
    x1 = tf.nn.leaky_relu(x1, alpha=0.1)
    x1 = Conv2D(kernel_num, (3, 3), padding='same', kernel_initializer='he_normal',
                name='con{}_2'.format(name_num))(x1)
    out = x + x1
    out = tf.nn.leaky_relu(out, alpha=0.1)
    return out


def res_feature_extract(input_tensor=(512, 512, 3), kernel_num=64):
    x = Input(input_tensor)
    feature_1 = layer_1 = res_conv_block(x, kernel_num, name_num=1)
    pool_1 = Conv2D(kernel_num, (1, 1), strides=2, name='pool_1')(layer_1)

    feature_2 = layer_2 = res_conv_block(pool_1, 2 * kernel_num, name_num=2)
    pool_2 = Conv2D(2 * kernel_num, (1, 1), strides=2, name='pool_2')(layer_2)

    feature_3 = layer_3 = res_conv_block(pool_2, 4 * kernel_num, name_num=3)
    pool_3 = Conv2D(4 * kernel_num, (1, 1), strides=2, name='pool_3')(layer_3)

    layer_4 = res_conv_block(pool_3, 8 * kernel_num, name_num=4)
    layer_4 = res_conv_block(layer_4, 8 * kernel_num, name_num=5)

    # psp
    feature_4 = layer_4 = psp_modele(layer_4)
    net = Conv2D(8 * kernel_num, (1, 1), strides=2, name='pool_4')(layer_4)

    # feature_5 = layer_5 = res_conv_block(pool_4, 16 * kernel_num, name_num=5)
    # net = Conv2D(8 * kernel_num, (1, 1), strides=2, name='pool_5')(pool_4)
    model = Model(inputs=x, outputs=[net, feature_1, feature_2, feature_3, feature_4])
    return model


def up_conv(input_layer, diff, kernel_num, name_num):
    x1 = Conv2DTranspose(kernel_num, (3, 3), padding='same', kernel_initializer='he_normal',
                         strides=2, name='up_conv_{}'.format(name_num))(input_layer)
    x1 = tf.nn.leaky_relu(x1, alpha=0.1)
    x1 = tf.concat([x1, diff], axis=-1)

    x1 = Conv2D(kernel_num, (3, 3), padding='same', kernel_initializer='he_normal',
                name='con{}_1'.format(name_num))(x1)
    x1 = tf.nn.leaky_relu(x1, alpha=0.1)
    return x1


def abs_layer(input_layer_y, input_layer_x):
    '''  取两个层差的绝对值 '''
    return tf.abs(Subtract()([input_layer_y, input_layer_x]))


def get_model(inputs=(512, 512, 3)):
    input_1 = Input(inputs)
    input_2 = Input(inputs)

    res_model = res_feature_extract()
    # 特征提取
    net_x, feature_1_x, feature_2_x, feature_3_x, feature_4_x = res_model(input_1)
    net_y, feature_1_y, feature_2_y, feature_3_y, feature_4_y = res_model(input_2)

    # 差值特征提取
    diff_1 = abs_layer(feature_1_y, feature_1_x)
    diff_2 = abs_layer(feature_2_y, feature_2_x)
    diff_3 = abs_layer(feature_3_y, feature_3_x)
    diff_4 = abs_layer(feature_4_y, feature_4_x)
    # diff_5 = abs_layer(feature_5_y, feature_5_x)

    # 上采样
    # up_6 = up_conv(net_y, diff_5, kernel_num=1024, name_num=6)
    up_7 = up_conv(net_y, diff_4, kernel_num=512, name_num=7)
    up_8 = up_conv(up_7, diff_3, kernel_num=256, name_num=8)
    up_9 = up_conv(up_8, diff_2, kernel_num=128, name_num=9)
    up_10 = up_conv(up_9, diff_1, kernel_num=64, name_num=10)

    # 分类层
    out = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(up_10)
    model = Model(inputs=[input_1, input_2], outputs=out)
    return model

if __name__ == '__main__':
    model = get_model()
    model.summary()


# print(model.get_layer('up_conv_7'))


# print(len(model.layers))
# for i in model.layers:
#     print(i,end='\n')
# # model.fit_generator()
#
# tf.keras.utils.plot_model(model, to_file='model.png')

'''
tf.nn.softmax()
tf.nn.sigmoid()

# 二元交叉熵，非ont-hot编码，sigmoid激活函数
tf.keras.losses.binary_crossentropy()

# 多元交叉熵， 非ont-hot编码， softmax激活函数
tf.keras.losses.sparse_categorical_crossentropy()

# 多元交叉熵， ont-hot编码， softmax激活函数
tf.keras.losses.categorical_crossentropy()
'''



