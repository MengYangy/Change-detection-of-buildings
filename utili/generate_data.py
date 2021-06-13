# -*- coding:UTF-8 -*-


"""
文件说明：
    
"""
import os
import random
import cv2 as cv
import numpy as np


def get_data(base_path):
    '''
    train_X, train_Y, train_label,test_X, test_Y, test_label = get_data()
    返回数据的目录+名字
    '''
    path_main_part = ['/one/', '/two/', '/lab/']
    img_names = os.listdir(base_path + path_main_part[0])
    img_names_num = len(img_names)
    train_img_num = int(img_names_num * 0.9)
    random.shuffle(img_names)
    train_img_names = img_names[:train_img_num]
    test_img_names = img_names[train_img_num:]

    train_X, train_Y, train_label =[],[],[]
    test_X, test_Y, test_label = [],[],[]
    for i in range(train_img_num):
        train_X.append(os.path.join(base_path + path_main_part[0],train_img_names[i]))
        train_Y.append(os.path.join(base_path + path_main_part[1], train_img_names[i]))
        train_label.append(os.path.join(base_path + path_main_part[2], train_img_names[i]))
    for i in range(img_names_num - train_img_num):
        test_X.append(os.path.join(base_path + path_main_part[0], test_img_names[i]))
        test_Y.append(os.path.join(base_path + path_main_part[1], test_img_names[i]))
        test_label.append(os.path.join(base_path + path_main_part[2], test_img_names[i]))
    return train_X, train_Y, train_label,test_X, test_Y, test_label



def load_data(input_path, isLab=False):
#     print(img1_paths, img2_paths)
    if isLab:
        img = cv.imread(input_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
        img = np.array(img, dtype='float') / 255.
    else:
        img = cv.imread(input_path)
        img = np.array(img, dtype=np.float32) / 127.5 - 1
    return img


def generate_train_val_data(train_path_list_x, train_path_list_y, train_path_list_lab, batch_size):
    '''
    生成训练、验证样本的方法一
    输出字典格式的函数，目前仍存在错误
    '''
    while True:
        train_x = []
        train_y = []
        train_lab = []
        for i in range(len(train_path_list_x)):
            if i % batch_size == 0 and i != 0:
                yield ({'input_x':train_x, 'input_y':train_y}, train_lab)
                train_x = []
                train_y = []
                train_lab = []
            train_x.append(load_data(train_path_list_x[i]))
            train_y.append(load_data(train_path_list_y[i]))
            train_lab.append(load_data(train_path_list_lab[i], isLab=True))


def generate_train_val_data_2(train_path_list_x, train_path_list_y, train_path_list_lab, batch_size):
    '''
    生成训练、验证样本的方法二
    预先定义好生成器输出的格式(（batch,h,w,3）,（batch,h,w,3）),（batch,h,w,1）
    '''
    while True:
        train_img_x = np.zeros((batch_size,512,512,3))
        train_img_y = np.zeros((batch_size,512,512,3))
        train_img_lab = np.zeros((batch_size,512,512,1))

        for i in range(len(train_path_list_x)):
            if i % batch_size == 0 and i != 0:
                yield (train_img_x, train_img_y), train_img_lab
            train_img_x[i % 6, :, :, :] = load_data(train_path_list_x[i])
            train_img_y[i % 6, :, :, :] = load_data(train_path_list_y[i])
            train_img_y[i % 6, :, :, :] = load_data(train_path_list_lab[i], isLab=True)


if __name__ == '__main__':
    train_list_x, train_list_y, train_list_lab, val_list_x, val_list_y, val_list_lab = get_data(
        base_path='E:/CDdata/LEVIR-CD/train/new')
    print(train_list_x[1])
    print(val_list_lab[1])

    img1 = load_data(train_list_x[1])
    lab1 = load_data(train_list_lab[1], isLab=True)
    print(img1.shape)
    print(lab1.shape)
    print(type(img1))

    # 训练集迭代生成器
    iter_train_data = generate_train_val_data(train_path_list_x=train_list_x,
                                   train_path_list_y=train_list_y,
                                   train_path_list_lab=train_list_lab,
                                   batch_size=2)
    # 验证集迭代生成器
    iter_val_data = generate_train_val_data(train_path_list_x=val_list_x,
                                   train_path_list_y=val_list_y,
                                   train_path_list_lab=val_list_lab,
                                   batch_size=4)

    # a = next(iter_val_data)
    # print(len(a[1]))
    # print('input_X', np.array(a[0].get('input_x')).shape)
    # print('input_y', np.array(a[0].get('input_y')).shape)
    # print(np.array(a[1]).shape)





