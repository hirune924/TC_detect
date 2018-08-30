#!/usr/bin/python 
# -*- coding: UTF-8 -*-


import glob
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import vgg

slim = tf.contrib.slim



def parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
            "name": tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['image'], parsed_features['name']


def read_image(image_raw, name):
    image = tf.decode_raw(image_raw, tf.float32)
    return image, name

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
   
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def model(images, name):
    # ニューラルネットワークを計算グラフで作成する
    x_image = tf.reshape(tf.cast(images, tf.float32), [-1, 64, 64, 1])
    x_image = tf.image.resize_images(x_image,[224,224])

    net, end_points = vgg.vgg_16(x_image, num_classes=2, is_training=False, dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='vgg_16',
               fc_conv_padding='VALID',
               global_pool=False)


    ## 形状変更
    #const1 = tf.constant(255, tf.float32)
    #reshape_image = tf.reshape(tf.cast(images, tf.float32), [-1, 64, 64, 1])
    #x_image = tf.divide(reshape_image, const1)
    # 
    ## 第2層 (畳み込み層)
    #W_conv1 = weight_variable([5, 5, 1, 32])
    #b_conv1 = bias_variable([32])
    #y_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # 
    ## 第3層 (プーリング層)
    #y_pool1 = max_pool_2x2(y_conv1)
    # 
    ## 第4層 (畳み込み層)
    #W_conv2 = weight_variable([5, 5, 32, 64])
    #b_conv2 = bias_variable([64])
    #y_conv2 = tf.nn.relu(conv2d(y_pool1, W_conv2) + b_conv2)
    # 
    ## 第5層 (プーリング層)
    #y_pool2 = max_pool_2x2(y_conv2)
    # 
    ## 形状変更
    #y_pool2_flat = tf.reshape(y_pool2, [-1, 32 * 32 * 64])
    # 
    ## 第6層 (全結合層)
    #W_fc1 = weight_variable([32 * 32 * 64, 1024])
    #b_fc1 = bias_variable([1024])
    #y_fc1 = tf.nn.relu(tf.matmul(y_pool2_flat, W_fc1) + b_fc1)
    # 
    ## 第7層 (全結合層)
    #W_fc2 = weight_variable([1024, 2])
    #b_fc2 = bias_variable([2])
    #y = tf.matmul(y_fc1, W_fc2) + b_fc2
    y = end_points["vgg_16/fc8"]
    soft = tf.nn.softmax(y)
    
    return name, soft

def main():
    #train_images, train_labels = read_tfrecord('03_TFrecord/train_5_TC.tfrecord')
    #train_op = model(train_images, train_labels)
    
    #dataset = tf.data.TFRecordDataset(["03_TFrecord/test_3.tfrecord"])\
    TFreco_list = glob.glob('03_TFrecord/test/*')
    dataset = tf.data.TFRecordDataset(TFreco_list)\
        .map(parse_function)\
        .map(read_image)\
        .batch(1)
    
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    # 初期化を行うための計算グラフを作成する。
    res = model(images, labels)
    saver = tf.train.Saver()
    step = 0
    submit = ""
    tmp = ""
    with tf.Session() as sess:
        saver.restore(sess,"./ckpt/model.ckpt-29700")
        while True:
            try:
                name, soft = sess.run(res)
                print('step: {}, name: {}, soft: {}'.format(step, name, soft))
                if soft[0,0] > 0.999:
                    submit += name[0] + "\t" + "0\n"
                else:
                    submit += name[0] + "\t" + "1\n"
                tmp += name[0] + "\t" + str(soft[0,0])  + "\t" + str(soft[0,1]) + "\n"
                step += 1
            except tf.errors.OutOfRangeError:
                print("End Of Data")
                with open("submit.tsv", mode='w') as f:
                    f.write(submit)
                with open("tmp.tsv", mode='w') as f:
                    f.write(tmp)
                break



if __name__ == '__main__':
    main()

