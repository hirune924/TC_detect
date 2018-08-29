#!/usr/bin/python 
# -*- coding: UTF-8 -*-


import glob
import os
import tensorflow as tf
from PIL import Image
import numpy as np

def parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['image'], parsed_features['label']


def read_image(image_raw, label):
    image = tf.decode_raw(image_raw, tf.uint8)
    return image, label


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


def model(images, labels):
    # ニューラルネットワークを計算グラフで作成する 
    # 形状変更
    const1 = tf.constant(255, tf.float32)
    reshape_image = tf.reshape(tf.cast(images, tf.float32), [-1, 64, 64, 1])
    x_image = tf.divide(reshape_image, const1)

    # 第2層 (畳み込み層)
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    y_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
     
    # 第3層 (プーリング層)
    y_pool1 = max_pool_2x2(y_conv1)
     
    # 第4層 (畳み込み層)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    y_conv2 = tf.nn.relu(conv2d(y_pool1, W_conv2) + b_conv2)
     
    # 第5層 (プーリング層)
    y_pool2 = max_pool_2x2(y_conv2)
     
    # 形状変更
    y_pool2_flat = tf.reshape(y_pool2, [-1, 32 * 32 * 64])
     
    # 第6層 (全結合層)
    W_fc1 = weight_variable([32 * 32 * 64, 1024])
    b_fc1 = bias_variable([1024])
    y_fc1 = tf.nn.relu(tf.matmul(y_pool2_flat, W_fc1) + b_fc1)
     
    # 第7層 (全結合層)
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y = tf.matmul(y_fc1, W_fc2) + b_fc2
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, depth=2,dtype=tf.float32), logits=y)
    soft = tf.nn.softmax(tf.matmul(y_fc1, W_fc2) + b_fc2)
    pred = tf.equal(tf.argmax(soft, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
     
    # 損失関数を計算グラフを作成する
    #t = tf.placeholder("float", [None, 2])
    #cross_entropy = -tf.reduce_sum(t * tf.log(y))
     
    # 次の(1)、(2)を行うための計算グラフを作成する。
    # (1) 損失関数に対するネットワークを構成するすべての変数の勾配を計算する。
    # (2) 勾配方向に学習率分移動して、すべての変数を更新する。
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    return train_step, accuracy

def main():
    #train_images, train_labels = read_tfrecord('03_TFrecord/train_5_TC.tfrecord')
    #train_op = model(train_images, train_labels)
    TC_list = glob.glob('03_TFrecord/TC/*')
    nonTC_list = glob.glob('03_TFrecord/nonTC/*')
    train_TC_dataset = tf.data.TFRecordDataset(TC_list[:13])\
        .map(parse_function)\
        .map(read_image)\
        .shuffle(1000)\
        .batch(64)\
        .repeat()
    train_nonTC_dataset = tf.data.TFRecordDataset(nonTC_list[:13])\
        .map(parse_function)\
        .map(read_image)\
        .shuffle(1000)\
        .batch(64)\
        .repeat()
    
    #train_dataset = tf.data.Dataset.zip((train_TC_dataset, train_nonTC_dataset))

    test_TC_dataset = tf.data.TFRecordDataset(TC_list[14:])\
        .map(parse_function)\
        .map(read_image)\
        .shuffle(1000)\
        .batch(64)\
        .repeat()
    test_nonTC_dataset = tf.data.TFRecordDataset(nonTC_list[14:])\
        .map(parse_function)\
        .map(read_image)\
        .shuffle(1000)\
        .batch(64)\
        .repeat()

    #test_dataset = tf.data.Dataset.zip((test_nonTC_dataset, test_TC_dataset))

    TC_iterator = tf.data.Iterator.from_structure(train_TC_dataset.output_types,
                                           train_TC_dataset.output_shapes)
    nonTC_iterator = tf.data.Iterator.from_structure(train_nonTC_dataset.output_types,
                                           train_nonTC_dataset.output_shapes)



    #train_iterator = train_dataset.make_one_shot_iterator()
    #train_non_iterator = train_non_dataset.make_one_shot_iterator()

    #test_iterator = test_dataset.make_one_shot_iterator()
    #test_non_iterator = test_non_dataset.make_one_shot_iterator()

    TC_images, TC_labels = TC_iterator.get_next()
    nonTC_images, nonTC_labels = nonTC_iterator.get_next()

    train_TC_init_op = TC_iterator.make_initializer(train_TC_dataset)
    test_TC_init_op = TC_iterator.make_initializer(test_TC_dataset)

    train_nonTC_init_op = nonTC_iterator.make_initializer(train_nonTC_dataset)
    test_nonTC_init_op = nonTC_iterator.make_initializer(test_nonTC_dataset)
    #non_img, non_label = train_non_iterator.get_next()

    #test_img, test_label = test_iterator.get_next()
    #test_non_img, test_non_label = test_non_iterator.get_next()

    # 初期化を行うための計算グラフを作成する。
    train_op, acc = model(tf.concat([TC_images, nonTC_images],0), tf.concat([TC_labels, nonTC_labels],0))
    #train_op, acc = model(tf.concat([img,non_img],0), tf.concat([label,non_label],0))
    #_, test_acc = model(tf.concat([test_non_img,test_img],0), tf.concat([test_non_label,test_label],0))
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(100):
            sess.run(train_TC_init_op)
            sess.run(train_nonTC_init_op)
            for step in range(300):
                _, accuracy = sess.run([train_op,acc])

            sess.run(test_TC_init_op)
            sess.run(test_nonTC_init_op)
            test_accuracy = sess.run(acc)
            print('step: {}, accuracy: {}, test_accuracy: {}'.format(step+1+epoch*300, accuracy, test_accuracy))

        saver.save(sess, "./ckpt/model.ckpt")


if __name__ == '__main__':
    main()

