#!/usr/bin/python 
# -*- coding: UTF-8 -*-


import glob
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

#OUTPUT_TFRECORD_NAME = "train_1_TC.tfrecord"  # アウトプットするTFRecordファイル名


def CreateTensorflowReadFile(img_files, out_file, label, flag):
    with tf.python_io.TFRecordWriter(out_file) as writer:
        for f in img_files:
            # ファイルを開く
            #with cv2.imread(f) as image:  # グレースケール
            image = cv2.imread(f,-1)
            if image is None:
                continue
            height = image.shape[0]
            width = image.shape[1]
            image_raw = image.tostring()
            #label = int( f[ f.rfind("-") + 1 : -4] )  # ファイル名からラベルを取得
            label = label
            name = os.path.basename(f)

            if flag:
                example = tf.train.Example(features=tf.train.Features(feature={
                        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
                        }))
            else:
                example = tf.train.Example(features=tf.train.Features(feature={
                        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('utf-8')])),
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
                        }))

            # レコード書込
            writer.write(example.SerializeToString())

def main():
    for target in range(8,17):
        # 書き込み
        files = glob.glob("01_extract/train_" + str(target) + "/train/TC/*.tif")
        CreateTensorflowReadFile(files , "03_TFrecord/train_" + str(target) + "_TC.tfrecord", 1, True)
 
        files = glob.glob("01_extract/train_" + str(target) + "/train/nonTC/*.tif")
        CreateTensorflowReadFile(files , "03_TFrecord/train_" + str(target) + "_nonTC.tfrecord", 0, True)

    for target in range(1,4):
        # 書き込み
        files = glob.glob("01_extract/test_" + str(target) + "/test/*.tif")
        CreateTensorflowReadFile(files , "03_TFrecord/test_" + str(target) + ".tfrecord", 0, False)


if __name__ == '__main__':
    main()

