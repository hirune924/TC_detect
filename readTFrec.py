import tensorflow as tf

sess = tf.InteractiveSession()

def parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['image'], parsed_features['label']

def read_image(image_raw, label):
    image = tf.decode_raw(image_raw, tf.uint8)
    return image, label

dataset = tf.data.TFRecordDataset("03_TFrecord/train_5_TC.tfrecord")\
        .map(parse_function)\
        .map(read_image)\
        .shuffle(4)\
        .batch(4)

iterator = dataset.make_one_shot_iterator()
next_elem = iterator.get_next()

print(sess.run(next_elem))
