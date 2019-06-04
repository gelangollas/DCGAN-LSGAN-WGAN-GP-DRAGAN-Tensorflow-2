import tensorflow as tf
import tf2lib as tl
import numpy as np
import random


#
# формирование датасета
#

def make_32x32_dataset(dataset, batch_size, drop_remainder=True, shuffle=True, repeat=1, keep_percent=100):
    if dataset == 'fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    else:
        raise NotImplementedError

    keep_index = train_images.shape[0] * keep_percent // 100
    train_images, train_labels = train_images[:keep_index], train_labels[:keep_index]

    # изменение размера изображения
    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img, [32, 32])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    # перемешивание датасета
    if shuffle:
        to_shuffle = list(zip(train_images, train_labels))
        random.shuffle(to_shuffle)
        train_images, train_labels = zip(*to_shuffle)
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        del to_shuffle

    # преобразование в tensorflow объект
    dataset = tl.memory_data_batch_dataset(train_images,
                                           batch_size,
                                           drop_remainder=drop_remainder,
                                           map_fn=_map_fn,
                                           shuffle=False,
                                           repeat=repeat)

    labels = tf.data.Dataset.from_tensor_slices(train_labels).batch(batch_size, drop_remainder=True)

    img_shape = (32, 32, train_images.shape[-1])
    len_dataset = len(train_images) // batch_size

    return dataset, labels, img_shape, len_dataset
