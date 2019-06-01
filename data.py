import tensorflow as tf
import tf2lib as tl
import numpy as np
import random


# ==============================================================================
# =                                  datasets                                  =
# ==============================================================================

def make_32x32_dataset(dataset, batch_size, drop_remainder=True, shuffle=True, repeat=1, keep_percent=100):
    if dataset == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    else:
        raise NotImplementedError

    keep_index = train_images.shape[0] * keep_percent // 100
    train_images, train_labels = train_images[:keep_index], train_labels[:keep_index]

    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img, [32, 32])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    if shuffle:
        to_shuffle = list(zip(train_images, train_labels))
        random.shuffle(to_shuffle)
        train_images, train_labels = zip(*to_shuffle)
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        del to_shuffle

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


def make_celeba_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, repeat=1):
    @tf.function
    def _map_fn(img):
        crop_size = 108
        img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
        img = tf.image.resize(img, [resize, resize])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          drop_remainder=drop_remainder,
                                          map_fn=_map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)
    img_shape = (resize, resize, 3)
    len_dataset = len(img_paths) // batch_size

    return dataset, img_shape, len_dataset


def make_anime_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, repeat=1):
    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img, [resize, resize])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          drop_remainder=drop_remainder,
                                          map_fn=_map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)
    img_shape = (resize, resize, 3)
    len_dataset = len(img_paths) // batch_size

    return dataset, img_shape, len_dataset


# ==============================================================================
# =                               custom dataset                               =
# ==============================================================================

def make_custom_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, repeat=1):
    @tf.function
    def _map_fn(img):
        # ======================================
        # =               custom               =
        # ======================================
        img = ...  # custom preprocessings, should output img in [0.0, 255.0]
        # ======================================
        # =               custom               =
        # ======================================
        img = tf.image.resize(img, [resize, resize])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          drop_remainder=drop_remainder,
                                          map_fn=_map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)
    img_shape = (resize, resize, 3)
    len_dataset = len(img_paths) // batch_size

    return dataset, img_shape, len_dataset
