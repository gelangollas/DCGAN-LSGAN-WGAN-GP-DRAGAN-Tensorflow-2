import tensorflow as tf
import tf2lib as tl
import numpy as np
import random


# ==============================================================================
# =                                  datasets                                  =
# ==============================================================================

def make_32x32_dataset(dataset, batch_size, drop_remainder=True, shuffle=True, repeat=1):
    if dataset == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = np.concatenate((train_images, test_images))
        train_labels = np.concatenate((train_labels, test_labels))
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_images = np.concatenate((train_images, test_images))
        train_labels = np.concatenate((train_labels, test_labels))
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        train_images = np.concatenate((train_images, test_images))
        train_labels = np.concatenate((train_labels, test_labels))
    else:
        raise NotImplementedError

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

    def encode_label_into_image(image, label):
        encoded = np.array(image)
        for i in range(encoded.shape[1]):
            encoded[0][i][0] = 0
        encoded[0][label][0] = 1
        return encoded

    for i in range(train_images.shape[0]):
        encoded = encode_label_into_image(train_images[i], train_labels[i])
        train_images[i] = encoded

    dataset = tl.memory_data_batch_dataset(train_images,
                                           batch_size,
                                           drop_remainder=drop_remainder,
                                           map_fn=_map_fn,
                                           shuffle=False,
                                           repeat=repeat)

    img_shape = (32, 32, train_images.shape[-1])
    len_dataset = len(train_images) // batch_size

    return dataset, img_shape, len_dataset


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
