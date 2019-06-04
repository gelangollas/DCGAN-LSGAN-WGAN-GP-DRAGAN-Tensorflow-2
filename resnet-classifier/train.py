from __future__ import print_function

import os
import pickle
import random
from random import shuffle

import cv2
import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from .network import *


def train(dataset_module=fashion_mnist,
          epochs=100,
          data_augmentation=False,
          gan_augmentation=False,
          gan_name='gan',
          samples_to_take=1000,
          path_to_samples='output/fashion-dcgan-cond/generated',
          path_to_results='results',
          keep_percent=100):
    """Обучение нейросети для данных параметров

    # Аргументы
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    """

    # размер изображения
    image_shape = (32, 32)

    # Параметры обучения
    batch_size = 128
    num_classes = 10
    random_seed = 1234  # для воспроизведения результатов
    dataset = dataset_module.__name__.split('.')[-1]

    # вычитание пиксельного среднего улучшает точность
    subtract_pixel_mean = True

    # для воспроизведения результатов
    random.seed(random_seed)

    # параметр resnet'a
    n = 5

    # версия resnet'a
    version = 1

    # вычисление глубины сети из парамметра n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # название модели
    model_type = 'ResNet%dv%d' % (depth, version)

    # загрузка данных
    (x_train, y_train), (x_test, y_test) = dataset_module.load_data()

    # уменьшение размера датасета до keep_percent процентов
    def shuffle_dataset(x, y):
        indexes = list(range(len(x)))
        shuffle(indexes, )
        res_x = [x[i] for i in indexes]
        res_y = [y[i] for i in indexes]
        return np.array(res_x), np.array(res_y)

    x_train, y_train = shuffle_dataset(x_train, y_train)
    keep_index = len(x_train) * keep_percent // 100
    x_train, y_train = x_train[:keep_index], y_train[:keep_index]

    # нормализация fashion_mnist датасета
    if dataset == 'fashion_mnist':
        x_train_resized = np.zeros((x_train.shape[0], *image_shape))
        x_test_resized = np.zeros((x_test.shape[0], *image_shape))

        for i in range(x_train.shape[0]):
            x_train_resized[i] = cv2.resize(x_train[i], image_shape)

        for i in range(x_test.shape[0]):
            x_test_resized[i] = cv2.resize(x_test[i], image_shape)

        x_train = x_train_resized.reshape((*x_train_resized.shape, 1))
        x_test = x_test_resized.reshape((*x_test_resized.shape, 1))

        y_train = y_train.reshape((*y_train.shape, 1))
        y_test = y_test.reshape((*y_test.shape, 1))

    elif dataset == 'cifar10':  # cifar10 уже в нужном виде
        pass

    # размерности изображения
    input_shape = x_train.shape[1:]

    # добавление сгенерированных картинок в выборку
    if gan_augmentation:
        gan_samples = np.zeros((0, *input_shape))
        gan_labels = np.zeros((0, 1))
        for i in range(num_classes):
            file_name = f'class_{i}.pkl'
            with open(path_to_samples + '/' + file_name, 'rb') as file:
                samples_i = pickle.load(file)
                gan_samples = np.concatenate([gan_samples, samples_i[:samples_to_take]])
                samples_taken = min(samples_i.shape[0], samples_to_take)
                gan_labels = np.concatenate([gan_labels, np.ones((samples_taken, 1)) * i])

        x_train = np.concatenate([x_train, gan_samples])
        y_train = np.concatenate([y_train, gan_labels])
        x_train, y_train = shuffle_dataset(x_train, y_train)

    # нормализация данных
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # вычитание пиксельного среднего в зависимости от флага
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    # преобразование меток класса в бинарный вектор
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    print(model_type)

    # приготовить папку с результатами обучения
    save_dir = path_to_results + f'/{dataset}_{keep_percent}_{model_type}_aug{data_augmentation}_{gan_name}{gan_augmentation}'
    model_name = f'best_model.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # сохранение пиксельного среднего
    with open(f'{save_dir}/pixelmean.pkl', 'wb') as f:
        pickle.dump(x_train_mean, f)

    # подготовка колбеков для сохранения модели и изменения скорости обучения
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1, save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # запуск обучения с или без стандартной аугментации
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            callbacks=callbacks)

    else:
        print('Using real-time data augmentation.')
        # препроцессинг и генерация аугментированных изображений в реальном времени
        datagen = ImageDataGenerator(
            rotation_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            validation_split=0.0)
        datagen.fit(x_train)
        # обучение модели на батчах, сгенерированных через datagen.flow()
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      validation_data=(x_test, y_test),
                                      epochs=epochs, verbose=1,
                                      steps_per_epoch=len(x_train) / batch_size,
                                      callbacks=callbacks)

    # оценка обученной модели
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    # сохранение истории обучения
    with open(f'{save_dir}/history.pkl', 'wb') as f:
        pickle.dump(history, f)
