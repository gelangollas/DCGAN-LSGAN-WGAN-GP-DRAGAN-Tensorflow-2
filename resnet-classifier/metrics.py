from __future__ import print_function

import os
import pickle

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from keras.datasets import cifar10, fashion_mnist
from keras.optimizers import Adam
from sklearn import metrics

from .network import *

#
# описание обученной нейронной сети
#

# параметры для генерации метрик по модели
model_folder = 'cifar10_15_ResNet32v1_augTrue_ganFalse'
path_to_results = 'results/' + model_folder
dataset = cifar10
num_classes = 10
image_shape = (32, 32)


# описание структуры нейронной сети, необходимо для загрузки весов обученной модели
n = 5
version = 1
if dataset == cifar10:
    input_shape = (32, 32, 3)
else:
    input_shape = (32, 32, 1)
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2
model_type = 'ResNet%dv%d' % (depth, version)
batch_size = 128

if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

# загрузка обученной модели с лучшей точностью
model.load_weights(path_to_results + '/best_model.h5')

(x_train, _), (x_test, y_test) = dataset.load_data()

#
# нормализация тестовых данных
#

if dataset == fashion_mnist:
    x_test_resized = np.zeros((x_test.shape[0], *image_shape))
    for i in range(x_test.shape[0]):
        x_test_resized[i] = cv2.resize(x_test[i], image_shape)
    x_test = x_test_resized.reshape((*x_test_resized.shape, 1))

    x_train_resized = np.zeros((x_train.shape[0], *image_shape))
    for i in range(x_train.shape[0]):
        x_train_resized[i] = cv2.resize(x_train[i], image_shape)
    x_train = x_train_resized.reshape((*x_train_resized.shape, 1))

    y_test = y_test.reshape((*y_test.shape, 1))

x_test = x_test.astype('float32') / 255
x_train = x_train.astype('float32') / 255

# загрузка пиксельного среднего, полученного при обучении
with open(path_to_results + '/pixelmean.pkl', 'rb') as f:
    x_train_mean = pickle.load(f)
x_test -= x_train_mean

y_test = keras.utils.to_categorical(y_test, num_classes)

# вычисление точности и ошибки модели
scores = model.evaluate(x_test, y_test, verbose=1)
# тестовое предсказывание классов, необходимое для построение метрик и графиков
y_predicted = model.predict(x_test, verbose=1)
test_labels, predicted_labels = y_test.argmax(axis=1), y_predicted.argmax(axis=1)


# подготовка папки для сохранения резульаттов
metrics_folder = './metrics/' + model_folder
if not os.path.isdir(metrics_folder):
    os.makedirs(metrics_folder)

# вычисление метрик
loss, accuracy = scores[0], scores[1]
conf_matrix = metrics.confusion_matrix(test_labels, predicted_labels)
precision = metrics.precision_score(test_labels, predicted_labels, average="macro")
recall = metrics.recall_score(test_labels, predicted_labels, average="macro")

# сохранение численных метрик
with open(metrics_folder + '/scores.txt', 'w') as f:
    lines = [f'Loss: {loss}\n',
             f'Accuracy: {accuracy}\n',
             f'Precision: {precision}\n',
             f'Recall: {recall}\n']
    f.writelines(lines)

# построение матрицы неточностей
cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
# plt.figure(figsize = (12,5))
sn.heatmap(cm, annot=False, linewidths=.5)
plt.savefig(metrics_folder + '/confusion_matrix.jpg')
plt.show()

# загрузка истории обучения модели
with open(f'{path_to_results}/history.pkl', 'rb') as f:
    history = pickle.load(f)
# построение графика обучения модели по точности распознавания на тестовой выборке и обучающей
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(metrics_folder + '/accuracy.jpg')
plt.show()
# построение графика обучения модели по ошибке распознавания на тестовой выборке и обучающей
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(metrics_folder + '/loss.jpg')
plt.show()
