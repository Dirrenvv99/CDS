# Adapted from the code on https://www.tensorflow.org/tutorials/images/cnn 
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt

# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

num_classes = 10

model_0 = models.Sequential([
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
    ])

model_1 = models.Sequential([
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
    ])

model_2 = models.Sequential([
    layers.Flatten(),
    layers.Dense(200, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
    ])

model_1_large = models.Sequential([
    layers.Flatten(),
    layers.Dense(10000, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
    ])


model_ex_4 = models.Sequential([
    layers.Flatten(),
    layers.Dense(400, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
    ])

model_ex_4.summary()


def run(model, epochs=20, lr=1e-5):
    model.compile(optimizer=tf.keras.optimizer.Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels))


    test_loss, test_acc = model.evaluate(test_images, test_labels)

    plot(history, test_loss, test_acc)


def plot(history, test_loss, test_acc):
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label = 'val_accuracy')
    plt.plot(test_loss, label='test_loss')
    plt.plot(test_acc, label = 'test_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

# ----------------------------
# 1.
'''
Parameters:
- Optimizer = Adam
- Learning rate = 1e-5
- Loss = Sparse Categorical CrossEntropy, Tensorflow states: Use this crossentropy loss function when there are two or more label classes.
- Metrics = Accuracy
- Epochs = 20

Q: Is early stopping convenient?
A: Yes it is

Discussion:


'''
# ----------------------------
# 2. 
'''
Parameters:
- Optimizer = Adam
- Learning rate = 1e-5
- Loss = Sparse Categorical CrossEntropy
- Metrics = Accuracy
- Epochs = 1000

Discussion, comparison with [https://arxiv.org/pdf/1611.03530.pdf]:

'''