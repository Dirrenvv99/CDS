# Adapted from the code on https://www.tensorflow.org/tutorials/images/cnn 
# from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models


# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
num_classes = 10

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Exercise 1
model_0 = models.Sequential([
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
    ])

model_1 = models.Sequential([
    layers.Flatten(),
    layers.Dense(200, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
    ])

model_2 = models.Sequential([
    layers.Flatten(),
    layers.Dense(300, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
    ])

# Exercise 2
model_1_large = models.Sequential([
    layers.Flatten(),
    layers.Dense(10000, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
    ])


# Run a certain model
def run(model, epochs=20, lr=1e-5):
    model.compile(optimizer=tf.keras.optimizer.Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    plot(history, test_loss, test_acc)


# Plot the learning curves
def plot(history, test_loss, test_acc):
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label = 'val_accuracy')
    plt.plot(test_loss, label='test_loss')
    plt.plot(test_acc, label = 'test_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

# ----------------------------
# NOTE!
# ANSWERS in answers.pdf
