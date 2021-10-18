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
# 1.
'''
Parameters (standard):
- Optimizer = Adam
- Learning rate = 1e-3
- Loss = Sparse Categorical CrossEntropy, Tensorflow states: Use this crossentropy loss function when there are two or more label classes.
- Metrics = Accuracy
- Epochs = 200

Discussion:
For 0 hidden layers (model_0), we've tried epochs 20,50,100,200 and learning rates 1e-{1,2,3,4,5} but it did not reach maximum test accuracy
For epochs=200 and lr=1e-3, the test accuracy was: 0.2969000041484833
The plot of the learning curve can be found in ex_1_hiddenlayers_0.png
Total params = 30730

For 1 hidden layer (model_1), the ideal learning rate was 1e-4
The test accuracy was: 0.5141000151634216
The model largely overfits, since the training accuracy increases while the validation accuracy slightly decreases
Early stopping would therefore be convenient.
The plot of the learning curve can be found in ex_1_hiddenlayers_1.png
Total params = 616,610

We've also tried a learning rate=1e-3 with epochs=400 but the training accuracy did not go above 0.6 
while the validation accuracy could not get above 0.5.

For 2 hidden layers (model_2), the learning rate=1e-4 and epochs=200.
The plot can be found in ex_1_hiddenlayers_2.png
Total params = 937460
Early stopping would be very convenient since the validation accuracy decreases rapidly
Test accuracy = 0.49619999527931213
Best validation accuracy = 0.5333

'''
# ----------------------------
# 2. 
'''
Parameters:
- Linear layer of 10000 hidden units
- Optimizer = Adam
- Learning rate = 1e-3
- Loss = Sparse Categorical CrossEntropy
- Metrics = Accuracy
- Epochs = 1000
- Model = model_1_large

Discussion, comparison with [https://arxiv.org/pdf/1611.03530.pdf]:
Test accuracy: 50.51
Train accuracy: 100
Learning rate: 1e-2

Our results for 1000 epochs with layer of 10000 units.
Learning rate = 1e-4

Params = 30830010
After two hours of running: 
- elapsed epochs = 68
- highest training accuracy: 0.9140
- highest validation accuracy: 0.5630
The validation accuracy is higher in comparison with the baseline
However, the model is largely overfitting on the data given that the 
train accuracy improves while the validation accuracy reaches its maximum ≈ 0.56

All results can be found in ex_2_log.log in folder logs
'''