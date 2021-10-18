import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models


# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# MLP, run with perceptron.py
model_ex_4 = models.Sequential([
    layers.Flatten(),
    layers.Dense(40, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(10, activation='softmax')
    ])

# CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


# -------------------------------
# 3.
'''
activations = ['relu', 'tanh']
optimizers = ['adam', 'adagrad', 'sgd']


Params = 122,570
Difference with ReLU and tanh:
ReLU: Test: loss=0.8949, accuracy=0.7075
tanh: Test: loss=0.9759, accuracy=0.6762
Plot for tanh can be found in ex_3_tanh.png
The reason why using tanh performs worse is because [UITLEG]

Optimizers:
- Adam: loss=0.8949, accuracy=0.7075
- Adagrad: loss=1.4249, accuracy=0.4919
- SGD: loss=1.1328, accuracy=0.6089
Plots can be found in:
- ex_3_adagrad.png
- ex_3_sgd.png
'''

# -------------------------------
# 4.
'''
Parameters both networks:
- epochs = 10
- learning rate = 1e-3
- loss = sparse_categorical_crossentropy
- optimizer = Adam

Number of parameters:
CNN: 122,570
MLP: 123,950

MLP results for 10 epochs with a learning rate of 1e-3:
loss=1.7072, accuracy=0.3809
plot: ex_4_MLP_10epochs.png
CNN was:
loss=0.8949, accuracy=0.7075

MLP for 100 epochs with a learning rate of 1e-4:
loss=1.4717, accuracy=0.4787
A plot can be found in the plots folder: ex_4_MLP_100epochs.png
Results can be found in ex_4_log.log in logs folder.
The model was largely overfitting.

The CNN clearly outperforms the MLP. This is because the MLP includes too many 
parameters leading to redudancy and overfitting. CNN is better for image classification 
due to less parameters, whereas the MLP is fully connected making the network unseemingly large.
'''

