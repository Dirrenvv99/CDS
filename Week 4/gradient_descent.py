# http://yann.lecun.com/exdb/mnist/
# ! pip3 install mlxtend
import numpy as np
from mlxtend.data import loadlocal_mnist


# Load data
train_X, train_y = loadlocal_mnist(
    images_path='data/train-images-idx3-ubyte', 
    labels_path='data/train-labels-idx1-ubyte'
)

test_X, test_y = loadlocal_mnist(
    images_path='data/t10k-images-idx3-ubyte', 
    labels_path='data/t10k-labels-idx1-ubyte'
)

#print('Dimensions train data: %s x %s' % (train_X.shape[0], train_X.shape[1]))

# --------------------------------------
# Gradient methods
train_X_3 = np.array([val for idx, val in enumerate(train_X) if train_y[idx] == 3])
train_X_7 = np.array([val for idx, val in enumerate(train_X) if train_y[idx] == 7])

# print('Dimensions X_3: %s x %s' % (train_X_3.shape[0], train_X_3.shape[1]))
# print('Dimensions X_7: %s x %s' % (train_X_7.shape[0], train_X_7.shape[1]))

# --------------------------------------
# Gradient descent

# Hyperparamaters
learning_rate = 0.001 # n

# 0: x = 3, 1: x = 7
train_X = np.concatenate((train_X_3, train_X_7))
train_y = np.concatenate((np.zeros(train_X_3.shape[0]), np.ones(train_X_7.shape[0])))

print('Dimensions train_X: %s x %s' % (train_X.shape[0], train_X.shape[1]))
print('Dimensions train_y: %s' % (train_y.shape))

wi = np.random.choice([0,1], len(train_y))

def sigma(x):
    return (1+np.exp(-x))**-1


def logistic_regression(x):
    pass


def gradient_descent(xi,wi):
    pass


def plot():
    pass

# --------------------------------------
# Momentum

# --------------------------------------
# Weight decay


