from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from tqdm import tqdm


def load_data(images_path, labels_path):
    # Load data
    parent_dir = Path.cwd().parent
    week_4 = f'{parent_dir}/Week 4/'

    x, y = loadlocal_mnist(
        images_path=f'{week_4}/{images_path}', 
        labels_path=f'{week_4}/{labels_path}'
    )

    # Take the 3's and 7's
    x_3 = np.array([val for idx, val in enumerate(x) if y[idx] == 3])
    x_7 = np.array([val for idx, val in enumerate(x) if y[idx] == 7])

    # 0: x = 3, 1: x = 7
    x = np.concatenate((x_3, x_7))
    t = np.concatenate((np.zeros(len(x_3)), np.ones(len(x_7))))

    x = x / 255                    # normalise x
    x = np.insert(x, 0, 1, axis=1) # append 1 for w_0

    return x, t


train_x, train_t = load_data(
    'data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte'
    )

test_x, test_t = load_data(
    'data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte'
    )


def clustering():
    # TODO: https://towardsdatascience.com/gaussian-mixture-models-implemented-from-scratch-1857e40ea566
    # Extra: https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/
    pass