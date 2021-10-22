import numpy as np
from gmm import GMM
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def load_data():
    """
    A data frame with 272 observations on 2 variables.
    eruptions  Eruption time in mins
    waiting    Waiting time to next eruption
    """
    with open("data.txt", 'r') as f:
        lines = f.readlines()

    # print(lines[1:])
    data = [[float(f) for f in l.split()] for l in lines[1:]]

    return np.array([[e[1], e[2]] for e in data])


def main():
    # Use GMM from gmm.py
    data = load_data()
    gmm = GMM(2)
    mu, sigma, gamma, llh = gmm.fit(data, 50, 0.001)


if __name__ == "__main__":
    main()
