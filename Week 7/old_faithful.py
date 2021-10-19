import numpy as np
from gmm import GMM
import matplotlib.pyplot as plt


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


def plot(X, gamma, K, colors):
    # get the assigned labels
    labels = np.argmax(gamma, axis = 1)

    fig, ax = plt.subplots()
    for k in range(K):
        cluster = np.where(labels == k)
        x = X[cluster]
        ax.scatter(x[:,0], x[:,1], c = colors[k], label = k, alpha = 0.7)

    plt.title('Old Faithful Data')
    plt.xlabel('eruptions')
    plt.ylabel('waiting')
    ax.legend()
    plt.show()

    # TODO: plot ellipse
    # TODO: recreate figure 11.11 : 6 plots


def main():
    # Use GMM from gmm.py
    data = load_data()
    gmm = GMM(2)
    mu, sigma, gamma, llh = gmm.fit(data, 30, 0.001)
    plot(data, gamma, 2, ['darkred', 'darkblue'])


if __name__ == "__main__":
    main()
