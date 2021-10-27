import numpy as np
import matplotlib.pyplot as plt

from mlxtend.data import loadlocal_mnist
from pathlib import Path


# EXERCISE 2.3 MIXTURE MODELS and EM
def load_data(images_path, labels_path):
    parent = Path.cwd().parent
    week_4 = f'{parent}/Week 4'

    # Load data
    x, y = loadlocal_mnist(
        images_path=f'{week_4}/{images_path}',
        labels_path=f'{week_4}/{labels_path}'
    )
    x = x / 255

    return x, y


class MMM:
    # Slides: 192-195
    def __init__(self, clusters):
        self.K = clusters

    def posterior(self, x, mu, pi):
        p_k = 1/self.K
        p_x_k = np.zeros((self.K, self.d))
        for k in range(self.K):
            p_x_given_k = np.prod(np.dot(mu[k]**x, (1-mu[k])**(1-x)))
            p_x_k[k] = pi[k] * p_x_given_k

        return np.sum(p_k * p_x_k)

    def ku_argmax(self, x, mu, pi):
        p_k = 1/self.K
        p_x_k = np.zeros((self.K, 1))
        for k in range(self.K):
            p_x_given_k = np.prod(np.dot(mu[k]**x, (1-mu[k])**(1-x)))
            p_x_k[k] = p_x_given_k

        return int(np.argmax(p_k * p_x_k))

    def log_likelihood(self, X, mu, pi):
        logs_pi = 0
        logs_data = 0
        for x in X:
            ku = self.ku_argmax(x, mu, pi)
            logs_pi += pi[ku]
            logs_data += np.prod(np.dot(mu[ku]**x, (1-mu[ku])**(1-x)))

        return np.log(logs_pi) + np.log(logs_data)

    def fit(self, X, max_iter=100, min_diff=0.01):
        # Initialization N, d
        self.N, self.d = X.shape
        # MNIST data N = 60000, K = 10, d = 784.
        print(f'X: {X.shape}, N: {self.N}, K: {self.K}, d: {self.d}')

        # Random initialization mu, pi
        random_row = np.random.randint(low=0, high=self.N, size=self.K)
        mu = np.array([X[row_index, :] for row_index in random_row])
        pi = np.full(self.K, 1/self.K)

        print(f'Means: {mu.shape}, Coefficients: {pi.shape}')

        print("Initial log likelihood")
        llh = [self.log_likelihood(X, mu, pi)]
        print(f'[{len(llh) -1}]: log likelihood = {llh[-1]}')

        # Clustering algorithm
        k_u = np.empty((self.N))
        
        for iter in range(max_iter):
            for n in range(self.N):
                k_u[n] = self.ku_argmax(X[n], mu, pi)

            for k in range(self.K):
                X_k = np.array([idx for idx, x in enumerate(k_u) if x == k])
                N_k = X_k.shape[0]
                # The sum of pi is always 1.0
                pi[k] = N_k / self.N
                for j in range(self.d):
                    mu[k, j] = 1/N_k * np.sum([X[u, j] for u in X_k])

            # Evaluate log likelihood
            llh.append(self.log_likelihood(X, mu, pi))
            diff = np.abs(llh[-2] - llh[-1])
            print(f'[{len(llh) -1}]: log likelihood = {llh[-1]}')
            print(f'[{len(llh) -1}]: difference = {diff}')
            # This is to stop the algorithm from continuing for too long
            if diff < min_diff:
                print(f'[{len(llh) -1}]: Changes are too small to continue')
                break

            # Plot after 10 and 20 epochs
            if iter + 1 in [10, 20]:
                plot(mu, pi, self.K)

        return mu, pi, llh


def main(clusters=10, iterations=50, diff_likelihood=0.0001):
    train_x, train_t = load_data(
        'data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte'
    )

    test_x, test_t = load_data(
        'data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte'
    )

    mmm = MMM(clusters)
    X = np.array(train_x)
    mu, pi, llh = mmm.fit(X, iterations, diff_likelihood)
    plot(mu, pi, clusters)


def plot(mu, pi, clusters):
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(1.5*cols, 2*rows))
    for k in range(clusters):
        ax = axes[k // cols, k % cols]
        ax.imshow(mu[k, :].reshape(28, 28), cmap='gray')
        ax.set_title(f'{round(pi[k], 3)}')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
