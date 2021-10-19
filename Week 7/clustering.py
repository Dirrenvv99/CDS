from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from numpy.core.shape_base import stack
from numpy.lib.shape_base import dstack
from tqdm import tqdm


def load_data(images_path, labels_path):
    # Load data
    parent_dir = Path.cwd().parent
    week_4 = f'{parent_dir}/Week 4/'

    x, y = loadlocal_mnist(
        images_path=f'{week_4}/{images_path}', 
        labels_path=f'{week_4}/{labels_path}'
    )

    x = x / 255

    return x, y


class GMM:
    # TODO: https://towardsdatascience.com/gaussian-mixture-models-implemented-from-scratch-1857e40ea566
    # Extra: https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/
    def __init__(self, clusters):
        self.K = clusters

    def gaussian(self, x, mu, sigma, k):
        # Values are split up to get more accurate error detection
        d = (x - mu).T
        s1 = np.linalg.inv(sigma)
        s = np.matmul(d, s1)
        e = np.matmul(s, d)
        n = np.exp(-1/2*e)
        d = np.sqrt((2*np.pi)**k * np.linalg.det(sigma))
        return n/d

    def log_likelihood(self, X, mu, sigma, pi):
        logs = [np.log(np.sum([pi[k] * self.gaussian(x, mu[k], sigma[k], k) for k in range(self.K)])) for x in X]
        return np.sum(logs)

    # EM-Algorithm
    def fit(self, X, max_iter=100, min_diff=0.01):
        # 1. Initialization
        # https://link.springer.com/article/10.3758/s13428-015-0697-6
        print(f'Data: {X.shape}')
        N, d = X.shape
        random_row = np.random.randint(low=0, high=N, size=self.K)
        mu = np.array([X[row_index,:] for row_index in random_row])
        sigma = np.array([np.cov(X.T) for _ in range(self.K)])
        pi = np.full(self.K, 1/self.K)

        print(f'Means: {mu.shape}, Covariances: {sigma.shape}, Coefficients: {pi.shape}')

        llh = [self.log_likelihood(X, mu, sigma, pi)]
        print(f'[{len(llh) -1}]: log likelihood = {llh[-1]}')

        gamma = np.zeros((N, self.K))

        for iter in range(max_iter):
            # 2. E-step
            print(f'E step {iter+1}')
            for k in range(self.K):
                for n in range(len(X)):
                    d = np.sum([pi[j] * self.gaussian(X[n], mu[j], sigma[j], j) for j in range(self.K)])
                    gamma[n, k] = pi[k] * self.gaussian(X[n], mu[k], sigma[k], self.K) / d

            # 3. M-step
            print(f'M step {iter+1}')
            for k in range(self.K):
                N_k = sum(gamma[:,k])
                mu[k] = 1 / N_k * np.sum([gamma[n,k] * X[n] for n in range(N)], axis=0)
                sigma[k] = 1 / N_k * np.sum([gamma[n,k] * (np.matmul((X[n] - mu[k])[np.newaxis].T, (X[n]-mu[k])[np.newaxis])) for n in range(N)], axis=0)
                pi[k] = N_k / N

            # 4. Evaluate log likelihood
            llh.append(self.log_likelihood(X, mu, sigma, pi))
            diff = np.abs(llh[-2] - llh[-1])
            print(f'[{len(llh) -1}]: log likelihood = {llh[-1]}')
            print(f'[{len(llh) -1}]: difference = {diff}')
            # This is to stop the algorithm from continueing for too long
            if diff < min_diff:
                print(f'[{len(llh) -1}]: Changes are too small to continue')
                break

        return mu, sigma, gamma, llh


def main():
    train_x, train_t = load_data(
    'data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte'
    )

    test_x, test_t = load_data(
    'data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte'
    )

    gmm = GMM(10)
    X = np.array(train_x)
    gmm.fit(X)


if __name__=="__main__":
    main()
