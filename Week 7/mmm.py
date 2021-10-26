import numpy as np
from mlxtend.data import loadlocal_mnist
from pathlib import Path

from numpy.core.defchararray import add


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
            p_x_given_k = np.array([mu[k,j]**x[j]*(1-mu[k,j])**(1-x[j]) for j in range(self.d)])
            p_x_k[k] = pi[k] * np.prod(p_x_given_k)

        return p_k * np.sum(p_x_k)
    
    def ku_argmax(self, x, mu, pi):
        p_k = 1/self.K
        p_x_k = np.zeros((self.K, 1))
        for k in range(self.K):
            p_x_given_k = np.array([mu[k,j]**x[j]*(1-mu[k,j])**(1-x[j]) for j in range(self.d)])
            p_x_k[k] = np.prod(p_x_given_k)

        return np.argmax(p_k * p_x_k)

    def log_likelihood(self, X, mu, pi):
        # xu is data point
        logs_pi = []
        logs_data = []
        for x in X:
            ku = self.ku_argmax(x, mu, pi)
            logs_pi.append(pi[ku])
            for j in range(self.d):
                logs_data.append(x[j]*np.log(mu[ku,j])+(1-x[j])*np.log(1-mu[ku,j]))

        return np.sum(logs_pi) + np.sum(logs_data)

    def fit(self, X, max_iter=100, min_diff=0.01):
        # 1. Initialization
        X = X[:100]
        self.N, self.d = X.shape
        # MNIST data N = 60000, K = 10, d = 784.
        print(f'X: {X.shape}, N: {self.N}, K: {self.K}, d: {self.d}')

        # Random initialization
        random_row = np.random.randint(low=0, high=self.N, size=self.K)
        mu = np.array([X[row_index,:] for row_index in random_row])
        pi = np.full(self.K, 1/self.K)

        print(f'Means: {mu.shape}, Coefficients: {pi.shape}')

        llh = [self.log_likelihood(X, mu, pi)]
        print(f'[{len(llh) -1}]: log likelihood = {llh[-1]}')

        k_u = np.empty((self.N))
        for iter in range(max_iter):

            for n in range(self.N):
                k_u[n] = self.ku_argmax(X[n], mu, pi)

            for k in range(self.K):
                X_k = np.array([idx for idx, x in enumerate(k_u) if x == k])
                N_k = X_k.shape[0]
                pi[k] = N_k / self.N
                for j in range(self.d):
                    mu[k,j] = 1/N_k * np.sum([X[u,j] for u in X_k])

            # 4. Evaluate log likelihood
            llh.append(self.log_likelihood(X, mu, pi))
            diff = np.abs(llh[-2] - llh[-1])
            print(f'[{len(llh) -1}]: log likelihood = {llh[-1]}')
            print(f'[{len(llh) -1}]: difference = {diff}')
            # This is to stop the algorithm from continueing for too long
            if diff < min_diff:
                print(f'[{len(llh) -1}]: Changes are too small to continue')
                break

        return mu, llh


def main():
    train_x, train_t = load_data(
    'data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte'
    )

    test_x, test_t = load_data(
    'data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte'
    )

    mmm = MMM(10)
    X = np.array(train_x)
    mu, llh = mmm.fit(X, 100, 0.001)


if __name__=="__main__":
    main()
