import numpy as np
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
    # https://web.stanford.edu/~lmackey/stats306b/doc/stats306b-spring14-lecture3_scribed.pdf
    def __init__(self, clusters):
        self.K = clusters
    
    def multinomial(self, x, mu, k):
        return np.prod(mu**x)

    def log_likelihood(self, X, mu, pi):
        logs = [np.log(np.sum([pi[k] * self.multinomial(x, mu[k], k) for k in range(self.K)])) for x in X]
        return np.sum(logs)

    # EM-Algorithm
    def fit(self, X, max_iter=100, min_diff=0.01):
        # 1. Initialization
        print(f'Data: {X.shape}')
        N, d = X.shape
        random_row = np.random.randint(low=0, high=N, size=self.K)
        mu = np.array([X[row_index,:] for row_index in random_row])
        # sigma = np.array([np.cov(X.T) for _ in range(self.K)])
        pi = np.full(self.K, 1/self.K)

        print(f'Means: {mu.shape}, Coefficients: {pi.shape}')

        llh = [self.log_likelihood(X, mu, pi)]
        print(f'[{len(llh) -1}]: log likelihood = {llh[-1]}')

        gamma = np.zeros((N, self.K))

        for iter in range(max_iter):
            # 2. E-step
            print(f'E step {iter+1}')
            for k in range(self.K):
                for n in range(len(X)):
                    d = np.sum([pi[j] * self.multinomial(X[n], mu[j], j) for j in range(self.K)])
                    gamma[n, k] = pi[k] * self.multinomial(X[n], mu[k], self.K) / d

            # 3. M-step
            print(f'M step {iter+1}')
            for k in range(self.K):
                N_k = sum(gamma[:,k])
                mu[k] = 1 / N_k * np.sum([gamma[n,k] * X[n] for n in range(N)], axis=0)
                pi[k] = N_k / N

            # 4. Evaluate log likelihood
            llh.append(self.log_likelihood(X, mu, pi))
            diff = np.abs(llh[-2] - llh[-1])
            print(f'[{len(llh) -1}]: log likelihood = {llh[-1]}')
            print(f'[{len(llh) -1}]: difference = {diff}')
            # This is to stop the algorithm from continueing for too long
            if diff < min_diff:
                print(f'[{len(llh) -1}]: Changes are too small to continue')
                break

        return mu, gamma, llh


def main():
    train_x, train_t = load_data(
    'data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte'
    )

    test_x, test_t = load_data(
    'data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte'
    )

    mmm = MMM(10)
    X = np.array(train_x)
    mu, gamma, llh = mmm.fit(X, 100, 0.001)


if __name__=="__main__":
    main()