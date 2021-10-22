import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


class GMM:
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

        plot(X, gamma, 2, ['darkred', 'darkblue'], sigma, mu, title='Initialization')

        for iter in range(max_iter):
            # 2. E-step
            print(f'E step {iter+1}')
            for k in range(self.K):
                for n in range(len(X)):
                    d = np.sum([pi[j] * self.gaussian(X[n], mu[j], sigma[j], j) for j in range(self.K)])
                    gamma[n, k] = pi[k] * self.gaussian(X[n], mu[k], sigma[k], self.K) / d
            
            if iter == 0:
                plot(X, gamma, 2, ['darkred', 'darkblue'], sigma, mu, f'Iteration {iter} E step')

            # 3. M-step
            print(f'M step {iter+1}')
            for k in range(self.K):
                N_k = sum(gamma[:,k])
                mu[k] = 1 / N_k * np.sum([gamma[n,k] * X[n] for n in range(N)], axis=0)
                sigma[k] = 1 / N_k * np.sum([gamma[n,k] * (np.matmul((X[n] - mu[k])[np.newaxis].T, (X[n]-mu[k])[np.newaxis])) for n in range(N)], axis=0)
                pi[k] = N_k / N

            if iter in [0,3,5,16]:
                plot(X, gamma, 2, ['darkred', 'darkblue'], sigma, mu, f'Iteration {iter}')

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


def plot(X, gamma, K, colors, sigma, mu, title):
    # get the assigned labels
    labels = np.argmax(gamma, axis = 1)
    print(mu.shape)
    print(sigma.shape)

    fig, ax = plt.subplots()
    for k in range(K):
        cluster = np.where(labels == k)
        x = X[cluster]
        ax.scatter(x[:,0], x[:,1], c = colors[k], label = k, alpha = 0.7)

        pearson = sigma[k][0][1]/np.sqrt(sigma[k][0][0] * sigma[k][1][1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        scale_x = np.sqrt(sigma[k][0][0])
        scale_y = np.sqrt(sigma[k][1][1])
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, edgecolor=colors[k], lw=2, 
                        facecolor='none', label='ellipse')

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mu[k][0], mu[k][1])
 
        ellipse.set_transform(transf + ax.transData)

        ax.add_patch(ellipse)

    plt.title(title)
    plt.xlabel('eruptions')
    plt.ylabel('waiting')
    ax.legend()
    plt.show()
