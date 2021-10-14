import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from tqdm import tqdm

def load_data(images_path, labels_path):
    # Load data
    x, y = loadlocal_mnist(
        images_path=images_path, 
        labels_path=labels_path
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


def sigma(x):
    return 1./(1.+np.exp(-x))


def error(w, x, t, lam=0):
    n = len(x)

    y = sigma(np.dot(w, x.T))
    E = -(np.dot(t, np.log(y)) + np.dot((1-t), np.log(1-y)))/n + lam/2 * (w**2).sum()

    return E


def error_gradient(w, x, t, lam=0):
    n, d = x.shape

    y = sigma(np.dot(w, x.T))
    E_grad = (np.dot((y-t), x) + lam * w)/n

    return E_grad


def hessian(w, x, t, lam=0): # TODO Niet zeker nu wel d,d shape
    n, d = x.shape
    y = sigma(np.dot(w, x.T))[:,np.newaxis]

    h = (np.dot(np.dot(x.T, y), np.dot(x.T, (1-y)).T) + lam*np.identity(d))/n

    return h


def gradient_descent(w, train_x, train_t, test_x, test_t, learning_rate, iter):
    train_error, test_error = [], []

    for _ in tqdm(range(iter)):
        w = w - learning_rate * error_gradient(w, train_x, train_t)
        train_error.append(error(w, train_x, train_t))
        test_error.append(error(w, test_x, test_t))

    return w, train_error, test_error


def gradient_descent_momentum(w, train_x, train_t, test_x, test_t, learning_rate, momentum, iter):
    train_error, test_error = [], []
    delta = 0

    for _ in tqdm(range(iter)):
        delta = -learning_rate * error_gradient(w, train_x, train_t) + momentum * delta
        w = w + delta
        train_error.append(error(w, train_x, train_t))
        test_error.append(error(w, test_x, test_t))

    return w, train_error, test_error


def weight_decay(w, train_x, train_t, test_x, test_t, learning_rate, lam, iter):
    train_error, test_error = [], []

    for _ in tqdm(range(iter)):
        w = w - learning_rate * error_gradient(w, train_x, train_t, lam)
        train_error.append(error(w, train_x, train_t))
        test_error.append(error(w, test_x, test_t))

    return w, train_error, test_error


def newton_method(w, train_x, train_t, test_x, test_t, learning_rate, lam, iter):
    train_error, test_error = [], []

    for _ in tqdm(range(iter)): # TODO FIX
        # print(w.shape)
        # print(hessian(w, train_x, train_t, lam).shape)
        # print(np.linalg.inv(hessian(w, train_x, train_t, lam)).shape)
        # print(error_gradient(w, train_x, train_t, lam).shape)

        w = w - learning_rate * np.dot(
            np.linalg.inv(hessian(w, train_x, train_t, lam)),
            error_gradient(w, train_x, train_t, lam))


        train_error.append(error(w, train_x, train_t))
        test_error.append(error(w, test_x, test_t))

    return w, train_error, test_error


def line_search(w, train_x, train_t, test_x, test_t, iter):
    """
    Select lambda based on whcih gets the lowest error
    """
    train_error, test_error = [], []
    lam = 0.5

    for _ in tqdm(range(iter)):
        grad = error_gradient(w, train_x, train_t, lam)
        w1 = w - 2*lam * grad
        w2 = w - lam * grad
        w3 = w - .5*lam * grad

        e1 = error(w1, train_x, train_t)
        e2 = error(w2, train_x, train_t)
        e3 = error(w3, train_x, train_t)

        if e1 < e2 and e1 < e3:
            lam = 2*lam
            w = w1
            train_error.append(e1)
        elif e3 < e2:
            lam = .5*lam
            w = w3
            train_error.append(e3)
        else:
            w = w2
            train_error.append(e2)

        test_error.append(error(w, test_x, test_t))

    return w, train_error, test_error


def conjugate(w, train_x, train_t, test_x, test_t, iter):
    train_error, test_error = [], []
    lam = 0.5
    d = 0
    prev_eg = error_gradient(w, train_x, train_t)

    for _ in tqdm(range(iter)):
        eg = error_gradient(w, train_x, train_t)
        beta = np.dot(eg-prev_eg, eg)/np.linalg.norm(prev_eg)
        d = -eg + beta*d
        prev_eg = eg
        print(lam)

        w1 = w + 2*lam * d
        w2 = w + lam * d
        w3 = w + .5*lam * d

        e1 = error(w1, train_x, train_t)
        e2 = error(w2, train_x, train_t)
        e3 = error(w3, train_x, train_t)

        if e1 < e2 and e1 < e3:
            lam = 2*lam
            w = w1
            train_error.append(e1)
        elif e3 < e2:
            lam = .5*lam
            w = w3
            train_error.append(e3)
        else:
            w = w2
            train_error.append(e2)
        
        test_error.append(error(w, test_x, test_t))

    return w, train_error, test_error


def stochastic_gradient_descent(w, train_x, train_t, test_x, test_t, learning_rate, iter):
    pass
    # train_error, test_error = [], [] #TODO deze helemaal

    # for _ in tqdm(range(iter)):
    #     w = w - learning_rate * error_gradient(w, train_x, train_t)
    #     train_error.append(error(w, train_x, train_t))
    #     test_error.append(error(w, test_x, test_t))

    # return w, train_error, test_error


def plot_error(title, train_error, test_error):
    plt.plot(train_error, label='train error')
    plt.plot(test_error, label='test error')
    plt.title(title)
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()


def main():
    np.random.seed(0)

    train_x, train_t = load_data(
        'data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte'
        )
    
    test_x, test_t = load_data(
        'data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte'
        )

    print('shape train_x: %s x %s' % (train_x.shape[0], train_x.shape[1]))
    print('shape train_t: %s' % (train_t.shape))

    # Initialize weights between [-1, 1]
    w = np.random.choice(20000, train_x.shape[1])/10000 - 1

    # Gradient descent ------------------------------------
    # learning_rate = 0.9
    # w, train_error, test_error = gradient_descent(
    #     w, train_x, train_t, test_x, test_t, learning_rate, 10000)
    # print(f"FINAL ERROR: {error(w, train_x, train_t)} and {error(w, test_x, test_t)}")
    # plot_error("Gradient Descent", train_error, test_error)

    # Logistic regression with Momentum -------------------
    # learning_rate = 0.9
    # momentum = 0.5
    # w, train_error, test_error = gradient_descent_momentum(
    #     w, train_x, train_t, test_x, test_t, learning_rate, momentum, 10000)
    # print(f"FINAL ERROR: {error(w, train_x, train_t)} and {error(w, test_x, test_t)}")
    # plot_error("Gradient Descent", train_error, test_error)

    # Logistic regression with weight decay ---------------
    # learning_rate = 0.9
    # lam = 0.1
    # w, train_error, test_error = weight_decay(
    #     w, train_x, train_t, test_x, test_t, learning_rate, lam, 5340)
    # print(f"FINAL ERROR: {error(w, train_x, train_t)} and {error(w, test_x, test_t)}")
    # plot_error("Gradient Descent", train_error, test_error)

    # Logistic regression with Newton method -------------- # TODO DOES NOT WORK
    # learning_rate = 0.9
    # lam = 0.1
    # w, train_error, test_error = newton_method(
    #     w, train_x, train_t, test_x, test_t, learning_rate, lam, 10)
    # print(f"FINAL ERROR: {error(w, train_x, train_t)} and {error(w, test_x, test_t)}")
    # plot_error("Gradient Descent", train_error, test_error)

    # Logistic regression with line search ---------------- # TODO andere line search voor deze en de volgende?
    # w, train_error, test_error = line_search(
    #     w, train_x, train_t, test_x, test_t, 224)
    # print(f"FINAL ERROR: {error(w, train_x, train_t)} and {error(w, test_x, test_t)}")
    # plot_error("Gradient Descent", train_error, test_error)

    # Logistic regression with conjugation ----------------
    # w, train_error, test_error = conjugate(
    #     w, train_x, train_t, test_x, test_t, 104)
    # print(f"FINAL ERROR: {error(w, train_x, train_t)} and {error(w, test_x, test_t)}")
    # plot_error("Gradient Descent", train_error, test_error)

    # Stochastic gradient descent -------------------------
    # learning_rate = 0.9
    # w, train_error, test_error = stochastic_gradient_descent(
    #     w, train_x, train_t, test_x, test_t, learning_rate, 5000)
    # print(f"FINAL ERROR: {error(w, train_x, train_t)} and {error(w, test_x, test_t)}")
    # plot_error("Gradient Descent", train_error, test_error)



if __name__ == "__main__":
    main()
