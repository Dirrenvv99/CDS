import numpy as np
import pandas as p


# epsilon function imported from exercise 4a
def epsilon(N,P, delta):
    if P <= N:
        return np.sqrt(8 * ( (2*P + 2)* np.log(2) - np.log(delta))/P)
    else:
        return np.sqrt((8 * ( N *  np.log((2*P)/N) + N - np.log(delta/4)))/P)


# Perceptron class imported from exercise 2
class Perceptron():
    def __init__(self, w, activation, lr=0.1):
        self.w = w
        self.activation = activation
        self.lr = lr

    def forward(self, X):
        out = []
        for xi in X:
            out.append(np.dot(xi, self.w))
        return np.array(self.activation(out))

    def update(self, X, Y):
        I = []
        for i in range(len(X)):
            I.append(X[i]*Y[i])
        I = np.array(I)
        S = I[np.dot(I, self.w) < 0]
        self.w += self.lr * np.sum(S, axis=0)

        return len(S) != 0


# choose N size
N = 10

# define weights
w_0 = np.random.randn(N)

# define test teacher
Xi_test = np.random.choice([-1,1], (10000, N))
teacher_P = Perceptron(w_0, lambda x : np.array([-1 if i < 0 else 1 for i in x]))
out_test = teacher_P.forward(Xi_test)

P_samples = [10,50,100,500,1000,90000]

def train(nruns=100):
    result_P_samples = []

    for P in P_samples:
        # generate input data
        Xi = np.random.choice([-1,1], (P, N))
        out = teacher_P.forward(Xi)

        result_classification = []
        for _ in range(nruns):
            # generate random weights and pass to perceptron
            w_student = np.random.randn(N)
            P = Perceptron(w_student, lambda x : np.array([-1 if i < 0 else 1 for i in x]))

            # Update perceptron
            while P.update(Xi, out):
                continue

            Y_hat = P.forward(Xi_test)

            result_classification.append(np.sum(out_test!=Y_hat)/len(Y_hat))

        result_P_samples.append(np.mean(result_classification))

    return result_P_samples


# Print table for generalization error and epsilon bound
def table(estimates, delta=0.01):
    for idx, P in enumerate(P_samples):
        eps = epsilon(N, P, delta)
        table = p.DataFrame(np.array([estimates[idx], eps]))
        print(table)


def main():
    samples = train()
    table(samples)


main()


# EXPLANATION AND RESULTS
'''
0 = generalization error
1 = epsilon generalization bound

for P = 100, the generalization error already goes below
0.1, but the epsilon generalization bound is way too conservative,
because we found P = 102938 (N=10) to give epsilon ≈ 0.1


          0
0  0.356080
1  3.985414
          0
0  0.094651
1  2.498554
          0
0  0.009104
1  1.917264
          0
0  0.008417
1  0.996339
         0
0  0.00000
1  0.74283
'''