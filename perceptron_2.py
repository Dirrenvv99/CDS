import numpy as np

class Perceptron():
    def __init__(self, N, activation, lr=0.1):
        self.w = np.random.choice(101, N) / 50 - 1
        self.activation = activation
        self.lr = lr

    def forward(self, X):
        out = []
        for xi in X:
            out.append(np.dot(xi, self.w))
        return np.array(self.activation(out))

    def update(self, X, Y):
        # print(self.w)
        I = []
        for i in range(len(X)):
            I.append(X[i]*Y[i])
        I = np.array(I)
        # print(I.shape)
        S = I[np.dot(I, self.w) < 0]
        self.w += self.lr * np.sum(S, axis=0)

        # print(len(S))

        return len(S) != 0


def main():
    N = 50
    for p in range(10, 121, 10):
        nruns = 100

        result = []
        result_classification = []

        for _ in range(nruns):
            P = Perceptron(N, lambda x : np.array([-1 if i < 0 else 1 for i in x]))
            X = np.random.choice([-1,1], (p, N))
            Y = np.random.choice([-1,1], p)
            i = 1
            while i < 1001 and  P.update(X,Y):
                i += 1
            result.append(i)

            Y_hat = P.forward(X)

            result_classification.append(np.sum(Y!=Y_hat))

        result = np.array(result)

        print(f"{p}: fraction {len(result[result < 1000]) / nruns} ")
        print("class_mean: \t", np.mean(result_classification))
        print("class_std: \t", np.std(result_classification))
        print("runs_mean: \t", np.mean(result))
        print("runs_std: \t", np.std(result))
        print('\n')


if __name__ == "__main__":
    main()
