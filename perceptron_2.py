import numpy as np

class Perceptron():
    """
    Perceptron, a supervised learning algorithm for binary classifiers
    """
    def __init__(self, N, activation_func, learning_rate=0.1):
        # Random weight initialization between [-1, 1]
        self.weights = np.random.choice(101, N) / 50 - 1
        self.activation_func = activation_func
        self.learning_rate = learning_rate

    def predict(self, data):
        """
        Predict the binary class using the data and weights
        """
        out = []
        for xi in data:
            out.append(np.dot(xi, self.weights))
        return np.array(self.activation_func(out))

    def update(self, X, Y):
        """
        Update the weights using the data and the corresponding class values.
        Returns a boolean, which is true if there was anything left to update
        """
        I = []
        for i in range(len(X)):
            I.append(X[i]*Y[i])
        I = np.array(I)

        S = I[np.dot(I, self.weights) < 0]
        self.weights += self.learning_rate * np.sum(S, axis=0)

        return len(S) != 0


def main():
    N = 50      # Number of dimensions
    nruns = 100 # Number of runs
    for p in range(10, 121, 10):
        result = []
        result_classification = []

        for _ in range(nruns):
            # Generate the data
            X = np.random.choice([-1,1], (p, N))
            Y = np.random.choice([-1,1], p)

            P = Perceptron(N, lambda x : np.array([-1 if i < 0 else 1 for i in x]))

            # Update the weights up to 1000 times,
            # otherwise the algorithm does not converge (Exercise 2)
            i = 1
            while i < 1001 and P.update(X,Y):
                i += 1
            result.append(i)

            # Evaluate the classification performance using the errors
            Y_hat = P.predict(X)
            result_classification.append(np.sum(Y!=Y_hat))

        result = np.array(result)

        print(f"N = {N},\tP = {p}: fraction {len(result[result < 1000]) / nruns} ")
        print("class_mean: \t", np.mean(result_classification))
        print("class_std: \t", np.std(result_classification))
        print("runs_mean: \t", np.mean(result))
        print("runs_std: \t", np.std(result))
        print('\n')


if __name__ == "__main__":
    main()
