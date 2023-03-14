import numpy as np
class Classifier:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X_train, y_train):
        self.weights = np.zeros((self.n_classes, X_train.shape[1]))
        for c in range(self.n_classes):
            y_train_c = np.where(y_train == c, 1, -1)
            self.weights[c] = self.sgd(X_train, y_train_c)

    def predict(self, X_test):
        y_pred = np.argmax(X_test.dot(self.weights.T), axis=1)
        return y_pred

    def sgd(self, X_train, y_train):
        lr = 0.1
        n_epochs = 50
        weights = np.zeros(X_train.shape[1])
        for epoch in range(n_epochs):
            for i, x in enumerate(X_train):
                if y_train[i] * np.dot(x, weights) < 1:
                    weights = weights + lr * ((y_train[i] * x) + (-2 * (1 / n_epochs) * weights))

                else:
                    pass
        return weights