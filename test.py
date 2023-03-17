#  testing out first custom Classifier



import numpy as np
from data_splitter import Train_Test



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cross-entropy loss function
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    loss = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Define the gradient descent optimizer
def gd_optimizer(grad, learning_rate):
    weights_update = -learning_rate * grad
    return weights_update

# Define the classifier
class Classifier:
    def __init__(self, n_classes):
        self.n_classes = n_classes




    def fit(self, X_train, y_train,n_classes, epochs=200, learning_rate=0.0001):
        self.weights = np.zeros((X_train.shape[1], n_classes))
        y_train_enc = np.eye(self.n_classes)[y_train]
        for epoch in range(epochs):
            z = X_train.dot(self.weights)
            a = sigmoid(z)
            grad = 1/X_train.shape[0] * X_train.T.dot(a - y_train_enc)
            weights_update = gd_optimizer(grad, learning_rate)
            self.weights += weights_update
            return weights_update


    def predict(self, X_test):
        z = X_test.dot(self.weights)
        y_pred = np.argmax(sigmoid(z), axis=1)
        return y_pred


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
                    weights = weights + lr * (-2 * (1 / n_epochs) * weights)


        return weights

    clf = Classifier(len(CATEGORIES))
    clf.fit(Classifier.X_train, Classifier.y_train)

    # Predict the classes of the test data
    y_pred = clf.predict(Classifier.X_test)

    # Calculate the accuracy
    accuracy = np.mean(y_pred == Classifier.y_test)
    print(f"Accuracy: {accuracy}")
