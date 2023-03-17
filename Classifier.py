import numpy as np
import sys
from logger import logging
from exception import SensorException


class Classifier:

    def __init__(self, n_classes):
        """
        Constructor for Classifier class.

        Parameters:
        -----------
        n_classes: int
            Number of classes for classification
        """
        self.n_classes = n_classes

    def fit(self, X_train, y_train):
        """
        Method to fit the classifier on training data.

        Parameters:
        -----------
        X_train: numpy array
            Array of shape (n_samples, n_features) containing training data.
        y_train: numpy array
            Array of shape (n_samples,) containing training labels.
        """
        try:
            self.weights = np.zeros((self.n_classes, X_train.shape[1]))
            for c in range(self.n_classes):
                y_train_c = np.where(y_train == c, 1, -1)
                self.weights[c] = self.sgd(X_train, y_train_c)
            logging.info("Classifier fit successfully")
        except Exception as e:
            logging.error("Error occurred in Classifier fit")
            raise SensorException(error_message=e, error_detail=sys)

    def predict(self, X_test):
        """
        Method to predict the class labels of test data.

        Parameters:
        -----------
        X_test: numpy array
            Array of shape (n_samples, n_features) containing test data.

        Returns:
        --------
        y_pred: numpy array
            Array of shape (n_samples,) containing predicted class labels.
        """
        try:
            y_pred = np.argmax(X_test.dot(self.weights.T), axis=1)
            logging.info("Classifier prediction successful")
            return y_pred
        except Exception as e:
            logging.error("Error occurred in Classifier predict")
            raise SensorException(error_message=e, error_detail=sys)

    def sgd(self, X_train, y_train):
        """
        Method to perform stochastic gradient descent.

        Parameters:
        -----------
        X_train: numpy array
            Array of shape (n_samples, n_features) containing training data.
        y_train: numpy array
            Array of shape (n_samples,) containing training labels.

        Returns:
        --------
        weights: numpy array
            Array of shape (n_features,) containing classifier weights.
        """
        try:
            lr = 0.1
            n_epochs = 50
            weights = np.zeros(X_train.shape[1])
            for epoch in range(n_epochs):
                for i, x in enumerate(X_train):
                    if y_train[i] * np.dot(x, weights) < 1:
                        weights = weights + lr * ((y_train[i] * x) + (-2 * (1 / n_epochs) * weights))

                    else:
                        pass
            logging.info("Stochastic gradient descent successful")
            return weights
        except Exception as e:
            logging.error("Error occurred in stochastic gradient descent")
            raise SensorException(error_message=e, error_detail=sys)

