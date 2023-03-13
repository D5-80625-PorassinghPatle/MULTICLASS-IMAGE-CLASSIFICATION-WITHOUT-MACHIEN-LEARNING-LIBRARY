from logger import logging
from exception import SensorException
import sys
import numpy as np
import random
from data_loader import dataloading
import Config

class Train_Test:
    def data_spliter():
        try:
            logging.info("start of Train_Test_Split")
            logging.info(f"{'>>' * 20} TRAIN_TEST_DATA {'<<' * 20}")
            training_data = dataloading.data_dump()
            random.shuffle(training_data)
            # Split the dataset into training and testing
            X = []
            y = []
            for features, label in training_data:
                X.append(features)
                y.append(label)
            X = np.array(X).reshape(-1, Config.image_size * Config.image_size)
            y = np.array(y)
            train_size = int(0.8 * len(X))
            X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]


            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise SensorException (e,sys)



