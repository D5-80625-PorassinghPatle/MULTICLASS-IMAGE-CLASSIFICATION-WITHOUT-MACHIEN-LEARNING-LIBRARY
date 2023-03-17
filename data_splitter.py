from logger import logging
from exception import SensorException
import sys
import logging
import random
from typing import List, Tuple
import numpy as np
import Config
from exception import SensorException
from data_loader import DataLoader


class DataSplitter:
    def data_splitter(self,training_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This method splits the dataset into training and testing datasets
        """
        try:
            logging.info("Starting Train-Test Split")
            logging.info(f"{'>>' * 20} TRAIN_TEST_DATA {'<<' * 20}")


            # Shuffle the dataset
            random.shuffle(training_data)

            # Split the dataset into features and labels
            X = []
            y = []
            for features, label in training_data:
                X.append(features)
                y.append(label)
            X = np.array(X).reshape(-1, Config.image_size * Config.image_size)
            y = np.array(y)

            # Split the dataset into training and testing datasets
            train_size = int(0.8 * len(X))
            X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

            logging.info(f"Number of images for X_Training: {len(X_train)}")
            logging.info(f"Number of images for X_testing: {len(X_test)}")
            logging.info(f"Number of images for y_Training: {len(y_train)}")
            logging.info(f"Number of images for y_testing: {len(y_test)}")

            return X_train, X_test, y_train, y_test
        except (FileNotFoundError, ValueError) as e:
            # Catch specific exceptions
            # Log the error message and raise a custom exception
            logging.error(f"An error occurred during data splitting: {e}")
            raise SensorException(f"Failed to split the data: {e}") from e
