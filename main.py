import os
import cv2
import numpy as np
from logger import logging
from data_loader import DataLoader
from data_splitter import DataSplitter
from Classifier import Classifier
from exception import SensorException
import Config

# Set up configuration parameters


# Set up logging configuration
logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

try:
    # Load data from directory
    data_loader = DataLoader()
    training_data = data_loader.data_dump()

    # Split data into training and testing sets
    data_splitter = DataSplitter()
    X_train, X_test, y_train, y_test = data_splitter.data_splitter(training_data)

    # Train classifier
    clf = Classifier(n_classes=len(Config.CATEGORIES))
    clf.fit(X_train, y_train)

    # Predict the classes of the test data
    y_pred = clf.predict(X_test)

    # Evaluate classifier performance
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")
    # Log results
    logging.info(f"Accuracy: {accuracy}")


except SensorException as e:
    logging.error(f"Error occurred: {e.error_message}")
