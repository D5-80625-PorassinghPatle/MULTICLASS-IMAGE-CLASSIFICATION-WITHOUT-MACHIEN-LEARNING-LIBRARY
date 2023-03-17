import os
import sys
import cv2
from logger import logging
import Config
from exception import SensorException


class DataLoader:
    """Loads and preprocesses image data for machine learning models.

    Attributes:
        None

    Methods:
        data_dump(): Loads image data from a directory and returns a list of preprocessed images.
    """

    def data_dump(self):
        """Loads image data from a directory and returns a list of preprocessed images.

        Args:
            None

        Returns:
            training_data (list): A list of preprocessed images and their corresponding labels.
        """
        try:
            # Log the start of the data loading process.
            logging.info("Starting data loading process.")

            # Initialize an empty list to store the preprocessed image data.
            training_data = []

            # Iterate through each category of image data.
            for category in Config.CATEGORIES:
                # Get the path to the category's image data directory.
                path = os.path.join(Config.DATADIR, category)

                # Get the label associated with this category.
                class_num = Config.CATEGORIES.index(category)

                # Iterate through each image in the category's directory.
                for img in os.listdir(path):
                    try:
                        # Load the image as grayscale.
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                        # Resize the image to the specified size.
                        img_resized = cv2.resize(img_array, (Config.image_size, Config.image_size))

                        # Append the preprocessed image and its label to the training data list.
                        training_data.append([img_resized, class_num])
                    except Exception as e:
                        # If there is an error loading or preprocessing an image, skip it and move on to the next one.
                        logging.warning(f"Error loading or preprocessing image: {os.path.join(path, img)}")

            # Log the number of images loaded and preprocessed.
            logging.info(f"Number of images loaded and preprocessed: {len(training_data)}")

            # Return the preprocessed training data.
            return training_data

        except Exception as e:
            # If there is an error loading the data, log the error and raise a custom exception.
            logging.error("Error loading data.")
            raise SensorException(error_message=e, error_detail=sys)
