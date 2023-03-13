import pandas as pd
import os
import sys
import random
import cv2
from logger import logging
import numpy as np
import Config
from exception import SensorException

class dataloading:

    def data_dump():
        try:
            logging.info("start of Dump process")
            logging.info(f"{'>>' * 20} DATA Dump{'<<' * 20}")
            training_data = []
            for category in Config.CATEGORIES:
                path = os.path.join(Config.DATADIR, category)
                class_num = Config.CATEGORIES.index(category)
                for img in os.listdir(path):
                    try:
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                        img_resized = cv2.resize(img_array, (Config.image_size, Config.image_size))
                        training_data.append([img_resized, class_num])
                    except Exception as e:
                        pass

            return (training_data)
        except Exception as e:
            logging.info(f"{'>>' * 20} Their is error in Data_loader{'<<' * 20}")

            raise SensorException(error_message=e, error_detail=sys)
