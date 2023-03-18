
import os
import cv2
from data_loader import DataLoader
from data_splitter import DataSplitter
from Classifier import Classifier
import Config
import csv
from logger import logging





clf = Classifier(len(Config.CATEGORIES))
# Load data from directory
data_loader = DataLoader()
training_data = data_loader.data_dump()

# Split data into training and testing sets
data_splitter = DataSplitter()
X_train, X_test, y_train, y_test = data_splitter.data_splitter(training_data)

# Train classifier
clf = Classifier(n_classes=len(Config.CATEGORIES))
clf.fit(X_train, y_train)



def classify_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_array, (128, 128))
    img_flat = img_resized.reshape(1, -1)
    class_num = clf.predict(img_flat)[0]
    class_name = Config.CATEGORIES[class_num]
    return class_name



# Configure the logging module
logging.basicConfig(filename='image_classification.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Create a new CSV file
with open('image_classification_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Image Name', 'Image Path', 'Predicted Result'])

    # Iterate through the images in the directory and classify them
    for i in os.listdir(r"C:\Users\admin\Desktop\Multi_class_image_classification\Unseen-Data-Test"):
        image_path = r"C:/Users/admin/Desktop/Attempt-2/alien_test/" + i
        class_name = classify_image(image_path)
        # Write the image name, path, and predicted result to the CSV file
        writer.writerow([i, image_path, class_name])
        logging.info(f"Image classified: {i}, {class_name}")

print("Results saved to image_classification_results.csv")
logging.info("Results saved to image_classification_results.csv")