
from Config import CATEGORIES
from data_splitter import Train_Test
from Classifier import Classifier
import os
import cv2



clf = Classifier(len(CATEGORIES))

def classify_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_array, (128, 128))
    img_flat = img_resized.reshape(1, -1)
    class_num = clf.predict(img_flat)[0]
    class_name = CATEGORIES[class_num]
    return class_name

for i in os.listdir(r"C:\Users\admin\Desktop\Multi_class_image_classification\Unseen-Data-Test"):
    if i.endswith(".csv"):
        pass
    else:
        print(i)
        image_path = r"C:/Users/admin/Desktop/Attempt-2/alien_test/"+i
        print (image_path)
        class_name = classify_image(image_path)
        print(f"Image classification: {class_name}")