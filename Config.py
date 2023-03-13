import os
image_size = 128
current_path = os.getcwd()
DATADIR = os.path.join(current_path, "dataset")
CATEGORIES = os.listdir(DATADIR)
