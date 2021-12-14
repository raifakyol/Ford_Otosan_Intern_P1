import os

DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "images") #image path
MASK_DIR = os.path.join(DATA_DIR, "masks") #mask path
JSON_DIR = os.path.join(DATA_DIR, "jsons") #json path
IMAGE_OUT_DIR = os.path.join(DATA_DIR, "masked_images")
MODELS_DIR = "../model"
AUG_IMAGE_DIR = os.path.join(DATA_DIR, "augmentation_image")
AUG_MASK_DIR = os.path.join(DATA_DIR, "augmentation_mask")   
PREDICT_DIR = os.path.join(DATA_DIR, "predict")

# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = True

# Bacth size
BACTH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS= 2