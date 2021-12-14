import json
import os
import numpy as np
import cv2
import tqdm
from constant import *


if not os.path.exists(MASK_DIR): #if MASK_DIR path is mask folder not found, create new folder
    os.mkdir(MASK_DIR)

json_list = os.listdir(JSON_DIR) # Create a list which contains every file name in "jsons" folder

for json_name in tqdm.tqdm(json_list):
    json_path = os.path.join(JSON_DIR, json_name) # Access and open json file as dictionary
    json_file = open(json_path, 'r') #file read
    json_dict = json.load(json_file) # Load json data

    # Create an empty mask whose size is the same as the original image's size
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    mask_path = os.path.join(MASK_DIR, json_name[:-9]+".png")

    # For every objects
    for obj in json_dict["objects"]:
        # Check the objects ‘classTitle’ is ‘Freespace’ or not.
        if obj['classTitle']=='Freespace':
            # Extract exterior points which is a point list that contains
            # every edge of polygon and fill the mask with the array.
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)
    
    # Write mask image into MASK_DIR folder
    cv2.imwrite(mask_path, mask.astype(np.uint8))