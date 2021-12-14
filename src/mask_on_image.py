import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *

if not os.path.exists(IMAGE_OUT_DIR):  #if IMAGE_OUT_DIR path is mask folder not found, create new folder
    os.mkdir(IMAGE_OUT_DIR)
    
mask_list = os.listdir(MASK_DIR) # Create a list which contains every file name in masks folder

for f in mask_list:  # Remove hidden files if any
    if f.startswith('.'):
        mask_list.remove(f)
        
#Convert images in jpg format to png format
# for maskname in tqdm.tqdm(masks_name):
#     img=cv2.imread(os.path.join(IMG_DIR,maskname[:-4]+".jpg")).astype(np.uint8)
#     cv2.imwrite(os.path.join(IMG_DIR,maskname),img)
#     os.remove(os.path.join(IMG_DIR,maskname[:-4]+".jpg"))#Delete jpg after saving as png

for mask_name in tqdm.tqdm(mask_list):
    # Access required folders
    mask_path      = os.path.join(MASK_DIR, mask_name)
    image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)
    
    # Read mask and corresponding original image
    mask  = cv2.imread(mask_path, 0).astype(np.uint8)
    image = cv2.imread(image_out_path).astype(np.uint8)

    # Change the color of the pixels on the original image that corresponds
    # to the mask part and create new image
    copy_img  = image.copy()
    image[mask==1, :] = (255, 0, 125)
    opac_image=(image/2+copy_img/2).astype(np.uint8)
 
    # Write output image into IMAGE_OUT_DIR folder
    cv2.imwrite(image_out_path, opac_image)

# Visualize created image if VISUALIZE option is chosen
if VISUALIZE:
    plt.figure()
    plt.imshow(opac_image)
    plt.show()