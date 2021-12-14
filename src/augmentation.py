import numpy as np
import cv2
import os
import torch
import tqdm
from torchvision import transforms as T
from constant import *
from PIL import Image

def image_aug(image_path):
    if not os.path.exists(AUG_IMAGE_DIR):
        os.mkdir(AUG_IMAGE_DIR)
    
    image_name_list = os.listdir(image_path)
    for image in tqdm.tqdm(image_name_list):
        img = Image.open(os.path.join(IMAGE_DIR, image))
        aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)
        
        img_aug = aug(img)
        new_path=image[:-4]+"-1"+".png"
        new_path=os.path.join(AUG_IMAGE_DIR, new_path)
        img_aug=np.array(img_aug)
        cv2.imwrite(new_path,img_aug)
              
def mask_aug(mask_path):   
    if not os.path.exists(AUG_MASK_DIR):
        os.mkdir(AUG_MASK_DIR)
    global mask
    mask_name_list = os.listdir(mask_path)
    for mask in tqdm.tqdm(mask_name_list):
        msk = cv2.imread(os.path.join(MASK_DIR, mask))
        global newm_path
        
        newm_path=mask[:-4]+"-1"+".png"
        newm_path = os.path.join(AUG_MASK_DIR, newm_path)
        cv2.imwrite(newm_path, msk)
        
image_aug(IMAGE_DIR)
mask_aug(MASK_DIR)