import glob
import cv2
import torch
import numpy as np
from constant import *

def tensorize_image(image_path_list, output_shape, cuda=False):
    # Create empty list
    local_image_list = []
    # For each image
    for image_path in image_path_list:
        # Access and read image
        image = cv2.imread(image_path)
        # Resize the image according to defined shape
        image = cv2.resize(image, output_shape)
        # Change input structure according to pytorch input structure
        torchlike_image = torchlike_data(image)
        # Add into the list
        local_image_list.append(torchlike_image)

    # Convert from list structure to torch tensor
    image_array = np.array(local_image_list, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()

    # If multiprocessing is chosen
    if cuda:
        torch_image = torch_image.cuda()

    return torch_image

def tensorize_mask(mask_path_list, output_shape, n_class, cuda=False):
    # Create empty list
    local_mask_list = []

    # For each masks
    for mask_path in mask_path_list:
        # Access and read mask
        mask = cv2.imread(mask_path, 0)
        # Resize the image according to defined shape
        mask = cv2.resize(mask, output_shape)
        # Apply One-Hot Encoding to image
        mask = one_hot_encoder(mask, n_class)
        # Change input structure according to pytorch input structure
        torchlike_mask = torchlike_data(mask)
        # Add into the list
        local_mask_list.append(torchlike_mask)
    # Convert from list structure to torch tensor
    mask_array = np.array(local_mask_list, dtype=np.int)
    torch_mask = torch.from_numpy(mask_array).float()
    if cuda:
        torch_mask = torch_mask.cuda()

    return torch_mask

def one_hot_encoder(res_mask,n_classes):
    #one hot encode
    #Create an np.array of zeros.
    one_hot=np.zeros((res_mask.shape[0],res_mask.shape[1],n_classes),dtype=np.int)
    for i,unique_value in enumerate(np.unique(res_mask)):
        one_hot[:,:,i][res_mask==unique_value]=1
    return one_hot

def torchlike_data(data):
    #transpose process 
    n_channels = data.shape[2]
    torchlike_data = np.empty((n_channels, data.shape[0], data.shape[1]))
    #Returns a new array of the given shape and type.
    #creates an array of these sizes
    for ch in range(n_channels):
        torchlike_data[ch] = data[:,:,ch] #torchlike_data[0]=data[:,:,0] 
    return torchlike_data


def image_mask_check(image_path_list, mask_path_list):
    # Check list lengths
    if len(image_path_list) != len(mask_path_list):
        print("There are missing files ! Images and masks folder should have same number of files.")
        return False

    # Check each file names
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        mask_name  = mask_path.split('/')[-1].split('.')[0]
        if image_name != mask_name:
            print("Image and mask name does not match {} - {}".format(image_name, mask_name)+"\nImages and masks folder should have same file names." )
            return False
    return True

if __name__ == '__main__':
    # Access images
    image_path=[] 
    for name in os.listdir(IMAGE_DIR):
        image_path.append(os.path.join(IMAGE_DIR,name))
        
    # Take images to number of batch size
    batch_image_list=image_path[:4]

    # Convert into Torch Tensor
    batch_image_tensor = tensorize_image(batch_image_list, (224, 224))

    # Check
    print("For features:\ndtype is "+str(batch_image_tensor.dtype))
    print("Type is "+str(type(batch_image_tensor)))
    print("The size should be ["+str(BACTH_SIZE)+", 3, "+str(HEIGHT)+", "+str(WIDTH)+"]") #3 deep.
    print("Size is "+str(batch_image_tensor.shape)+"\n")

    # Access masks
    mask_path=[] 
    for name in os.listdir(MASK_DIR):
        mask_path.append(os.path.join(MASK_DIR,name))
        
    # Take masks to number of batch size    
    batch_mask_list=mask_path[:4]

    # Convert into Torch Tensor
    batch_mask_tensor = tensorize_mask(batch_mask_list, (HEIGHT, WIDTH), 2)

    # Check
    print("For labels:\ndtype is "+str(batch_mask_tensor.dtype))
    print("Type is "+str(type(batch_mask_tensor)))
    print("The size should be ["+str(BACTH_SIZE)+", 2, "+str(HEIGHT)+", "+str(WIDTH)+"]") #2 deep
    print("Size is "+str(batch_mask_tensor.shape))