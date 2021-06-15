import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import FAZ_Preprocess  # FAZ Preprocessing
from src.levelset import levelset       # Level Set
from src.metrics import scoring         # Metrics
from src.utils import binarize_phi      # Handy ultility function
from src.segmentsingle import segmentsingle     # Segment single


def visualization(model, raw_folder, mask_folder, number_image, SCALE):
    '''Plot segmentation results of different methods
    Parameters:
    ----------
    model : pytorch model
            a pytorch preloaded model
    raw_folder: string
            path to image to be segmented
    mask_folder: string
            path to mask (label) of raw images
    SCALE: int
            scale of the image to be resized, depending on DNN model
    '''
    num_cols = 2 # number of method compared
    ious = np.zeros((number_image, num_cols))
    
    image_name = os.listdir(raw_folder)

    # Select images according to number_image
    try:
        image_name = image_name[0:number_image]
        rows, columns = number_image, num_cols
    except:
        rows, columns = len(image_name), num_cols
    
    # Initiate figure
    fig=plt.figure(figsize=(10 * num_cols, 10 * number_image))
    
    for idx, name in enumerate(image_name): 
        ################ Compute ####################### 
        # Loading image and mask
        impath = os.path.join(raw_folder, name)
        im = cv2.imread(impath, cv2.IMREAD_GRAYSCALE) # ensure input image is 2D array and pixels range in [0, 255]

        # Processed (segmented)
        enhanced, mask, phi, delta = segmentsingle(im, model, SCALE, False)
       
        # Load labels
        mask_name  = name.replace("tif","png")
        label = cv2.imread(os.path.join(mask_folder, mask_name), cv2.IMREAD_GRAYSCALE)
        if np.max(label) > 1:
            label = (label/np.max(label) > 0).astype(np.uint8)  # make sure label image is binary    
        
        # Resize label
        label_scaled = cv2.resize(label, (SCALE, SCALE))
                   
        ################ Visualization #################
        binary_phi = binarize_phi(phi)  # make phi binary, not grayscale
        enhanced_scaled = cv2.resize(enhanced, (SCALE, SCALE))

        # UNet
        iou, FP, FN = scoring(label_scaled, mask)
        ious[idx, 0] = iou
        fig.add_subplot(rows, columns, num_cols*idx+1 )
        plt.title(f'{name} \n Just only Unet: IoU: {iou} FP: {FP} FN: {FN}')
        plt.imshow(enhanced_scaled)
        plt.contour(mask, levels=np.logspace(-4.7, -3., 10), colors='white', alpha=0.2)
        plt.imshow(mask, cmap='gray', alpha=0.15), plt.axis('off')
        
        # Unet + LevelSet
        iou, FP, FN = scoring(label_scaled, binary_phi)
        ious[idx, 1] = iou
        fig.add_subplot(rows, columns, num_cols*idx +2)
        plt.title(f'{name} \n Unet + Levelset: IoU: {iou} FP: {FP} FN: {FN}')
        plt.imshow(enhanced_scaled)
        plt.contour(binary_phi, levels=np.logspace(-4.7, -3., 10), colors='white', alpha=0.2)
        plt.imshow(binary_phi, cmap='gray', alpha=0.15), plt.axis('off')
        
        # Unet + Erosion + Levelset:
        '''
        iou, FP, FN = scoring(labels, new_phi)
        fig.add_subplot(rows, columns, 3*idx +3)
        plt.title(f'Unet + Erosion + Levelset: IoU: {iou} FP: {FP} FN: {FN}')
        plt.imshow(image)
        plt.imshow(new_phi, cmap='gray', alpha=0.15)
        '''
    return fig, ious