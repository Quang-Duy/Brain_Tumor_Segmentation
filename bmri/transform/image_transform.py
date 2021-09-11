# imports
import os
import logging
import coloredlogs
import cv2
import numpy as np


# Third party imports
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
from PIL import Image


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG, logger=logger)

""" Equalization experiment failed due to worse dice_loss compared to baseline model

def histogram_equalization(rgb_img):
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return equalized_img
"""



def augment_data(images, masks, save_path, dset_name, augment=True):
    '''Convert images from .tif to .jpg, and apply augmentation (horizontal flip + vertical flip + rotate)

    Args:
        images (list): list of path of images
        masks (list): list of path of masks_images
        save_path (str): path to save processed images
        augment (bool): apply augmentation or not

    Returns:

    '''
    logger.info('Augmenting data ...')
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images), desc=dset_name):
        # idx = index, x = image, y = mask

        #== Extracting directory name and image name ==#
        # Ex: dataset/raw/kaggle_3m\TCGA_DU_6400_19830518\TCGA_DU_6400_19830518_39.tif -> TCGA_DU_6400_19830518_39
        dir_name = x.split('/')[-2]
        name = x.split('/')[-1].split('.')[0]
        true_mask_name = '/'.join(y.split('/')[:-1]) + f'/{name}' + '_mask.tif'

        x = cv2.imread(x, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR
        y = cv2.imread(true_mask_name, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR

        """
        x = histogram_equalization(x)
        y = histogram_equalization(y)
        """

        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = VerticalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
        else:
            X = [x]
            Y = [y]

        idx = 0
        for i, m in zip(X, Y):
            if len(X) == 1:
                temp_image_name = f'{name}.jpg'
                temp_mask_name = f'{name}.jpg'
            else:
                temp_image_name = f'{name}_{idx}.jpg'
                temp_mask_name = f'{name}_{idx}.jpg'

            image_path = os.path.join(save_path, 'image/', temp_image_name)
            mask_path = os.path.join(save_path, 'mask/', temp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1
