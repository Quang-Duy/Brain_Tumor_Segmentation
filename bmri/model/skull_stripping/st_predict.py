# imports
import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Third party level imports
import cv2
import numpy as np
import tensorflow as tf
from skimage.io import imsave
from skimage import img_as_float, exposure

# Project level imports
from bmri.data.dataset import create_dir
from bmri.model.skull_stripping.net import unet
from bmri.model.skull_stripping.data import load_data

path = 'dataset/raw/kaggle_3m'
weights_path = 'models/skull_stripping/weights_128.h5'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def gray2rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def main(mean=30.0, std=50.0):
    #== Load model with weights ==#
    model = unet()
    model.load_weights(weights_path)

    #== Make directories ==#
    save_dir = os.path.join('dataset', 'processed', 'skull_stripping')
    create_dir(save_dir)

    #== Load dataset ==#
    for folder in glob.glob(f'{path}/*'):
        images, masks, img_names = load_data(folder)
        original_imgs = images.astype(np.uint8)

        images -= mean
        images /= std

        # make predictions
        brain_pred = model.predict(images, verbose=1)

        # save images with segmentation and ground truth mask overlay
        for i in range(len(images)):
            pred = brain_pred[i]
            image = original_imgs[i]
            mask = masks[i]
            save_name = img_names[i]
            save_folder = '_'.join(save_name.split('_')[:-1])
            create_dir(os.path.join(save_dir, save_folder))

            # segmentation mask is for the middle slice
            image_rgb = gray2rgb(image[:, :, 1])

            # prediction contour image
            pred = (np.round(pred[:, :, 0]) * 255.0).astype(np.uint8)
            contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pred = np.zeros(pred.shape)
            cv2.drawContours(pred, contours, -1, (255, 0, 0), 1)

            # ground truth contour image
            mask = (np.round(mask[:, :, 0]) * 255.0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask = np.zeros(mask.shape)
            cv2.drawContours(mask, contours, -1, (255, 0, 0), 1)

            # combine image with contours
            pred_rgb = np.array(image_rgb)
            annotation = pred_rgb[:, :, 1]
            annotation[np.maximum(pred, mask) == 255] = 0
            pred_rgb[:, :, 0] = pred_rgb[:, :, 1] = pred_rgb[:, :, 2] = annotation
            pred_rgb[:, :, 2] = np.maximum(pred_rgb[:, :, 2], mask)
            pred_rgb[:, :, 0] = np.maximum(pred_rgb[:, :, 0], pred)

            percentiles = np.percentile(pred_rgb, (0.5, 99.5))
            scaled = exposure.rescale_intensity(pred_rgb, in_range=tuple(percentiles))

            imsave(os.path.join(save_dir, save_folder, save_name + '_scaled.png'), scaled)
            imsave(os.path.join(save_dir, save_folder, save_name + '.png'), pred_rgb)
            # cv2.imwrite(os.path.join(save_dir, save_folder, save_name + '.png'), pred_rgb)

        break



if __name__ == '__main__':
    main()