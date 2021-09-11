# imports
import logging
import coloredlogs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Project level imports
from bmri.model.bmri_metrics import dice_loss, dice_coef, iou

# Third party imports
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import pandas as pd
from tqdm import tqdm
import cv2
from glob import glob
import numpy as np
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score, precision_score

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG, logger=logger)

HEIGHT = 256
WIDTH = 256
SEED = 42


def get_data(path):
    x = sorted(glob(os.path.join(path, 'image', '*.jpg')))
    y = sorted(glob(os.path.join(path, 'mask', '*.jpg')))
    return x, y


def save_results(image, mask, y_pred, save_image_path):
    #== Concatenate all images into 1 image following the format: image - mask - y_pred ==#
    line = np.ones((HEIGHT, 10, 3)) * 128

    #== Mask ==#
    mask = np.expand_dims(mask, axis=-1)  # (256, 256, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (256, 256, 3)

    #== Predicted Mask ==#
    y_pred = np.expand_dims(y_pred, axis=-1)  # (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  # (256, 256, 3)
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


def main(experiment_version, experiment_name, model_path, SAVE=False):
    # == Seeding ==#
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    #== Loading model ==#
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(model_path)

    #== Loading test set ==#
    dset_path = os.path.join('dataset', 'processed')
    test_path = os.path.join(dset_path, 'test')

    X_test, y_test = get_data(test_path)

    #== Evaluation and Prediction ==#
    score = []
    for x, y in tqdm(zip(X_test, y_test), total=len(X_test)):
        name = x.split('\\')[-1].split('.')[0]

        #== Reading the image ==#
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.
        x = np.expand_dims(x, axis=0)

        #== Reading the mask ==#
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y = mask/255.
        y = y.astype(np.int32)

        #== Prediction ==#
        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)

        y_pred = y_pred > 0.5  # Apply threshold of 0.5
        y_pred = y_pred.astype(np.int32)

        if SAVE:
            os.makedirs(os.path.join('models', experiment_version, experiment_name, 'eval_result'), exist_ok=True)
            save_image_path = os.path.join('models', experiment_version, experiment_name, 'eval_result', f'{name}.png')
            save_results(image, mask, y_pred, save_image_path)

            #== Flatten the array ==#
            y = y.flatten()
            y_pred = y_pred.flatten()

            #== Calculating the metrics values ==#
            acc_value = accuracy_score(y, y_pred)
            f1_value = f1_score(y, y_pred, labels=[0, 1], average='binary', zero_division=1)
            jac_value = jaccard_score(y, y_pred, labels=[0, 1], average='binary', zero_division=1)
            recall_value = recall_score(y, y_pred, labels=[0, 1], average='binary', zero_division=1)
            precision_value = precision_score(y, y_pred, labels=[0, 1], average='binary', zero_division=1)
            score.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

            df = pd.DataFrame(score, columns=['Image', 'Accuracy', 'F1', 'Jaccard', 'Recall', 'Precision'])
            df.to_csv(os.path.join('models', experiment_version, experiment_name, 'eval_result', 'score.csv'), index=False)




if __name__ == '__main__':
    experiment_version = 'bmri_train_v1.0.0'
    experiment_name = 'vgg19_unet_50-epochs'
    model_path = 'models/bmri_train_v1.0.0/vgg19_unet_50-epochs/model.h5'
    main(experiment_version, experiment_name, model_path, SAVE=True)