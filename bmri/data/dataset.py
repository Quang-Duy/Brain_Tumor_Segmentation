"""

"""
# Standard Dist
import glob
import logging
import coloredlogs
import os

# Third party imports
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG, logger=logger)

PATH = 'dataset/raw/kaggle_3m'
SEED = 42

def load_data(path, split=0.2):
    '''Load up the data from the dataset/raw dir

    Args:
        path: path to dataset (default: dataset/raw/kaggle_3m)
        split: split dataset into 60 train, 20 valid and 20 test

    Returns:

    '''
    #== Load the images and mask ==#
    logger.info('Loading dataset ...')
    images = []
    masks = []
    for fname in glob.glob(f'{path}/*/*.tif'):
        fname = fname.replace('\\', '/')
        if 'mask' not in fname:
            images.append(fname)
        else:
            fname = fname.replace('_mask', '')
            masks.append(fname)


    images = sorted(images)
    masks = sorted(masks)

    logger.debug(f'Images: {len(images)} --- Annotation: {len(masks)}')

    #== Split the data ==#
    logger.info('Splitting dataset ...')
    # Split (train, valid) and test set
    split_size = int(len(images) * split)
    X_train_valid, X_test = train_test_split(images, test_size=split_size, random_state=SEED)
    y_train_valid, y_test = train_test_split(masks, test_size=split_size, random_state=SEED)

    # Split train and valid set
    split_size = int(len(X_train_valid) * 0.25)
    X_train, X_valid = train_test_split(X_train_valid, test_size=split_size, random_state=SEED)
    y_train, y_valid = train_test_split(y_train_valid, test_size=split_size, random_state=SEED)

    logger.debug(f'Train: ({len(X_train)}, {len(y_train)}) --- validation: ({len(X_valid)}, {len(y_valid)}) --- test: ({len(X_test)}, {len(y_test)})')
    logger.info('Finished Operation!')

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def create_dir(new_path):
    '''Create new directory

    Args:
        new_path: new folder to create

    Returns:

    '''
    if not os.path.exists(new_path):
        os.makedirs(new_path)

if __name__ == '__main__':
    load_data(PATH)