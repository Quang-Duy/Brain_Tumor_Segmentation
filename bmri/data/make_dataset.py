'''Preprocessing dataset'''

# imports
import logging
import coloredlogs

# Third party imports

# Project Level Imports
from bmri.data.dataset import load_data, create_dir
from bmri.transform.image_transform import augment_data

DSET_PATH = 'dataset/raw/kaggle_3m'

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG, logger=logger)

def preprocessing_dset():
    logger.info('Begin preprocessing ...')
    # Load dataset
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data(DSET_PATH)

    logger.info('Creating new directories to store processed data ...')
    create_dir('dataset/processed/train/image/')
    create_dir('dataset/processed/train/mask/')
    create_dir('dataset/processed/valid/image/')
    create_dir('dataset/processed/valid/mask/')
    create_dir('dataset/processed/test/image/')
    create_dir('dataset/processed/test/mask/')

    augment_data(X_train, y_train, 'dataset/processed/train/', 'train_dset', augment=True)
    augment_data(X_valid, y_valid, 'dataset/processed/valid/', 'valid_dset', augment=False)
    augment_data(X_test, y_test, 'dataset/processed/test/', 'test_dset', augment=False)


if __name__ == '__main__':
    preprocessing_dset()
