'''Train the bmri models, using MLFlow to keep track of train results

Usage:
    mlflow ui
'''

# imports
import logging
import yaml
from datetime import datetime
import coloredlogs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Project level imports
from bmri.model.make_model import get_model
from bmri.model.bmri_metrics import dice_loss, dice_coef, iou

# Third party imports
import tensorflow as tf
import keras
import mlflow
import cv2
from glob import glob
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG, logger=logger)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


HEIGHT = 256
WIDTH = 256
SEED = 42


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=SEED)
    return x, y


def get_data(path):
    x = sorted(glob(os.path.join(path, 'image', '*.jpg')))
    y = sorted(glob(os.path.join(path, 'mask', '*.jpg')))
    return x, y


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (WIDTH, HEIGHT))
    x = x/255.0
    x = x.astype(np.float32)
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (WIDTH, HEIGHT))
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([HEIGHT, WIDTH, 3])
    y.set_shape([HEIGHT, WIDTH, 1])
    return x, y


def tf_dataset(x, y, batch=8):
    '''Optimizing loading dataset to train

    Args:
        x (list): list of images
        y (list): list of mask_images
        batch: number of batch; default = 8

    Returns:
        dataset: dataset after applying slices, parse, batch and prefetch
    '''
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


def main(experiment_name, SAVE=False):
    logger.info('Initilizing Hyperparameters ...')

    mlflow.set_experiment(experiment_name)
    new_changes = 'vgg19_unet_50-epochs'
    run_name = f'model-run_{new_changes}'

    with mlflow.start_run(run_name=run_name) as run:
        train_params = yaml.safe_load(open('params.yaml'))['train']
        callback_params = yaml.safe_load(open('params.yaml'))['callback']
        model_params = yaml.safe_load(open('params.yaml'))['model']

        #== Seeding ==#
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        #== Hyperparameters ==#
        batch_size = train_params['BATCH_SIZE']
        learning_rate = float(train_params['LR'])
        epochs = train_params['EPOCHS']

        os.makedirs(f'models/{experiment_name}/{new_changes}', exist_ok=True)
        model_checkpoint_path = os.path.join('models', f'{experiment_name}', new_changes, 'best_weight.h5')
        full_model_path = os.path.join('models', f'{experiment_name}', new_changes, 'model.h5')
        csv_path = os.path.join('models', f'{experiment_name}', new_changes, 'train_score.csv')

        logger.info('Loading dataset ...')
        #== Dataset ==#
        dset_path = os.path.join('dataset', 'processed')
        train_path = os.path.join(dset_path, 'train')
        valid_path = os.path.join(dset_path, 'valid')

        X_train, y_train = get_data(train_path)
        X_train, y_train = shuffling(X_train, y_train)
        X_valid, y_valid = get_data(valid_path)

        logger.info('Create pipeline for dataset ...')
        #== Pipeline ==#
        train_dset = tf_dataset(X_train, y_train, batch=batch_size)
        valid_dset = tf_dataset(X_valid, y_valid, batch=batch_size)

        logger.info('Initilizing model ...')
        #== Model ==#
        model_name = model_params['NAME']
        pool_size = tuple(map(int, model_params['POOL_SIZE'].split(',')))
        num_filters = model_params['FILTERS']
        activation = model_params['ACTIVATION']
        output_activation = model_params['OUTPUT_ACTIVATION']

        model = get_model(pool_size, num_filters, activation, output_activation, model_name)
        if not model:
            raise ModuleNotFoundError(f'Model was not found! Double check model name!!')

        metrics = [dice_coef, iou, Recall(), Precision()]
        model.compile(loss=dice_loss, optimizer=Adam(learning_rate), metrics=metrics)

        callbacks = [
            ModelCheckpoint(model_checkpoint_path, verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor=callback_params['monitor'], factor=0.1, patience=callback_params['reduce_lr_patience'], min_lr=float(callback_params['min_lr']), verbose=1),
            CSVLogger(csv_path),
            TensorBoard(),
            EarlyStopping(monitor=callback_params['monitor'], patience=callback_params['early_stop_patience'], restore_best_weights=False)
        ]

        mlflow.log_params(train_params)
        mlflow.log_params(model_params)
        mlflow.log_params(callback_params)

        logger.info('Training model ...')
        logger.debug(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}", )
        if len(tf.config.list_physical_devices('GPU')) > 0:
            logger.debug(f'Utilizing GPU: {tf.test.gpu_device_name()} to train!')
        else:
            logger.debug('No GPUs found. Utilizing CPU to train!')

        history = model.fit(
            train_dset,
            epochs=epochs,
            validation_data=valid_dset,
            callbacks=callbacks,
            shuffle=False
        )

        for count in range(0, epochs):
            for metric in history.history:
                mlflow.log_metric(f'{metric}', round(history.history[metric][count], 4))


        if SAVE:
            model.save(full_model_path)
            mlflow.log_artifact(local_path=full_model_path)
            mlflow.log_artifact(local_path=model_checkpoint_path)
            logger.debug(f'Model saved under {full_model_path}')

    logger.info('Finished training experiment!')


if __name__ == '__main__':
    experiment_name = 'bmri_train_v1.0.0'
    main(experiment_name, SAVE=True)
