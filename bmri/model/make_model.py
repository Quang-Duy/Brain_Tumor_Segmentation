# imports
import logging
import coloredlogs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Project level imports
from bmri.model.bmri_UNet_model import build_unet
from bmri.model.bmri_Unet_resnet50_model import build_unet_resnet50
from bmri.model.bmri_Unet_Vgg19_model import build_vgg19_unet

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG, logger=logger)

HEIGHT = 256
WIDTH = 256


def get_model(pool_size, num_filters, activation, output_activation, model_name: str = None):
    switch = {
        'unet': build_unet(
                    input_shape=(HEIGHT, WIDTH, 3),
                    pool_size=pool_size,
                    num_filters=num_filters,
                    activation=activation,
                    output_activation=output_activation
                ),
        'vgg19_unet': build_vgg19_unet(
                    input_shape=(HEIGHT, WIDTH, 3),
                    pool_size=pool_size,
                    num_filters=num_filters,
                    activation=activation,
                    output_activation=output_activation
                ),
        'unet_resnet50': build_unet_resnet50(
                    input_shape=(HEIGHT, WIDTH, 3),
                    pool_size=pool_size,
                    num_filters=num_filters,
                    activation=activation,
                    output_activation=output_activation
                )
    }
    model = switch.get(model_name, None)
    if model:
        logger.debug(f'Model `{model_name}` selected!')
    else:
        logger.debug(f'Model not found!!!')

    return model
