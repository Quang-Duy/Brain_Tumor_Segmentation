# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Third party imports
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Input, Concatenate
from tensorflow.keras.models import Model

POOL_SIZE = (2, 2)
NUM_FILTERS = 64
ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'sigmoid'


def conv_block(input, num_filters, activation):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x


def encoder_block(input, num_filters, pool_size, activation):
    x = conv_block(input, num_filters, activation)
    p = MaxPool2D(pool_size)(x)
    return x, p


def decoder_block(input, skip_features, num_filters, pool_size, activation):
    x = Conv2DTranspose(num_filters, pool_size, strides=2, padding='same')(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, activation)
    return x


def build_unet(input_shape, pool_size, num_filters, activation, output_activation):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, num_filters, pool_size, activation)
    s2, p2 = encoder_block(p1, num_filters * 2, pool_size, activation)
    s3, p3 = encoder_block(p2, num_filters * 4, pool_size, activation)
    s4, p4 = encoder_block(p3, num_filters * 8, pool_size, activation)

    b1 = conv_block(p4, num_filters * 16, activation)

    d1 = decoder_block(b1, s4, num_filters * 8, pool_size, activation)
    d2 = decoder_block(d1, s3, num_filters * 4, pool_size, activation)
    d3 = decoder_block(d2, s2, num_filters * 2, pool_size, activation)
    d4 = decoder_block(d3, s1, num_filters, pool_size, activation)

    outputs = Conv2D(1, 1, padding='same', activation=output_activation)(d4)

    model = Model(inputs, outputs, name='U-Net')
    return model


if __name__ == '__main__':
    input_shape = (256, 256, 3)
    model = build_unet(input_shape, POOL_SIZE, NUM_FILTERS, ACTIVATION, OUTPUT_ACTIVATION)
    model.summary()