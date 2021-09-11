# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Third party imports
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

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


def decoder_block(input, skip_features, num_filters, pool_size, activation):
    x = Conv2DTranspose(num_filters, pool_size, strides=2, padding='same')(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, activation)
    return x


def build_vgg19_unet(input_shape, pool_size, num_filters, activation, output_activation):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output
    s2 = vgg19.get_layer("block2_conv2").output
    s3 = vgg19.get_layer("block3_conv4").output
    s4 = vgg19.get_layer("block4_conv4").output

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output

    """ Decoder """
    d1 = decoder_block(b1, s4, num_filters * 8, pool_size, activation)
    d2 = decoder_block(d1, s3, num_filters * 4, pool_size, activation)
    d3 = decoder_block(d2, s2, num_filters * 2, pool_size, activation)
    d4 = decoder_block(d3, s1, num_filters, pool_size, activation)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation=output_activation)(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model


if __name__ == '__main__':
    input_shape = (256, 256, 3)
    model = build_vgg19_unet(input_shape, POOL_SIZE, NUM_FILTERS, ACTIVATION, OUTPUT_ACTIVATION)
    model.summary()