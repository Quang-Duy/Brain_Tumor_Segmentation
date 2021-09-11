# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Third party imports

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

POOL_SIZE = (2, 2)
NUM_FILTERS = 64
ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'sigmoid'

"""
import segmentation_models as sm
def build_unet_resnet50(input_shape, n_classes, output_activation, BACKBONE='resnet50'):
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters

    model = sm.Unet(BACKBONE, input_shape=input_shape, classes=n_classes, activation=output_activation, encoder_weights=None, encoder_freeze=True)

    return model
"""

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


def build_unet_resnet50(input_shape, pool_size, num_filters, activation, output_activation):
    #== Input ==#
    inputs = Input(input_shape)

    #== Pre-trained ResNet50 Model ==#
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    #== Encoder ==#
    s1 = resnet50.layers[0].output # resnet50.get_layer("input_1").output: ERROR: after the first run, input_1 becomes input_2, and so on!
    s2 = resnet50.get_layer("conv1_relu").output
    s3 = resnet50.get_layer("conv2_block3_out").output
    s4 = resnet50.get_layer("conv3_block4_out").output

    #== Bridge ==#
    b1 = resnet50.get_layer('conv4_block6_out').output

    #== Decoder ==#
    d1 = decoder_block(b1, s4, num_filters * 8, pool_size, activation)
    d2 = decoder_block(d1, s3, num_filters * 4, pool_size, activation)
    d3 = decoder_block(d2, s2, num_filters * 2, pool_size, activation)
    d4 = decoder_block(d3, s1, num_filters, pool_size, activation)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation=output_activation)(d4)

    model = Model(inputs, outputs, name="ResNet50_Unet")
    return model


if __name__ == '__main__':
    input_shape = (256, 256, 3)
    model = build_unet_resnet50(input_shape, POOL_SIZE, NUM_FILTERS, ACTIVATION, OUTPUT_ACTIVATION)
    model.summary()