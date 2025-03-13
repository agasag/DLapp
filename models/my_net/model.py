#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    12-Mar-2025 11:20:40

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    imageinput_unnormalized = keras.Input(shape=(128,128,3), name="imageinput_unnormalized")
    imageinput = keras.layers.Normalization(axis=(1,2,3), name="imageinput_")(imageinput_unnormalized)
    conv = layers.Conv2D(8, (3,3), padding="same", name="conv_")(imageinput)
    batchnorm = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_")(conv)
    relu = layers.ReLU()(batchnorm)
    fc = layers.Reshape((-1,), name="fc_preFlatten1")(relu)
    fc = layers.Dense(2, name="fc_")(fc)
    softmax = layers.Softmax()(fc)

    model = keras.Model(inputs=[imageinput_unnormalized], outputs=[softmax])
    return model
