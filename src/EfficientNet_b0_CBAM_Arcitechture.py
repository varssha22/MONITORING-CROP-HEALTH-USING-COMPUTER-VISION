import tensorflow as tf
from tensorflow.keras import layers, models
import math

def se_block(inputs, se_ratio=0.25):
    filters = inputs.shape[-1]
    se_filters = max(1, int(filters * se_ratio))
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Conv2D(se_filters, 1, activation='relu')(se)
    se = layers.Conv2D(filters, 1, activation='sigmoid')(se)
    return layers.Multiply()([inputs, se])
    

def cbam_block(inputs, reduction_ratio=16):
    # Channel Attention
    channel_avg = layers.GlobalAveragePooling2D()(inputs)
    channel_max = layers.GlobalMaxPooling2D()(inputs)
    
    shared_dense = tf.keras.Sequential([
        layers.Dense(inputs.shape[-1] // reduction_ratio, activation='relu'),
        layers.Dense(inputs.shape[-1])
    ])

    avg_out = shared_dense(channel_avg)
    max_out = shared_dense(channel_max)

    channel_attention = layers.Add()([avg_out, max_out])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1, 1, inputs.shape[-1]))(channel_attention)
    x = layers.Multiply()([inputs, channel_attention])

    # Spatial Attention using only Keras layers
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    x = layers.Multiply()([x, spatial_attention])

    return x
    

def mbconv_block(inputs, out_channels, expansion_factor, kernel_size, strides, se_ratio=0.25):
    in_channels = inputs.shape[-1]
    x = inputs

    # Expansion phase
    if expansion_factor != 1:
        x = layers.Conv2D(in_channels * expansion_factor, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)

    # Depthwise conv
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # Squeeze-and-Excitation
    x = se_block(x, se_ratio=se_ratio)

    # Projection phase
    x = layers.Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Skip connection
    if strides == 1 and in_channels == out_channels:
        x = layers.Add()([inputs, x])
    
    return x
    

def EfficientNetB0_CBAM(input_shape=(224,224, 3), num_classes=38, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # MBConv blocks (adapted from official B4 config)
    # (repeats, out_channels, kernel_size, strides)
    x = mbconv_block(x,out_channels=16,expansion_factor=1,kernel_size=3,strides=1)
    block_configs = [
        (2, 24, 3, 2),
        (2, 40, 5, 2),
        (3, 80, 3, 2),
        (3, 112, 5, 1),
        (4, 192, 5, 2),
        (1, 320, 3, 1),
    ]

    expansion_factor = 6
    for repeats, out_channels, kernel_size, strides in block_configs:
        for i in range(repeats):
            x = mbconv_block(
                x,
                out_channels=out_channels,
                expansion_factor=expansion_factor,
                kernel_size=kernel_size,
                strides=strides if i == 0 else 1
            )

    # Head
    x = layers.Conv2D(1280, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # Convolutional Block Attention Module
    x = cbam_block(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
