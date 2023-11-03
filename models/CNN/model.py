import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold,KFold
from tensorflow.keras.constraints import MaxNorm


def residual_block(x, filters, kernel_size=3, stride=1):
    """
    This function will create the residual block for the neural network
    """

    identity = x
    
    # Layers 1
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Layers 2
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Layers 3
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # If the stride is not equal to 1, then we need to change the shape of the identity So that the input and the output can be added together
    # If not we pass this step becasue the shape of the input and the output will be the same
    if stride != 1:
        identity = Conv1D(filters, 1, strides=stride)(identity)
        identity = BatchNormalization()(identity)
        
    x = tf.keras.layers.add([x, identity])
    x = Activation('relu')(x)
    
    return x



def ResNet_model(input_shape,num_classes):
    """
    This is the function for building up the resnet model

    """
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Add residual blocks - Stage 2
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)
    
    
    # Stage - 3
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 128, stride=1)
    
    # stage - 4
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)
    x = residual_block(x, 256, stride=1)
    x = residual_block(x, 256, stride=1)
    x = residual_block(x, 256, stride=1)
    x = residual_block(x, 256, stride=1)
    
    # Stage - 5
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512, stride=1)
    x = residual_block(x, 512, stride=1)
    
    # AveragePooling1D and NN
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    return model


def ResNet_model_min_layers(input_shape,num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Add residual blocks
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    return model