import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold,KFold


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

def model_train(X_train,y_train,model,folds,_epochs):

    # Define the number of folds (k)
    k = folds

    # Initialize StratifiedKFold
    # skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Initialize a list to store validation results
    validation_accuracy_results = []
    validation_loss_results = []
    
    # Loop through the folds
    for train_index, val_index in kf.split(X_train, y_train):
        
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Create and compile your model (if not already done)

        # Train the model on X_train_fold and y_train_fold
        hsitory = model.fit(X_train_fold, y_train_fold, epochs=_epochs, batch_size=32,validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping])

        # Evaluate the model on X_val_fold and store the validation results
        val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold)
        validation_accuracy_results.append(val_accuracy)
        validation_loss_results.append(val_loss)
    
    return model,validation_accuracy_results,validation_loss_results
