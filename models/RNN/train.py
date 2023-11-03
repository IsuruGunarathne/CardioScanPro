import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold,KFold
from tensorflow.keras.constraints import MaxNorm




# Function for training the RNN model
def model_train_RNN(X_train,y_train,model,folds,_epochs):

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
    f = 1
    for train_index, val_index in kf.split(X_train, y_train):
        
        print(f"====================Fold {f} Started ===============================")
        
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Create and compile your model (if not already done)

        # Train the model on X_train_fold and y_train_fold
        hsitory = model.fit(X_train_fold, y_train_fold, epochs=_epochs, batch_size=32,validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping])

        # Evaluate the model on X_val_fold and store the validation results
        val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold)
        validation_accuracy_results.append(val_accuracy)
        validation_loss_results.append(val_loss)
        
        
        
        print(f"====================Fold {f} Finished ===============================")
        print("========================================================================")
        
        f = f + 1
    
    return model,validation_accuracy_results,validation_loss_results

