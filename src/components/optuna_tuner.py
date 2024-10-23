import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import AUC
from keras.regularizers import l2

def load_data(file_path='data/updated.csv'):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['label']).values
    y = df['label'].values

    # Encode and preprocess
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, y_train, y_val

def objective(trial):
    # Hyperparameters to tune
    units = trial.suggest_categorical('units', [64, 128, 256])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3])
    l2_reg = trial.suggest_categorical('l2_reg', [1e-4, 1e-3])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    epochs = trial.suggest_categorical('epochs', [30, 50])

    X_train, X_val, y_train, y_val = load_data()

    # Build model
    model = Sequential()
    model.add(Dense(units, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', AUC()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, lr_reduction], verbose=0)

    return history.history['val_loss'][-1]

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)  # Adjust n_trials as needed

    print(f"Best hyperparameters: {study.best_params}")
