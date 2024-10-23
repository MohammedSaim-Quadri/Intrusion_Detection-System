import kerastuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from optuna_tuner import load_data  # Reuse the data loading function

def model_builder(hp):
    model = Sequential()
    units = hp.Int('units', min_value=64, max_value=512, step=64)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    l2_reg = hp.Choice('l2_reg', values=[0.001, 0.01, 0.1])
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])

    model.add(Dense(units, input_dim=load_data()[0].shape[1], activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for _ in range(hp.Int('layers', 2, 5)):
        model.add(Dense(units // 2, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(load_data()[2].shape[1], activation='softmax'))

    if optimizer_choice == 'adam':
        optimizer = Adam()
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop()
    else:
        optimizer = SGD()

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', AUC()])

    return model

if __name__ == "__main__":
    tuner = kt.BayesianOptimization(
        model_builder,
        objective='val_loss',
        max_trials=10,
        directory='tuner_results',
        project_name='IDS_bayesian'
    )

    X_train, X_val, y_train, y_val = load_data()

    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3),
                            ReduceLROnPlateau(monitor='val_loss', patience=2)])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters: {best_hps}")
