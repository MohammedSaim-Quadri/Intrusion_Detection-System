import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, plot_importance  # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import AUC
from keras.regularizers import l2
from dataclasses import dataclass

from src.utils import evaluate_XGB, evaluate_NN, save_object, save_nn_model


@dataclass
class ModelTrainerConfig:
    model_train_obj_file_path: str = os.path.join("artifacts", "model_trained.pkl")
    model_NNtrain_obj_file_path: str = os.path.join("artifacts", "model_trained.keras")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_XGB_trainer(self, train_transformed_df, test_transformed_df):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_transformed_df.iloc[:, :-1],  # Select all columns except the last for features
                train_transformed_df.iloc[:, -1],   # Select the last column for target (label)
                test_transformed_df.iloc[:, :-1],   # Same for test data
                test_transformed_df.iloc[:, -1]     # Select the last column for test target
            )


            # List of Model that can be tried for this Perticular Problem
            # models = {
            #     "Random Forest": RandomForestClassifier(),
            #     "Decision Tree": DecisionTreeClassifier(),
            #     "Gradient Boosting": GradientBoostingClassifier(),
            #     "XGBRegressor": XGBClassifier()
            # }
            XGB_model = XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                eval_metric="mlogloss",
                colsample_bytree=0.8,
                learning_rate=0.1,
                max_depth=7,
                n_estimators=200,
                subsample=1.0,
            )
            logging.info(f"model Started to Trained on both training and testing dataset")
            result = evaluate_XGB(X_train, y_train, X_test, y_test, XGB_model)
            save_object(
                file_path=self.model_trainer_config.model_train_obj_file_path,
                obj=XGB_model
            )
            logging.info(f"Completetd with Model Training")
            return result
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_NN_trainer(self, train_transformed_df, test_transformed_df):
        try:
            logging.info("Split training and test input data")
            # Split the data into features (X) and labels (y)
            X_train = train_transformed_df.iloc[:, :-1]  # All columns except the last
            y_train = train_transformed_df.iloc[:, -1]  # Already label-encoded

            X_test = test_transformed_df.iloc[:, :-1]  # All columns except the last
            y_test = test_transformed_df.iloc[:, -1]  # Already label-encoded

            # Convert encoded labels to one-hot encoding
            y_train_categorical = to_categorical(y_train)
            y_test_categorical = to_categorical(y_test)

            # Optional: Further split the train set into training and validation sets
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train, y_train_categorical, test_size=0.2, random_state=42, stratify=y_train
            )

            # Normalize the feature data using MinMaxScaler
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_split)
            X_test_scaled = scaler.transform(X_test_split)

            logging.info("Data preprocessing complete")

            model = Sequential()

            # Input layer: 
            # Using hp_units=128, hp_l2=0.001, hp_dropout=0.1, hp_activation='relu'
            model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))

            # First hidden layer
            model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))

            # Second hidden layer
            model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))

            # Third hidden layer
            model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))

            # Fourth hidden layer
            model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))

            # Output layer: softmax for multi-class classification
            model.add(Dense(y_train_categorical.shape[1], activation='softmax'))

            hp_learning_rate = 0.0001  # Hyperparameter for learning rate

            optimizer = Adam(learning_rate=hp_learning_rate)

            # Compile the model
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', AUC()])

            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-8)


            
            logging.info(f"model Started to Trained on both training and testing dataset")
            result = evaluate_NN(X_train_scaled, y_train_split, X_test_scaled, y_test_split,early_stopping, lr_reduction, model)
            save_nn_model(
                file_path=self.model_trainer_config.model_NNtrain_obj_file_path,
                model=model
            )
            logging.info(f"Completetd with Model Training")
            return result
        except Exception as e:
            raise CustomException(e, sys)