import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import * 
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException

def save_object(file_path, obj,):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            

    except Exception as e:
        raise CustomException(e, sys)
    

def select_top_55_features(train_transformed_data, target_column='label', top_n=55):
    try:
        # Separate the features and the target
        X = train_transformed_data.drop(columns=[target_column])
        y = train_transformed_data[target_column]
        feature_names = X.columns  # Store feature names before scaling

        # Step 1: Scale the feature data
        sc = StandardScaler()
        scale_X = sc.fit_transform(X)

        # Step 2: Fit LDA model
        lda = LinearDiscriminantAnalysis()
        lda.fit(scale_X, y)

        # Step 3: Compute feature importance scores using LDA coefficients
        lda_coefficients = np.exp(np.mean(np.abs(lda.coef_), axis=0))  # Mean of abs(coef_)
        
        # Step 4: Create a DataFrame to store feature names and their corresponding scores
        df_feature_score = pd.DataFrame({
            'Feature': feature_names,
            'Score': lda_coefficients
        })

        # Step 5: Sort the DataFrame by scores in descending order and select top N features
        imp_feature = df_feature_score.sort_values('Score', ascending=False).head(top_n)

        # Step 6: Return the list of top N important feature names
        top_important_columns = imp_feature['Feature'].tolist()
        top_important_columns.append('label')

        return top_important_columns

    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_XGB(X_train, y_train,X_test,y_test,model):
    try:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)


        ypred_XGB_proba = model.predict_proba(X_test)  # Use predict_proba for ROC AUC
    
        ypred_train_proba = model.predict_proba(X_train)  # Use predict_proba for ROC AUC

        report = {
            'Testing Accuracy Score': accuracy_score(y_test, y_test_pred),
            'Training Accuracy Score': accuracy_score(y_train, y_train_pred),
            'Testing f1 Score': f1_score(y_test, y_test_pred, average='weighted'),
            'Training f1 Score': f1_score(y_train, y_train_pred, average='weighted'),
            'Testing Recall Score': recall_score(y_test, y_test_pred, average='weighted'),
            'Training Recall Score': recall_score(y_train, y_train_pred, average='weighted'),
            'Testing Precision Score': precision_score(y_test, y_test_pred, average='weighted'),
            'Training Precision Score': precision_score(y_train, y_train_pred, average='weighted'),
            'Balance Accuracy Score':balanced_accuracy_score(y_test, y_test_pred),
            'ROC_AUC_test' : roc_auc_score(y_test, ypred_XGB_proba, multi_class='ovr'),
            'ROC_AUC_train' : roc_auc_score(y_train, ypred_train_proba, multi_class='ovr')
        }

        # for i in range(len(list(model_accuracy_metrics))):
        #     report[list(model_accuracy_metrics.keys())[i]] = model_accuracy_metrics.get(i)         
        # for i in range(len(list(models))):
        #     model = list(models.values())[i]
        #     para=param[list(models.keys())[i]]

        #     gs = GridSearchCV(model,para,cv=3)
        #     gs.fit(X_train,y_train)

        #     model.set_params(**gs.best_params_)
        #     model.fit(X_train,y_train)

        #     #model.fit(X_train, y_train)  # Train model

        #     y_train_pred = model.predict(X_train)

        #     y_test_pred = model.predict(X_test)

        #     train_model_score = r2_score(y_train, y_train_pred)

        #     test_model_score = r2_score(y_test, y_test_pred)

        #     report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
    

# Function to save NN models
def save_nn_model(file_path, model):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the NN model in HDF5 format
        model.save(file_path)  # Use .h5 extension for HDF5 format

    except Exception as e:
        raise CustomException(e, sys)

# Helper function to load NN models
def load_nn_model(file_path):
    try:
        model = load_model(file_path)  # Automatically handles .keras files
        #logging.info(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_NN(X_train, y_train, X_test, y_test, early_stopping, lr_reduction, model):
    try:
        # Train the model with validation data
        history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=50, batch_size=32, callbacks=[early_stopping, lr_reduction]
        )
        # Predict probabilities for ROC AUC
        y_train_proba = model.predict(X_train)
        y_test_proba = model.predict(X_test)

        # Convert probabilities to class predictions
        y_train_pred = y_train_proba.argmax(axis=1)
        y_test_pred = y_test_proba.argmax(axis=1)

        # Calculate performance metrics
        report = {
            'Testing Accuracy Score': accuracy_score(y_test.argmax(axis=1), y_test_pred),
            'Training Accuracy Score': accuracy_score(y_train.argmax(axis=1), y_train_pred),
            'Testing F1 Score': f1_score(y_test.argmax(axis=1), y_test_pred, average='weighted'),
            'Training F1 Score': f1_score(y_train.argmax(axis=1), y_train_pred, average='weighted'),
            'Testing Precision': precision_score(y_test.argmax(axis=1), y_test_pred, average='weighted'),
            'Training Precision': precision_score(y_train.argmax(axis=1), y_train_pred, average='weighted'),
            'Testing Recall': recall_score(y_test.argmax(axis=1), y_test_pred, average='weighted'),
            'Training Recall': recall_score(y_train.argmax(axis=1), y_train_pred, average='weighted'),
            'ROC AUC (Test)': roc_auc_score(y_test, y_test_proba, multi_class='ovr'),
            'ROC AUC (Train)': roc_auc_score(y_train, y_train_proba, multi_class='ovr')
        }

        return report

    except Exception as e:
        raise CustomException(e,sys)