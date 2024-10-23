import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass
from src.logger import logging  
from src.exception import CustomException  
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from src.utils import select_top_55_features,save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.data_tr_conf = DataTransformationConfig()

    def get_datatransformer(self):
        feature_pipeline = Pipeline(
                steps=[
                    (
                        "replace_infinity",
                        FunctionTransformer(
                            lambda X: X.replace([np.inf, -np.inf], np.nan),
                            validate=False,
                        ),
                    ),
                    (
                        "drop_nulls",
                        FunctionTransformer(lambda X: X.dropna(), validate=False),
                    ),
                ]
            )
        preprocessing_pipeline = ColumnTransformer(
                transformers=[
                    # Step 1: Apply the feature pipeline to all remaining columns except 'label'
                    (
                        "features",
                        feature_pipeline,
                        slice(0,None),
                    ),
                ],
                remainder="passthrough",
                )
        
        return preprocessing_pipeline

    def initialize_transformation(self, train_path, test_path):
        logging.info("Data Tranformation initialized.")
        try:
            logging.info("Reading train and test data.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Completed reading data.")

            unnecessary_cols = [
                "Dst Port",
                "Protocol",
                "Timestamp",
                "Src IP",
                "Src Port",
                "Dst IP",
                "Flow ID",
            ]

            train_df = train_df.drop(unnecessary_cols, axis = 1)
            test_df = test_df.drop(unnecessary_cols, axis = 1)
            logging.info("Dropped unnecessary columns from train, test.")

            target = 'label'

            preprocess_pipeline = self.get_datatransformer()
            logging.info("Loaded the preprocessing pipeline object.")

            le = LabelEncoder()
            logging.info("Initialized Label Encoder.")

            train_df[target] = le.fit_transform(train_df[target])
            test_df[target] = le.transform(test_df[target])
            logging.info("Encoded target column of train and test.")

            train_arr = preprocess_pipeline.fit_transform(train_df)
            test_arr = preprocess_pipeline.transform(test_df)
            logging.info("Applied preprocess pipleine to data.")

            feature_columns = train_df.columns
            logging.info("Converting transformed data array back to DataFrame.")

            train_transform_df = pd.DataFrame(train_arr, columns= feature_columns)
            test_transform_df = pd.DataFrame(test_arr, columns= feature_columns)
            logging.info("Conversion to DataFrame completed.")

            logging.info('Selecting the top 55 cols')
            imp_cols = select_top_55_features(train_transform_df)

            logging.info('Updated train and test Datasets with new updated Feature')
            train_transformed_df = train_transform_df[imp_cols]
            test_transformed_df = test_transform_df[imp_cols]
            logging.info(f"Traing Dataframe Columns: {train_transformed_df.columns}\nTesting Dataframe Columns: {test_transformed_df.columns}")

            save_object(file_path = self.data_tr_conf.preprocessor_obj_file_path, obj= preprocess_pipeline)
            logging.info(f"Saved preprocessing object.")

            return (train_transformed_df, test_transformed_df, self.data_tr_conf.preprocessor_obj_file_path)
            

        except Exception as e:
            raise CustomException(e, sys)