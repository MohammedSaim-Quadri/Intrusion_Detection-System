import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging  
from src.exception import CustomException  
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "IDS_data.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initialize_data_ingestion(self):
        logging.info("Data ingestion started....")
        try:
            df = pd.read_csv("dataset/train_data.csv")
            logging.info(f'Data has been read successfully. Shape of the dataset: {df.shape}')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)
            logging.info(f'Raw data saved at: {self.ingestion_config.raw_data_path}')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f'Data split completed. Training set size: {train_set.shape}, Test set size: {test_set.shape}')

            train_set.to_csv(self.ingestion_config.train_path, index=False, header=True)
            logging.info(f'Training data saved at: {self.ingestion_config.train_path}')

            # Saving the test set
            test_set.to_csv(self.ingestion_config.test_path, index=False, header=True)
            logging.info(f'Test data saved at: {self.ingestion_config.test_path}')

            return self.ingestion_config.test_path, self.ingestion_config.train_path
        
        except Exception as e:
            # Log the error and raise custom exception
            logging.error(f"Error occurred during data ingestion: {str(e)}")
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    test_path, train_path = obj.initialize_data_ingestion()
    data_transform = DataTransformation()
    data_transform.initialize_transformation(train_path, test_path)


