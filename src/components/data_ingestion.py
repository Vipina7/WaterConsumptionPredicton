import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    val_data_path:str = os.path.join('artifacts','val.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def intiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            train = pd.read_csv('dataset/train.csv')
            test = pd.read_csv('dataset/test.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            logging.info('Artifacts folder created')

            logging.info("Train test split initiated for splitting the train set into train and validation sets")
            train_set, val_set = train_test_split(train, test_size=0.2, random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index = False, header = True)
            val_set.to_csv(self.data_ingestion_config.val_data_path, index = False, header = True)
            test.to_csv(self.data_ingestion_config.test_data_path, index = False, header = True)
            logging.info('Ingestion of data is completed')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.val_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    try:
        data_obj = DataIngestion()
        train_path, val_path, _ = data_obj.intiate_data_ingestion()

        transform_obj = DataTransformation()
        train_scaled, val_scaled = transform_obj.initiate_data_transformation(train_path=train_path, val_path=val_path)
        
        model_training_obj = ModelTrainer()
        print(model_training_obj.initiate_model_training(train_scaled, val_scaled))
    
    except Exception as e:
        raise CustomException(e,sys)