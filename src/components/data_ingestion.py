import os
import sys
from src.logger import logging
from src.exception import CustomException
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
            
            train = train.drop(columns=['Timestamp','Humidity'], axis=1)
            test = test.drop(columns=['Humidity'], axis=1)
            logging.info('Dropping the necessary columns')

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
    data_obj = DataIngestion()
    data_obj.intiate_data_ingestion()