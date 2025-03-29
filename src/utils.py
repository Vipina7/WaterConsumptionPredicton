import os
import sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

def impute_amenities(X_train_encoded, y_train_encoded):
    try:
        rf_impute = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_impute.fit(X_train_encoded, y_train_encoded)

        return rf_impute
    
    except Exception as e:
        raise CustomException(e,sys)
    
def treat_outliers(train,val):
    num_cols = [col for col in train.columns if train[col].dtype != 'O' and col not in ['Water_Consumption','Appliance_Usage','Guests']]
    for col in num_cols:
        q3 = train[col].quantile(0.75)
        q1 = train[col].quantile(0.25)
        iqr = q3 - q1
        upper_bound = round((q3 + (1.5 * iqr)),2)
        lower_bound = round((q1 - (1.5 * iqr)),2)

        train[col] = train[col].apply(lambda x : upper_bound if x > upper_bound else x)
        train[col] = train[col].apply(lambda x : lower_bound if x < lower_bound else x)

    for col in num_cols:
        q3 = val[col].quantile(0.75)
        q1 = val[col].quantile(0.25)
        iqr = q3 - q1
        upper_bound = round((q3 + (1.5 * iqr)),2)
        lower_bound = round((q1 - (1.5 * iqr)),2)

        val[col] = val[col].apply(lambda x : upper_bound if x > upper_bound else x)
        val[col] = val[col].apply(lambda x : lower_bound if x < lower_bound else x)
    logging.info('Outliers treatment successful.')
    
    return "Ouliers treated."

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Saved preprocessing object - {obj}")

    except Exception as e:
        raise CustomException(e, sys)