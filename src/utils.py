import os
import sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

def impute_amenities(X_train_encoded, y_train_encoded):
    try:
        X_train_encoded = X_train_encoded.drop(columns = ['Water_Consumption'])
        rf_impute = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_impute.fit(X_train_encoded, y_train_encoded)

        return rf_impute
    
    except Exception as e:
        raise CustomException(e,sys)
    
def treat_outliers(train,val=None):
    num_cols = [col for col in train.columns if train[col].dtype != 'O' and col not in ['Water_Consumption','Appliance_Usage','Guests']]
    for col in num_cols:
        q3 = train[col].quantile(0.75)
        q1 = train[col].quantile(0.25)
        iqr = q3 - q1
        upper_bound = round((q3 + (1.5 * iqr)),2)
        lower_bound = round((q1 - (1.5 * iqr)),2)

        train[col] = train[col].apply(lambda x : upper_bound if x > upper_bound else x)
        train[col] = train[col].apply(lambda x : lower_bound if x < lower_bound else x)

    if val is not None:
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
    
def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    try:
        train_report = dict()
        test_report = dict()
        best_estimators = dict()

        logging.info('Model training initiated')
        for i in tqdm(range(len(list(models)))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            params=param[model_name]

            gs = GridSearchCV(model,params,cv=3, n_jobs=-1)
            gs.fit(X_train,y_train)

            best_estimators[model_name] = gs.best_estimator_

            y_train_pred = gs.best_estimator_.predict(X_train)

            y_test_pred = gs.best_estimator_.predict(X_test)

            train_model_score = []
            test_model_score = []

            #train model scores
            rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            mae = mean_absolute_error(y_train, y_train_pred)
            r2 = r2_score(y_train, y_train_pred)

            train_model_score.extend([rmse, mae, r2])
            train_report[list(models.keys())[i]] = train_model_score

            #test model scores
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae_test = mean_absolute_error(y_test, y_test_pred)
            r2_test = r2_score(y_test, y_test_pred)
                
            test_model_score.extend([rmse_test, mae_test, r2_test])

            test_report[list(models.keys())[i]] = test_model_score
            
        train_report_df = pd.DataFrame.from_dict(train_report, orient = 'index', columns = ["RMSE", "MAE", "R²"])
        test_report_df = pd.DataFrame.from_dict(test_report, orient = 'index', columns = ["RMSE", "MAE", "R²"])

        train_report_df.to_csv('artifacts/train_model_performances.csv')    
        test_report_df.to_csv('artifacts/test_model_performance.csv')
        
        return train_report_df, test_report_df, best_estimators
         
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)