import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

class ModelTrainerConfig:
    trained_model_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def generate_train_test(self, train_scaled, val_scaled):
        target = 'Water_Consumption'
        X_train, y_train = train_scaled.drop(columns = target), train_scaled[target]
        X_test, y_test = val_scaled.drop(columns = target), val_scaled[target]
        logging.info("Train and test sets generated")

        return (X_train, X_test, y_train, y_test)

    def initiate_model_training(self, train_scaled, val_scaled):
        try:
            models = {
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest": RandomForestRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "Linear Regression": LinearRegression(),
                        "SVR": SVR(),
                        "XGBRegressor": XGBRegressor(objective='reg:squarederror'),
                        "AdaBoost Regressor": AdaBoostRegressor(),
                    }

            
            # Hyperparameter Grid (for Multi-Output Models)
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.7, 0.75, 0.8, 0.85, 0.9],
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': [1.0, 'sqrt', 'log2'],
                    'n_estimators': [32, 64, 128, 256]
                },
                "Linear Regression": {},
                "SVR": {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto']
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [32, 64, 128, 256]
                }
                ,
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [32, 64, 128, 256]
                }
            }

            X_train, X_test, y_train, y_test = self.generate_train_test(train_scaled=train_scaled, val_scaled=val_scaled)
            train_report_df, test_report_df, best_estimators = evaluate_models(X_train, X_test, y_train, y_test, models, params)

            best_model_name = test_report_df['RMSE'].idxmin()
            best_model = best_estimators[best_model_name]
            
            save_object(
                file_path = self.model_trainer_config.trained_model_path,
                obj = best_model
            )
            logging.info('Saved the model')

            predicted=best_model.predict(X_test)

            return np.sqrt(mean_squared_error(y_test, predicted))
        
        except Exception as e:
            raise CustomException(e, sys)