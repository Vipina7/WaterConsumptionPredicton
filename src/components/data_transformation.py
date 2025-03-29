import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import impute_amenities, save_object, treat_outliers
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

@dataclass
class DataTransformationConfig:
    standard_scaler_path:str = os.path.join('artifacts','standard_scaler.pkl')
    label_encoder_path:str = os.path.join('artifacts','label_encoder.pkl')
    rf_impute_path:str = os.path.join('artifacts', 'rf_impute.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_obj(self):
        try:
            logging.info("Obtaining preprocessor object.")
            scaler = StandardScaler()
            le = LabelEncoder()

            return scaler, le
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, val_path):
        try:
            logging.info("Initiating data transformation.")
            train = pd.read_csv(train_path)
            val = pd.read_csv(val_path)
            logging.info("Train and validation sets successfully imported.")

            valid_classes = ["Low", "Middle", "Upper Middle", "Rich"]
            train['Income_Level'] = train['Income_Level'].apply(lambda x: x if x in valid_classes else 'Unknown')
            val['Income_Level'] = val['Income_Level'].apply(lambda x: x if x in valid_classes else 'Unknown')
            logging.info("Income Level optimized.")

            train['Apartment_Type'] = train['Apartment_Type'].fillna('Unknown')
            val['Apartment_Type'] = val['Apartment_Type'].fillna('Unknown')
            logging.info("Apartment type null values imputed.")

            train['Temperature'] = train['Temperature'].fillna(round(train['Temperature'].mean(),2))
            val['Temperature'] = val['Temperature'].fillna(round(train['Temperature'].mean(),2))
            logging.info("Temperature null values imputed.")

            train["Appliance_Usage"] = train["Appliance_Usage"].fillna(-1)
            val["Appliance_Usage"] = val["Appliance_Usage"].fillna(-1)
            logging.info("Appliance Usage null values imputed.")

            train['is_null_amenities'] = train['Amenities'].isnull().astype('int')
            val['is_null_amenities'] = val['Amenities'].isnull().astype('int')

            train_encoded = pd.get_dummies(train.drop(columns = ['Amenities']), drop_first=True, dtype=int)
            val_encoded = pd.get_dummies(val.drop(columns = ['Amenities']), drop_first=True, dtype=int)

            scaler_obj, impute_encoder_obj = self.get_preprocessor_obj()

            X_train_encoded = train_encoded[train_encoded['is_null_amenities']==0]
            X_test_encoded = train_encoded[train_encoded['is_null_amenities']==1].drop(columns = ['Water_Consumption'])
            y_train_encoded = impute_encoder_obj.fit_transform(train[train['is_null_amenities']==0]['Amenities'])
            val_encoded_test = val_encoded[val_encoded['is_null_amenities']==1].drop(columns=['Water_Consumption'])
            logging.info("Data prepped for advanced imputing of Amenities feature.")

            rf_impute = impute_amenities(X_train_encoded=X_train_encoded, y_train_encoded=y_train_encoded)
            train.loc[train['is_null_amenities']==1, 'Amenities'] = impute_encoder_obj.inverse_transform(rf_impute.predict(X_test_encoded))
            val.loc[val['is_null_amenities']==1, 'Amenities'] = impute_encoder_obj.inverse_transform(rf_impute.predict(val_encoded_test))
            logging.info("Amenities imputed successfully")

            treat_outliers(train=train, val=val)

            train['Guests'] = train['Guests'].apply(lambda x: -1 if x < 0 else x)
            val['Guests'] = val['Guests'].apply(lambda x : -1 if x < 0 else x)
            logging.info("Guests checked for errors.")

            train_encoded = pd.get_dummies(train, drop_first=True, dtype = int)
            val_encoded = pd.get_dummies(val, drop_first=True, dtype = int)

            train_encoded = train_encoded.drop(columns = 'Apartment_Type_Cottage')
            val_encoded = val_encoded.drop(columns = 'Apartment_Type_Cottage')

            features_to_scale = ["Residents", "Temperature", "Water_Price", "Period_Consumption_Index", "Guests", "Appliance_Usage"]
            train_scaled = scaler_obj.fit_transform(train[features_to_scale])
            val_scaled = scaler_obj.transform(val[features_to_scale])

            train_scaled_df = pd.DataFrame(train_scaled, columns=features_to_scale, index=train_encoded.index)
            val_scaled_df = pd.DataFrame(val_scaled, columns=features_to_scale, index=val_encoded.index)

            train_scaled = pd.concat([train_scaled_df, train_encoded.drop(columns=features_to_scale)],  axis=1)
            val_scaled = pd.concat([val_scaled_df, val_encoded.drop(columns=features_to_scale)], axis=1)
            logging.info("Data preparation for model training complete.")

            save_object(
                file_path=self.data_transformation_config.standard_scaler_path,
                obj = scaler_obj
            )

            save_object(
                file_path=self.data_transformation_config.label_encoder_path,
                obj = impute_encoder_obj
            )

            save_object(
                file_path=self.data_transformation_config.rf_impute_path,
                obj = rf_impute
            )

            return train_scaled, val_scaled
            
        except Exception as e:
            raise CustomException(e,sys)
