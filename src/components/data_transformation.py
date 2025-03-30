import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import impute_amenities, save_object, treat_outliers
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
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

            train['Timestamp'] = pd.to_datetime(train['Timestamp'], format = "%d/%m/%Y %H")
            val['Timestamp'] = pd.to_datetime(val['Timestamp'], format = "%d/%m/%Y %H")

            train['Week'] = train['Timestamp'].dt.weekday
            val['Week'] = val['Timestamp'].dt.weekday
            logging.info("Week added.")

            train['Month'] = train['Timestamp'].dt.month
            val['Month'] = val['Timestamp'].dt.month
            logging.info("Month added.")

            train['Day'] = train['Timestamp'].dt.day
            val['Day'] = val['Timestamp'].dt.day
            logging.info("Day added.")

            train = train.drop(columns = ['Timestamp'])
            val = val.drop(columns = ['Timestamp'])
            logging.info("Timestamp dropped.")

            train['Humidity'] = pd.to_numeric(train['Humidity'], errors= 'coerce')
            val['Humidity'] = pd.to_numeric(val['Humidity'], errors= 'coerce')
            logging.info("Humidity corrected for negative values")

            train['Humidity'] = train['Humidity'].fillna(round(train['Humidity'].median(),2))
            val['Humidity'] = val['Humidity'].fillna(round(train['Humidity'].median(),2))
            logging.info("Humidity imputed.")

            valid_classes = ["Low", "Middle", "Upper Middle", "Rich"]
            train['Income_Level'] = train['Income_Level'].apply(lambda x: x if x in valid_classes else 'Unknown')
            val['Income_Level'] = val['Income_Level'].apply(lambda x: x if x in valid_classes else 'Unknown')

            train['Income_Level'] = train['Income_Level'].replace({'Unknown':1, 'Low':2, 'Middle':3, 'Upper Middle':4, 'Rich':5})
            val['Income_Level'] = val['Income_Level'].replace({'Unknown':1, 'Low':2, 'Middle':3, 'Upper Middle':4, 'Rich':5})
            logging.info("Income Level optimized and encoded.")

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

            train_encoded['Temp_WaterPrice_Interaction'] = train_encoded['Temperature'] * train_encoded['Water_Price']
            val_encoded['Temp_WaterPrice_Interaction'] = val_encoded['Temperature'] * val_encoded['Water_Price']
            logging.info("Temperature and water price interaction added")

            train_encoded['Guests_ApplianceUsage_Interaction'] = train_encoded['Guests'] * train_encoded['Appliance_Usage']
            val_encoded['Guests_ApplianceUsage_Interaction'] = val_encoded['Guests'] * val_encoded['Appliance_Usage']
            logging.info("Guests and appliance usage interaction added")

            train_encoded['ApplianceUsage_Squared'] = train_encoded['Appliance_Usage'] ** 2
            val_encoded['ApplianceUsage_Squared'] = val_encoded['Appliance_Usage'] ** 2
            logging.info("Appliance usage squared added.")

            train_encoded['Month_sin'] = np.sin(2 * np.pi * train_encoded['Month'] / 12)
            val_encoded['Month_sin'] = np.sin(2 * np.pi * val_encoded['Month'] / 12)
            logging.info("Cyclical feature for month added.")

            train_encoded['Day_sin'] = np.sin(2 * np.pi * train_encoded['Day'] / 31)
            val_encoded['Day_sin'] = np.sin(2 * np.pi * val_encoded['Day'] / 31)
            logging.info("Day cyclical feature added.")

            features_to_remove = [
                "Apartment_Type_Unknown",
                "Amenities_Jacuzzi",
                "Apartment_Type_Bungalow",
                "Amenities_Swimming Pool",
                "Apartment_Type_Detached",
                "Apartment_Type_Cottage",
                "Month",
                "Day"
                ]

            train_encoded = train_encoded.drop(columns = features_to_remove)
            val_encoded = val_encoded.drop(columns = features_to_remove)

            features_to_scale = ["Residents", "Temperature", "Humidity", "Water_Price", 
                                 "Period_Consumption_Index", "Guests", "Temp_WaterPrice_Interaction",
                                 "Guests_ApplianceUsage_Interaction", "ApplianceUsage_Squared", "Month_sin", "Day_sin"]
            
            train_scaled = scaler_obj.fit_transform(train_encoded[features_to_scale])
            val_scaled = scaler_obj.transform(val_encoded[features_to_scale])

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
