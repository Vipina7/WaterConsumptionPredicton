import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object,treat_outliers


class PredictPipeline:
    def __init__(self):
        pass

    def transform_test_df(self, test_data_path):
        try:
            scaler_path = 'artifacts/standard_scaler.pkl'
            rf_impute_path = 'artifacts/rf_impute.pkl'
            le_path = 'artifacts/label_encoder.pkl'

            scaler = load_object(file_path=scaler_path)
            rf_impute = load_object(file_path=rf_impute_path)
            le = load_object(file_path=le_path)

            # Load the test data
            test = pd.read_csv('dataset/test.csv', index_col='Timestamp', header=0)
            
            # remove the unwanted columns and impute the missing values and treat outliers
            test = test.drop(columns=['Humidity'])

            valid_classes = ["Low", "Middle", "Upper Middle", "Rich"]
            test['Income_Level'] = test['Income_Level'].apply(lambda x: x if x in valid_classes else 'Unknown')

            test['Apartment_Type'] = test['Apartment_Type'].fillna('Unknown')

            train_data = pd.read_csv(test_data_path)
            test['Temperature'] = test['Temperature'].fillna(round(train_data['Temperature'].mean(),2))

            test["Appliance_Usage"] = test["Appliance_Usage"].fillna(-1)

            test['is_null_amenities'] = test['Amenities'].isnull().astype('int')
            test_encoded = pd.get_dummies(test.drop(columns=['Amenities']), drop_first=True, dtype=int)
            test_impute = test_encoded[test_encoded['is_null_amenities']==1]
            test.loc[test['is_null_amenities']==1, 'Amenities'] = le.inverse_transform(rf_impute.predict(test_impute))

            treat_outliers(train=test, val=None)

            test['Guests'] = test['Guests'].apply(lambda x: -1 if x < 0 else x)
            test_df = pd.get_dummies(test, drop_first=True, dtype=int)

            test_df = test_df.drop(columns = 'Apartment_Type_Cottage')

            features_to_scale = ["Residents", "Temperature", "Water_Price", "Period_Consumption_Index", "Guests", "Appliance_Usage"]
            test_scaled = scaler.transform(test_df[features_to_scale])

            test_scaled_df = pd.DataFrame(test_scaled, columns=features_to_scale, index=test_df.index)
            test_data = pd.concat([test_scaled_df, test_df.drop(columns=features_to_scale)],  axis=1)
        
            return test_data
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict(self, test_data):
        try:
            model_path = 'artifacts/model.pkl'
            model = load_object(file_path=model_path)

            test_data['Water_Consumption_Prediction'] = model.predict(test_data)
            test_data.to_csv('artifacts/predicted_data.csv')

            return "Prediction Successful."
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    try:
        test_path = 'dataset/test.csv'
        prediction_obj = PredictPipeline()
        test_data = prediction_obj.transform_test_df(test_data_path=test_path)

        print(prediction_obj.predict(test_data))

    except Exception as e:
        raise CustomException(e,sys)

