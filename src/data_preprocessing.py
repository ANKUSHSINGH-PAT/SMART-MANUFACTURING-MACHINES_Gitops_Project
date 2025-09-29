import sys
import pandas as pd 
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.custom_exception import CustomException
import os

logger=logging.getLogger(__name__)
class DataPreprocessor:
    def __init__(self,input_file_path:str,output_file_path:str):
        self.input_file_path=input_file_path
        self.output_file_path=output_file_path
        self.df=None
        self.features=None

        os.makedirs(os.path.dirname(self.output_file_path),exist_ok=True)
        logger.info(f"Output directory created at {os.path.dirname(self.output_file_path)}")
        logger.info("DataPreprocessor instance created")
    
    def load_data(self):
        try:
            self.df=pd.read_csv(self.input_file_path)
            logger.info(f"Data loaded successfully from {self.input_file_path}")
        except Exception as e:
            logger.error(f"Error loading data from {self.input_file_path}: {e}")
            raise CustomException(e,sys)
        
    def preprocess_data(self):
        try:
            self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"] , errors='coerce')
            self.df.dropna(inplace=True)
            categorical_cols = ['Operation_Mode','Efficiency_Status']
            for col in categorical_cols:
                self.df[col] = self.df[col].astype('category')
        
            self.df.reset_index(drop=True,inplace=True)

            self.df["Year"] = self.df["Timestamp"].dt.year
            self.df["Month"] = self.df["Timestamp"].dt.month
            self.df["Day"] = self.df["Timestamp"].dt.day

            self.df["Hour"] = self.df["Timestamp"].dt.hour
            logger.info("Missing values handled and index reset")

            columnt_encoder=['Operation_Mode','Efficiency_Status']
            labelencoder=LabelEncoder()
            for col in columnt_encoder:
                self.df[col]=labelencoder.fit_transform(self.df[col])
            logger.info("Categorical columns encoded successfully")

            self.df.drop(columns=['Timestamp','Machine_ID'],inplace=True)
            logger.info("Unnecessary columns dropped successfully")
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException(e,sys)
    def split_and_scale(self):
        try:
            features = [
        'Operation_Mode', 'Temperature_C', 'Vibration_Hz',
        'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
        'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
        'Predictive_Maintenance_Score', 'Error_Rate_%','Year', 'Month', 'Day', 'Hour'
            ]
            x=self.df[features]
            y=self.df['Efficiency_Status']

            scaler=StandardScaler()
            x_scaled=scaler.fit_transform(x)
        

            x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
            logger.info("Data split into training and testing sets successfully")
            joblib.dump(x_train,os.path.join(self.output_file_path,'x_train.pkl'))
            joblib.dump(x_test,os.path.join(self.output_file_path,'x_test.pkl'))
            joblib.dump(y_train,os.path.join(self.output_file_path,'y_train.pkl'))
            joblib.dump(y_test,os.path.join(self.output_file_path,'y_test.pkl'))
            logger.info("Training and testing sets saved successfully")

            joblib.dump(scaler,os.path.join(self.output_file_path,'scaler.pkl'))
            logger.info("Scaler object saved successfully")
        except Exception as e:
            logger.error(f"Error during data splitting and scaling: {e}")

    def run_preprocessing(self):
        self.load_data()
        self.preprocess_data()
        self.split_and_scale()
        logger.info("Data preprocessing completed successfully")

if __name__=="__main__":
    input_file_path='artifacts/raw/data.csv'
    output_file_path='artifacts/processed/'
    preprocessor=DataPreprocessor(input_file_path,output_file_path)
    preprocessor.run_preprocessing()




