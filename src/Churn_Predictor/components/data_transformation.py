import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from src.Churn_Predictor.entity.config_entity import DataTransformationConfig
from src.Churn_Predictor import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    

    def initiate_data_transformation(self):
        logger.info("Reading data from csv file")
        df = pd.read_csv(self.config.data_path)


        logger.info("Dropping the 'customerID' column from the dataset")
        df.drop('customerID', axis=1, inplace=True)

        logger.info("Finding Null values in the dataset")
        null_counts = df.isnull().sum()
        logger.info(f"Null value counts:\n{null_counts}")

        logger.info("Converting 'TotalCharges' to numeric, coercing errors to NaN")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        logger.info("Finding Null values in 'TotalCharges' after conversion")
        total_charges_null_count = df['TotalCharges'].isnull().sum()
        logger.info(f"Null value count in 'TotalCharges' after conversion: {total_charges_null_count}")

        logger.info("Filling mean_values in 'TotalCharges' Null Rows")
        mean_total_charges = df['TotalCharges'].mean()
        df['TotalCharges'].fillna(mean_total_charges, inplace=True)
        logger.info("Mean value filled in 'TotalCharges' Null Rows")

        logger.info("Finding the Duplicated values in the dataset")
        duplicated_count = df.duplicated().sum()
        logger.info(f"Duplicated value count: {duplicated_count}")

        logger.info("Dropping the Duplicated values in the dataset")
        df.drop_duplicates(keep='first', inplace=True)
        logger.info("Duplicated values dropped")

        return df
    
    def initiate_data_preprocessing(self, df):
        logger.info("Preparing data for preproceesing")
        df = df.replace({
            'No internet service': 'No',
            'No phone service': 'No'
            })
        
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
               'PaperlessBilling', 'MultipleLines', 'OnlineSecurity', 
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 
               'StreamingTV', 'StreamingMovies']
        
        multi_cols = ['InternetService', 'Contract', 'PaymentMethod']

        le_target = LabelEncoder()
        df['Churn'] = le_target.fit_transform(df['Churn'])

        for cols in binary_cols:
            le = LabelEncoder()
            df[cols] = le.fit_transform(df[cols])
        
        df = pd.get_dummies(data=df, columns=multi_cols, drop_first=True)
        logger.info("Data preprocessing completed")

        logger.info(f"Final info of the dataframe after preprocessing: {df.shape}")
        logger.info(f"Final columns of the dataframe after preprocessing: {df.columns.to_list()}")
        return df

    def initiate_train_test_split(self, df):
        logger.info("Initiating train test split")
        train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])
        logger.info("Train test split completed")

        train.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)

        logger.info(f"Train and test data saved in {self.config.root_dir}")
        logger.info(f"Train data shape: {train.shape}")
        logger.info(f"Test data shape: {test.shape}")
        