import pandas as pd
from xgboost import XGBClassifier
from src.Churn_Predictor.entity.config_entity import ModelTrainerConfig
from src.Churn_Predictor import logger
import joblib
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_model(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop(columns=[self.config.target_column])
        y_train = train_data[self.config.target_column]
        X_test = test_data.drop(columns=[self.config.target_column])
        y_test = test_data[self.config.target_column]

        xgb = XGBClassifier(
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            n_estimators=self.config.n_estimators,
            scale_pos_weight=self.config.scale_pos_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state
        )
        xgb.fit(X_train, y_train)
        logger.info("Model training completed.")
    
        joblib.dump(xgb, os.path.join(self.config.root_dir, self.config.model_name))