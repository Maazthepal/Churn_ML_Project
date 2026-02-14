from src.Churn_Predictor.constants import *
from src.Churn_Predictor.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.Churn_Predictor.utils.common import read_yaml, create_directories
from pathlib import Path


class ConfigurationManager:
    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH,
                 schema_file_path=SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_slug=config.dataset_slug,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzipped_data_dir=config.unzipped_data_dir,
            all_schema=schema
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )
        return data_transformation_config 
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        model_trainer_config = self.config.model_trainer
        model_trainer_params = self.params.XGBBoost
        target_column = list(self.schema.TARGET_COLUMN.keys())[0]
        create_directories([model_trainer_config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=model_trainer_config.root_dir,
            train_data_path=model_trainer_config.train_data_path,
            test_data_path=model_trainer_config.test_data_path,
            model_name=model_trainer_config.model_name,
            target_column=target_column,
            learning_rate=model_trainer_params.learning_rate,
            max_depth=model_trainer_params.max_depth,
            n_estimators=model_trainer_params.n_estimators,
            scale_pos_weight=model_trainer_params.scale_pos_weight,
            subsample=model_trainer_params.subsample,
            colsample_bytree=model_trainer_params.colsample_bytree,
            random_state=model_trainer_params.random_state
        )
        
        return model_trainer_config
    