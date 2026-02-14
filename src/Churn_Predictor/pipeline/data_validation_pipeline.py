from src.Churn_Predictor.config.configuration import ConfigurationManager
from src.Churn_Predictor.components.data_validation import DataValidation
from src.Churn_Predictor import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.validate_all_columns()
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME} stage: {e}")
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME} stage")
        data_validation = DataValidationPipeline()
        data_validation.initiate_data_validation()
        logger.info(f"Completed {STAGE_NAME} stage")
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} stage: {e}")
        raise e
    