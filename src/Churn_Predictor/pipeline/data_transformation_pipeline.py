from src.Churn_Predictor import logger
from src.Churn_Predictor.components.data_transformation import DataTransformation
from src.Churn_Predictor.config.configuration import ConfigurationManager

STAGE_NAME = "Data Transformation Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            df = data_transformation.initiate_data_transformation()
            df = data_transformation.initiate_data_preprocessing(df)
            data_transformation.initiate_train_test_split(df)
        except Exception as e:
            logger.exception(f"An error occurred during data transformation: {e}")
        
if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME} stage")
        data_transformation = DataTransformationPipeline()
        data_transformation.initiate_data_transformation()
        logger.info(f"Completed {STAGE_NAME} stage")
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} stage: {e}")
        raise e