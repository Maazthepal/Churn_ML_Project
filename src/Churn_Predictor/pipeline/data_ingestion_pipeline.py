from src.Churn_Predictor.config.configuration import ConfigurationManager
from src.Churn_Predictor.components.data_ingestion import DataIngestion
from src.Churn_Predictor import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def  initiate_data_ingestion(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_data()
            data_ingestion.extract_zip_file()
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME} stage: {e}")
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME} stage")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.initiate_data_ingestion()
        logger.info(f"Completed {STAGE_NAME} stage")
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} stage: {e}")
        raise e