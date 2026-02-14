from src.Churn_Predictor.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.Churn_Predictor.pipeline.data_validation_pipeline import DataValidationPipeline
from src.Churn_Predictor.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.Churn_Predictor.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.Churn_Predictor import logger


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.error(f"Error in {STAGE_NAME} stage: {e}")
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<")
    data_validation = DataValidationPipeline()
    data_validation.initiate_data_validation()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.error(f"Error in {STAGE_NAME} stage: {e}")
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>> Stage: {STAGE_NAME} started <<<")
    data_transformation = DataTransformationPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.error(f"Error in {STAGE_NAME} stage: {e}")
    raise e

STAGE_NAME = "Model Trainer stage"
try:
   logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<") 
   model_trainer = ModelTrainerPipeline()
   model_trainer.initiate_model_trainer()
   logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e