from src.Churn_Predictor.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline
from src.Churn_Predictor.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.Churn_Predictor.pipeline.data_validation_pipeline import DataValidationPipeline
from src.Churn_Predictor.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.Churn_Predictor.pipeline.model_tuner_pipeline import ModelTunerPipeline
from src.Churn_Predictor.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.Churn_Predictor import logger
from dotenv import load_dotenv

load_dotenv()

RUN_HYPERPARAMETER_TUNING = True

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

if RUN_HYPERPARAMETER_TUNING:
    STAGE_NAME = "Model Hyperparameter Tuning Stage"
    try:
        logger.info(f">>>>>>> {STAGE_NAME} started <<<<<<<")
        logger.info(" Running hyperparameter tuning with Optuna...")
        logger.info("This will take 30-60 minutes for 100 trials...")
        
        model_tuner = ModelTunerPipeline()
        best_params, best_score = model_tuner.initiate_model_tuning()
        
        logger.info(f"Best validation F1 score: {best_score:.4f}")
        logger.info(f"Best parameters saved to: artifacts/model_tuner/best_params.yaml")
        logger.info(f"Copy these params to params.yaml for production use")
        logger.info(f">>>>>>> {STAGE_NAME} completed <<<<<<<\nx==========x\n")
    except Exception as e:
        logger.exception(e)
        logger.warning("Tuning failed! Continuing with default params from params.yaml")
else:
    logger.info("Skipping Hyperparameter Tuning")
    logger.info("Using parameters from params.yaml")
    logger.info("(Set RUN_HYPERPARAMETER_TUNING=True to enable tuning)\n")

STAGE_NAME = "Model Trainer stage"
try:
   logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<") 
   model_trainer = ModelTrainerPipeline()
   model_trainer.initiate_model_trainer()
   logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<") 
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.initiate_model_evaluation()
    logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e