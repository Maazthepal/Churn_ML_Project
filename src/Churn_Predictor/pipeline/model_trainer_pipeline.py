from src.Churn_Predictor.config.configuration import ConfigurationManager
from src.Churn_Predictor.components.model_trainer import ModelTrainer
from src.Churn_Predictor import logger

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        try:
            config = ConfigurationManager()
            modeltrainer_config = config.get_model_trainer_config()
            modeltrainer = ModelTrainer(config=modeltrainer_config)
            modeltrainer.train_model()
        except Exception as e:
            logger.exception(f"Error occurred: {e}")

if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME} stage")
        model_trainer = ModelTrainerPipeline()
        model_trainer.initiate_model_trainer()
        logger.info(f"Completed {STAGE_NAME} stage")
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} stage: {e}")
        raise e
    