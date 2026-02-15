from src.Churn_Predictor import logger
from src.Churn_Predictor.config.configuration import ConfigurationManager
from src.Churn_Predictor.components.model_tuner import ModelTuner


STAGE_NAME = "Model Tuner Stage"

class ModelTunerPipeline:
    def __init__(self):
        pass

    def initiate_model_tuning(self):
        try:
            config = ConfigurationManager()
            model_tuner_config = config.get_model_tuner_config()
            model_tuner = ModelTuner(config=model_tuner_config)
            best_params, best_score = model_tuner.tune_hyperparameters()
            return best_params, best_score
    
            logger.info(f"Tuning Completed! Best F1 score: {best_score:.4f}")
            print(f"The Best Params: {best_params}: {best_score}")

        except Exception as e:
            logger.exception(f"Error in {STAGE_NAME}: {e}")
            raise e
        

if __name__ == '__main__':
    try:
        logger.info(f">>> stage: {STAGE_NAME} started <<<")
        obj = ModelTunerPipeline()
        best_params, best_score = obj.initiate_model_tuning()
        logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e