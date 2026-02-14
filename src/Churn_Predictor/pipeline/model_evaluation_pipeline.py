from dotenv import load_dotenv
from src.Churn_Predictor import logger
from src.Churn_Predictor.config.configuration import ConfigurationManager
from src.Churn_Predictor.components.model_evaluation import ModelEvaluation

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try:
            from dotenv import load_dotenv
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.initiate_model_evaluation()
        except Exception as e:
            logger.exception(f"Error in {STAGE_NAME}: {e}")
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME}")
        model_evaluation_pipeline = ModelEvaluationPipeline()
        model_evaluation_pipeline.initiate_model_evaluation()
        logger.info(f"Completed {STAGE_NAME}")
    except Exception as e:
        logger.exception(f"Error in {STAGE_NAME}: {e}")
        raise e