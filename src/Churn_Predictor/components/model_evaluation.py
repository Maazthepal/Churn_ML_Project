import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.Churn_Predictor.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
from src.Churn_Predictor import logger
from src.Churn_Predictor.utils.common import save_json

# Load environment variables at module level
load_dotenv()


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_model(self, actual, pred):
        """Calculate evaluation metrics"""
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    def setup_mlflow_auth(self):
        """Setup MLflow authentication from environment variables"""
        username = os.getenv('MLFLOW_TRACKING_USERNAME')
        password = os.getenv('MLFLOW_TRACKING_PASSWORD')
        
        if not username or not password:
            logger.warning("MLflow credentials not found in environment. Using local tracking only.")
            return False
        
        # Set credentials as environment variables for MLflow
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
        
        logger.info(f"MLflow authentication configured for user: {username}")
        return True
    
    def initiate_model_evaluation(self):
        """Evaluate model and log to MLflow"""
        logger.info("Starting model evaluation")
        
        # Load test data and model
        logger.info(f"Loading test data from: {self.config.test_data_path}")
        test_data = pd.read_csv(self.config.test_data_path)
        
        logger.info(f"Loading model from: {self.config.model_path}")
        model = joblib.load(self.config.model_path)

        # Prepare test data
        test_x = test_data.drop(self.config.target_column, axis=1)
        test_y = test_data[self.config.target_column]
        
        logger.info(f"Test data shape: {test_x.shape}")
        logger.info(f"Target column: {self.config.target_column}")

        # Setup MLflow authentication
        auth_success = self.setup_mlflow_auth()
        
        # Set tracking and registry URI
        logger.info(f"Setting MLflow tracking URI: {self.config.mlflow_uri}")
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        
        # Set experiment
        mlflow.set_experiment("Churn_Prediction_Model_Evaluation")
        
        # Get tracking URL type
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        try:
            with mlflow.start_run():
                logger.info("MLflow run started")
                
                # Make predictions
                pred = model.predict(test_x)
                
                # Calculate metrics
                metrics = self.evaluate_model(test_y, pred)
                
                logger.info("=" * 50)
                logger.info("MODEL EVALUATION METRICS")
                logger.info("=" * 50)
                logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
                logger.info(f"Precision: {metrics['precision']:.4f}")
                logger.info(f"Recall:    {metrics['recall']:.4f}")
                logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
                logger.info("=" * 50)

                # Log parameters and metrics to MLflow
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics(metrics)
                
                # Save metrics locally
                save_json(path=Path(self.config.metric_file_name), data=metrics)
                logger.info(f"Metrics saved to: {self.config.metric_file_name}")

                # Log model to MLflow
                if tracking_url_type_store != "file":
                    # Remote tracking (DagsHub)
                    logger.info("Logging model to remote MLflow server")
                    mlflow.sklearn.log_model(
                        model, 
                        "model",
                        registered_model_name="Churn_Prediction_Model"
                    )
                else:
                    # Local tracking
                    logger.info("Logging model to local MLflow")
                    mlflow.sklearn.log_model(model, "model")
                
                logger.info("Model evaluation completed successfully")
                
        except Exception as e:
            logger.error(f"Error during MLflow tracking: {e}")
            logger.warning("Continuing without MLflow tracking")
            
            # Save metrics locally even if MLflow fails
            metrics = self.evaluate_model(test_y, model.predict(test_x))
            save_json(path=self.config.metric_file_name, data=metrics)
            logger.info(f"Metrics saved locally to: {self.config.metric_file_name}")
            
            raise e