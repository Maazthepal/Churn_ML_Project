import os
import pandas as pd
import yaml
import mlflow
import optuna
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from dotenv import load_dotenv
from src.Churn_Predictor import logger
from src.Churn_Predictor.entity.config_entity import ModelTunerConfig
import plotly

load_dotenv()


class ModelTuner:
    def __init__(self, config: ModelTunerConfig):
        self.config = config
        
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function for Optuna to optimize
        
        trial: Optuna trial object
        Returns: F1 score (to be maximized)
        """
        # Define parameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
        }
        
        # Train model with suggested parameters
        model = XGBClassifier(
            **params,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=10,
            verbosity=0  # Suppress XGBoost output
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict and calculate F1 score
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        
        return f1
    
    def tune_hyperparameters(self):
        """
        Main tuning method using Optuna
        """
        logger.info("Starting hyperparameter tuning with Optuna")
        
        # Load data
        logger.info(f"Loading training data from: {self.config.train_data_path}")
        train_data = pd.read_csv(self.config.train_data_path)
        
        # Split features and target
        X = train_data.drop(columns=[self.config.target_column])
        y = train_data[self.config.target_column]
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Churn_Prediction_Optuna_Tuning")
        
        # Setup MLflow callback for Optuna
        from optuna.integration.mlflow import MLflowCallback
        
        mlflc = MLflowCallback(
            tracking_uri=self.config.mlflow_uri,
            metric_name="f1_score"
        )
        
        # Create Optuna study
        logger.info(f"Creating Optuna study: {self.config.study_name}")
        study = optuna.create_study(
            direction='maximize',  # Maximize F1 score
            study_name=self.config.study_name,
            sampler=optuna.samplers.TPESampler(seed=42)  # Tree-structured Parzen Estimator
        )
        
        # Run optimization
        logger.info(f"Starting optimization with {self.config.n_trails} trials...")
        logger.info("This may take a while...")
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.config.n_trails,
            callbacks=[mlflc],
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info("=" * 70)
        logger.info("OPTIMIZATION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Best trial number: {study.best_trial.number}")
        logger.info(f"Best F1 score: {best_score:.4f}")
        logger.info(f"Best parameters:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info("=" * 70)
        
        # Save best parameters to YAML
        self._save_best_params(best_params)
        
        # Generate optimization visualizations
        self._generate_visualizations(study)
        
        return best_params, best_score
    
    def _save_best_params(self, best_params):
        """Save best parameters to YAML file"""
        # Format parameters for YAML (matching params.yaml structure)
        yaml_params = {
            'XGBBoost': {
                'n_estimators': int(best_params['n_estimators']),
                'max_depth': int(best_params['max_depth']),
                'learning_rate': float(best_params['learning_rate']),
                'scale_pos_weight': float(best_params['scale_pos_weight']),
                'subsample': float(best_params['subsample']),
                'colsample_bytree': float(best_params['colsample_bytree']),
                'min_child_weight': int(best_params['min_child_weight']),
                'gamma': float(best_params['gamma']),
                'reg_alpha': float(best_params['reg_alpha']),
                'reg_lambda': float(best_params['reg_lambda']),
                'random_state': 42
            }
        }
        
        # Save to file
        with open(self.config.best_params_path, 'w') as f:
            yaml.dump(yaml_params, f, default_flow_style=False)
        
        logger.info(f"Best parameters saved to: {self.config.best_params_path}")
        logger.info("You can now copy these to params.yaml to use them in training!")
    
    def _generate_visualizations(self, study):
        """Generate Optuna visualization plots"""
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_slice
            )
                        
            # Create visualizations directory
            viz_dir = os.path.join(self.config.root_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # 1. Optimization history
            fig1 = plot_optimization_history(study)
            fig1.write_html(os.path.join(viz_dir, 'optimization_history.html'))
            
            # 2. Parameter importances
            fig2 = plot_param_importances(study)
            fig2.write_html(os.path.join(viz_dir, 'param_importances.html'))
            
            # 3. Slice plot
            fig3 = plot_slice(study)
            fig3.write_html(os.path.join(viz_dir, 'param_slice.html'))
            
            logger.info(f"Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")