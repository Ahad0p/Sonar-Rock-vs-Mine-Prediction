import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting input data into train and test sets")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define all models here
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "KNN": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0),
                "CatBoost": CatBoostClassifier(verbose=False, train_dir="catboost_logs"),
                "AdaBoost": AdaBoostClassifier(),
            }

            # Define hyperparameters for each model
            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10.0]
                },
                "Random Forest": {
                    "n_estimators": [50, 100],
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [3, 5, 10],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100],
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7]
                },
                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100],
                },
                "CatBoost": {
                    "depth": [4, 6],
                    "iterations": [50, 100],
                    "learning_rate": [0.01, 0.1]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                }
            }

            logging.info("Starting model evaluation")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model = model_report["best_model"]
            best_model_name = model_report["best_model_name"]
            best_model_score = model_report["scores"][best_model_name]

            # Log the best model found
            logging.info(f"Best Model: {best_model_name} with accuracy: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No good model found with enough performance.")

            # Save the trained model
            logging.info(f"Saving the best model: {best_model_name}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict using the best model
            predictions = best_model.predict(X_test)

            # Calculate and log accuracy
            acc = accuracy_score(y_test, predictions)
            logging.info(f"Test Accuracy of the best model: {acc * 100:.2f}%")

            return acc

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise CustomException(e, sys)
