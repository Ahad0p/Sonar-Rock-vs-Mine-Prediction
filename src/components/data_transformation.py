import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')
    label_encoder_obj_file_path = os.path.join('artifact', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, feature_columns):
        """Returns preprocessing object for standardizing numeric data"""
        try:
            logging.info("Creating preprocessing pipeline for numerical features.")
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, feature_columns)
            ])
            logging.info("Preprocessing pipeline created successfully.")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info(f"Reading training data from: {train_path}")
            train_df = pd.read_csv(train_path, header=None)
            logging.info(f"Reading testing data from: {test_path}")
            test_df = pd.read_csv(test_path, header=None)

            logging.info("Splitting input features and target column.")
            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]

            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            logging.info("Getting preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object(X_train.columns)

            logging.info("Encoding target labels using LabelEncoder.")
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            logging.info("Applying preprocessing to training and testing features.")
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_transformed, np.array(y_train_encoded)]
            test_arr = np.c_[X_test_transformed, np.array(y_test_encoded)]

            logging.info(f"Saving preprocessing object at: {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            logging.info(f"Saving label encoder at: {self.data_transformation_config.label_encoder_obj_file_path}")
            save_object(self.data_transformation_config.label_encoder_obj_file_path, label_encoder)

            logging.info("Data transformation completed successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
