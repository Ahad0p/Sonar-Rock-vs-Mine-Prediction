from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ ,_= transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    print("Model Accuracy:", model_trainer.initiate_model_trainer(train_arr, test_arr))
