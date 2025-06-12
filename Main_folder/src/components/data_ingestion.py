import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
#print(f"Root directory: {root_dir}")
if root_dir not in sys.path:
    sys.path.append(root_dir)

    
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
        
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train and test split completed")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train and test data saved")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    logging.info("Starting data ingestion process")
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Train data saved at: {train_data}")
    print(f"Test data saved at: {test_data}")
    logging.info("Data ingestion completed successfully")

    logging.info("Starting data transformation process")
    data_transformation = DataTransformation()
    train_array, test_array, _  = data_transformation.initiate_data_transformation(train_data, test_data)
    logging.info("Data transformation completed successfully")

    logging.info('Model trainer initiated')
    model_trainer = ModelTrainer()
    r2_squr, _ = model_trainer.initiate_model_trainer(train_array=train_array, 
                                         test_array = test_array)
    print(f"R2 Score of the best model: ", r2_squr)
    logging.info("Model training completed successfully")
    logging.info("End of the script")
    