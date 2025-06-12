import sys
import pandas as pd
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def predict(self, features):
        try:
            logging.info("Predicting the data point")
            preprocessor_path = "artifacts/preprocessor.pkl"
            model_path = "artifacts/model.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info(f"Features before transform: \n{features}")
            transformed_feature = preprocessor.transform(features)
            predictions = model.predict(transformed_feature)
            logging.info(f"Prediction: {predictions}")
            return predictions
        except Exception as e:  
            raise CustomException(e, sys)
    
class CustomData:
    def __init__(self,
                gender: str,
                race_ethnicity: str,
                parental_level_of_education: str,
                lunch: str,
                test_preparation_course: str,
                writing_score: float,
                reading_score: float):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {"gender": [self.gender],
                                      "race_ethnicity": [self.race_ethnicity],
                                        "parental_level_of_education": [self.parental_level_of_education],
                                        "lunch": [self.lunch],
                                        "test_preparation_course": [self.test_preparation_course],
                                        "writing_score": [self.writing_score],
                                        "reading_score": [self.reading_score]}
            logging.info(f"Custom data input dict: {custom_data_input_dict}")

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


