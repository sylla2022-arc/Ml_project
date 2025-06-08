
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.utils import save_object

from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj = os.path.join("artifacts", 'preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_onject(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            caterigorical_columns = ['gender', 'race_ethnicity',
                                      'parental_level_of_education', 
                                      'lunch',
                                        'test_preparation_course']

            num_pipeline = Pipeline( steps = [('imputer', SimpleImputer(strategy='median')),
                                     ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy='most_frequent')),
                                             ('ohe', OneHotEncoder()),
                                     ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info('Numerical column scaling completed')
            logging.info('Categorical column scaling completed')


            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, caterigorical_columns)
            ])

            return preprocessor
            
        except Exception as e:
            raise CustomException (e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation initiated")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_onject()


            target_column_name = 'math_score'
            numerical_features = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Obtained input and target features for train and test data")

            # Transforming the input features

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applied preprocessing object on train and test data")
            # Transforming the target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Concatenated input and target features for train and test data")
            # Saving the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj,
                obj=preprocessing_obj
            )
            logging.info("Saved preprocessing object")
            return (
                train_arr,
                  test_arr, 
                  self.data_transformation_config.preprocessor_obj
            )
            


        except Exception as e:
            raise CustomException(e, sys)
    



