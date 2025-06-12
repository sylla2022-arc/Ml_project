import os
import sys
import numpy as np
import pandas as pd
import joblib

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.utils import evaluate_models, save_object
@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer initiated")
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
       
                },
                "Random Forest":{

                    'n_estimators': [8,16,32,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,.75,0.8,0.85,0.9],
        
                    'n_estimators': [8,16,32,64,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.05,.001],
                    'n_estimators': [8,16,32,64,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,256]
                }
                
            }


            model_report : dict = evaluate_models(x_train = X_train,y_train =  y_train,
                                                 x_test=  X_test, y_test= y_test, model = models, param = params)
            logging.info(f"Model report: {model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            
            logging.info("Saving the best model")
            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model
            )
            pred = best_model.predict(X_test)
            r2_sqaure = r2_score(y_test, pred)
            logging.info(f"R2 score of the best model: {r2_sqaure}")

            return r2_sqaure, best_model

        except Exception as e:
            raise CustomException(e, sys)
    