import os
import sys
import joblib
from src.exception import CustomException
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV 


def save_object(file_path, obj):
    """
    Function to save an object as a pickle file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train,
                            x_test, y_test, model, param):
    try:
        report = {}
        for i in range(len(list(model))):
            model_name = list(model.values())[i]
            parramters = param[list(model.keys())[i]]
            

            grid_search= GridSearchCV(estimator=model_name,
                                       param_grid=parramters,
                                       refit = True,
                                       cv=3, n_jobs=-1, verbose=2)
            
            grid_search.fit(x_train, y_train)
            model_name.set_params(**grid_search.best_params_)
            logging.info(f"Best parameters for {list(model.keys())[i]}: {grid_search.best_params_}")

            model_name.fit(x_train, y_train)

            y_train__pred= model_name.predict(x_train)
            y_test_pred = model_name.predict(x_test)

            train_model_score = r2_score(y_train, y_train__pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(model.keys())[i]] = test_model_score 

        return report
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Function to load a pickle file.
    """
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)
   
            
