import os
import sys
import joblib
from src.exception import CustomException
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
                            x_test, y_test, model):
    try:
        report = {}
        for i in range(len(list(model))):
            model_name = list(model.values())[i]
            model_name.fit(x_train, y_train)

            y_train__pred= model_name.predict(x_train)
            y_test_pred = model_name.predict(x_test)

            train_model_score = r2_score(y_train, y_train__pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(model.keys())[i]] = test_model_score 

        return report
    except Exception as e:
        raise CustomException(e, sys)
   
            
