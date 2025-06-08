import os
import sys
import joblib
from src.exception import CustomException
import logging


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