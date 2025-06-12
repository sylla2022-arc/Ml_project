from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import load_object
from src.Pipeline.predict_pipeline import PredictPipeline, CustomData
from src.logger import logging


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html') 
    
    else :
        data = CustomData(
            gender=str(request.form.get('gender')),
            race_ethnicity=str(request.form.get('race_ethnicity')),
            parental_level_of_education=str(request.form.get('parental_level_of_education')),
            lunch=str(request.form.get('lunch')),
            test_preparation_course=str(request.form.get('test_preparation_course')),
            writing_score=float(request.form.get('writing_score')), # type: ignore
            reading_score=float(request.form.get('reading_score')) # type: ignore
        )
    
        preds_data = data.get_data_as_data_frame()
        print(f"Dataframe for prediction: {preds_data}")
        
        logging.info(f"Dataframe for prediction: {preds_data}")

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(features=preds_data)

        return render_template('home.html', contex_result= np.round(results[0], 2))
    
if __name__ == "__main__":
    print("App running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
    