import pickle
from flask import Flask,request,render_template
from jinja2.utils import pass_context
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

#route for a home page

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Handle POST request - for now return a placeholder response
        data = CustomData(
            Gender=request.form.get('Gender', ''),
            Married=request.form.get('Married', ''),
            Dependents=request.form.get('Dependents', ''),
            Education=request.form.get('Education', ''),
            Self_Employed=request.form.get('Self_Employed', ''),
            ApplicantIncome=int(request.form.get('ApplicantIncome', 0)),
            CoapplicantIncome=int(request.form.get('CoapplicantIncome', 0)),
            LoanAmount=int(request.form.get('LoanAmount', 0)),
            Loan_Amount_Term=int(request.form.get('Loan_Amount_Term', 0)),
            Credit_History=int(request.form.get('Credit_History', 0)),
            Property_Area=request.form.get('Property_Area', '')
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        print(f"Prediction result: {results[0]}")
        return render_template('home.html',results=results[0])

if __name__ == "__main__":
    application.run(host="0.0.0.0",debug=True)