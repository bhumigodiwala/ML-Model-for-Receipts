from flask import Flask, render_template, request
from urllib.parse import quote
import os

import pandas as pd
import numpy as np
from model import gen_df, linear_reg

app = Flask(__name__)

# Load the data (replace this with your actual data)
monthly_data = gen_df('data/data_daily.csv')

# Linear regression model
weights, predictions_2022 = linear_reg(monthly_data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    month = int(request.form['month'])
    prediction = predictions_2022
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
