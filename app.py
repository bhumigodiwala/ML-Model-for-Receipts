from flask import Flask, render_template, request
import numpy as np
from model import gen_df, linear_reg, nn_prediction

app = Flask(__name__)

monthly_data = gen_df('data/data_daily.csv')

# Trained model weights (replace with your actual trained weights)
weights = np.load('trained_weights.npz')
nn_pred = nn_prediction(monthly_data)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    month = int(request.form['month'])
    print("Input Month:", month)

    calender = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August', 
                9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    
    # Ensure the input month is within the expected range
    if 1 <= month <= 12:
        # Calculate prediction
        prediction = nn_pred.T
        print((" Month: {} Prediction: {}").format(month, prediction[0][month-1]))

        return render_template('index.html', prediction=prediction[0][month-1], month=month, calender=calender[month])
    else:
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
