from flask import Flask, render_template, request
import numpy as np
from model import gen_df, linear_reg, nn_prediction

app = Flask(__name__)

monthly_data = gen_df('data/data_daily.csv')

# Trained model weights (replace with your actual trained weights)
weights = np.load('trained_weights.npz')
nn_pred = nn_prediction(monthly_data)
print(nn_pred.T[0])

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    month = int(request.form['month'])
    print("Input Month:", month)
    
    # Ensure the input month is within the expected range
    if 1 <= month <= 12:
        # Calculate prediction
        prediction = nn_pred.T
        print("Prediction:", prediction[0])

        return render_template('index.html', prediction=prediction[0][month-1])
    else:
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
