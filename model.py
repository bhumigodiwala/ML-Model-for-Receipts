# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gen_df(filename):
    '''
    Returns the dataframe from the data.csv file after Data Pre-processing

    filename: path to .csv file
    '''

    # Load data from a CSV file and perform data preprocessing
    df = pd.read_csv(filename)
    mapping = {df.columns[0]:'Date', df.columns[1]: 'Receipt_Count'}
    df = df.rename(columns=mapping)
    df.Date = pd.to_datetime(df.Date)
    # print(df)

    # Resample the data to monthly frequency
    monthly_data = df.resample('M', on='Date').sum()

    # Add a new column 'Month' to use as a feature
    monthly_data['Month'] = range(1, len(monthly_data) + 1)
    # print(monthly_data)
    return monthly_data

def linear_reg(dataframe):
    # Linear regression model
    X = dataframe[['Month']].values
    y = dataframe['Receipt_Count'].values

    # Use the normal equation to compute the weights
    X_bias = np.c_[np.ones(X.shape[0]), X]
    weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

    # Predict the values for each month in 2022
    months_2022 = np.arange(len(dataframe) + 1, len(dataframe) + 13).reshape(-1, 1)
    predictions_2022 = np.c_[np.ones(len(months_2022)), months_2022] @ weights

    return predictions_2022

def plot_preds(predictions,model_name):
    # Visualize the results
    plt.plot(monthly_data.index, monthly_data['Receipt_Count'], label='Observed')
    plt.plot(pd.date_range(start='2022-01-01', periods=12, freq='M'), predictions, label='Predicted', linestyle='--', marker = '.')
    plt.title('Monthly Scanned Receipts Prediction for 2022')
    plt.xlabel('Date')
    plt.ylabel('Receipt Count')
    plt.legend()
    plt.savefig(('output_plots/{}.jpg').format(model_name))
    print('Output Plots Generated Successfully!')

monthly_data = gen_df('data/data_daily.csv')

# Linear Regression (Base Line) Model 
predictions_2022 = linear_reg(monthly_data)
plot_preds(predictions_2022,'linear_regression')

