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
    # Save the trained model weights to a file
    np.save('model_weights/linear_regression_weights.npy', weights)

    # Predict the values for each month in 2022
    months_2022 = np.arange(len(dataframe) + 1, len(dataframe) + 13).reshape(-1, 1)
    predictions_2022 = np.c_[np.ones(len(months_2022)), months_2022] @ weights

    return weights, predictions_2022

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

class NN:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def initialize_weights(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        weights_input_hidden = np.random.randn(input_size, hidden_size)
        biases_hidden = np.zeros((1, hidden_size))
        weights_hidden_output = np.random.randn(hidden_size, output_size)
        biases_output = np.zeros((1, output_size))
        return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

    def forward_pass(self, X, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output):
        hidden_layer_input = np.dot(X, weights_input_hidden) + biases_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_output
        predicted_output = self.sigmoid(output_layer_input)
        return hidden_layer_output, predicted_output

    def backward_pass(self, X, y, hidden_layer_output, predicted_output, weights_hidden_output):
        output_error = y - predicted_output
        output_delta = output_error * self.sigmoid_derivative(predicted_output)

        hidden_layer_error = output_delta.dot(weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(hidden_layer_output)

        return output_delta, hidden_layer_delta

    def update_weights(self, X, hidden_layer_output, output_delta, hidden_layer_delta,
                    weights_input_hidden, biases_hidden, weights_hidden_output, biases_output,
                    learning_rate):
        weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
        biases_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        weights_input_hidden += X.T.dot(hidden_layer_delta) * learning_rate
        biases_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

        return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def train_and_predict(self, X_train, y_train, X_predict, hidden_size, epochs, learning_rate):
        input_size = X_train.shape[1]
        output_size = y_train.shape[1]

        weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = self.initialize_weights(input_size, hidden_size, output_size)

        # Training
        for epoch in range(epochs):
            # Forward pass
            hidden_layer_output, predicted_output = self.forward_pass(X_train, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output)

            # Backward pass
            output_delta, hidden_layer_delta = self.backward_pass(X_train, y_train, hidden_layer_output, predicted_output, weights_hidden_output)

            # Update weights
            weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = self.update_weights(X_train, hidden_layer_output, output_delta, hidden_layer_delta,
                                                                                                    weights_input_hidden, biases_hidden, weights_hidden_output, biases_output,
                                                                                                    learning_rate)

            # Print MSE for every 100 epochs
            if epoch % 100 == 0:
                mse = self.mean_squared_error(y_train, predicted_output)
                print(f"Epoch {epoch}, Mean Squared Error: {mse}")

        # Save trained weights
        np.savez('trained_weights.npz', weights_input_hidden=weights_input_hidden,
                biases_hidden=biases_hidden, weights_hidden_output=weights_hidden_output, biases_output=biases_output)

        # Prediction
        hidden_layer_output, predicted_output = self.forward_pass(X_predict, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output)
        return predicted_output

monthly_data = gen_df('data/data_daily.csv')
print(monthly_data)

# # Linear Regression (Base Line) Model 
weights, predictions_2022 = linear_reg(monthly_data)
plot_preds(predictions_2022,'linear_regression')
plt.close()

def nn_prediction(df):
    # Extract features (X) and labels (y)
    X_train = df[['Month']].values
    y_train = df[['Receipt_Count']].values

    # Months for prediction in 2022
    X_predict_2022 = np.arange(1, 13).reshape(-1, 1)  # Assuming you want to predict for months 1 to 12 of 2022

    # Normalize the data
    X_train_normalized = X_train / np.max(X_train)
    y_train_normalized = y_train / np.max(y_train)
    X_predict_normalized = X_predict_2022 / np.max(X_train)  # Normalize based on training data

    nn = NN()
    # Train and predict
    nn_predictions_2022 = nn.train_and_predict(X_train_normalized, y_train_normalized, X_predict_normalized, hidden_size=4, epochs=1000, learning_rate=0.1)

    # Print predictions for each month in 2022
    for month, prediction in zip(X_predict_2022.flatten(), nn_predictions_2022.flatten()):
        print(f"Predicted Receipt Count for Month {month} in 2022: {prediction * np.max(y_train)}")
    nn_pred = (nn_predictions_2022* np.max(y_train))
    return nn_pred

nn_pred = nn_prediction(monthly_data)
plot_preds(nn_pred,'neural_network')
plt.close()