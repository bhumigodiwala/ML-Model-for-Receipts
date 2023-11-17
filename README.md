# ML-Model-for-Receipts

The problem statement provides data for the count of receipts for every month in the year 2021. Based on this knowledge we design a machine learning model that predicts the approximate number of the scanned receipts for each month for year 2022. The solution involves training a machine learning model from scratch, saving the weights of the trained model and based on that making predictions. The implementation is then deployed in form of a web application with the help of DOCKER.

Home Page of Web Application:

https://github.com/bhumigodiwala/ML-Model-for-Receipts/blob/main/app_images/home_page.png

The web application returns the approximate count of scanned receipts in 2022 as the user enters the number of the month (1-12) as shown below:

Receipt Count Prediction for April (4) 2022:

https://github.com/bhumigodiwala/ML-Model-for-Receipts/blob/main/app_images/April_pred.png

To Create and Update the virtual environment RUN:

```
conda create -n mlreceipts
conda activate mlreceipts
conda env export > mlreceipts
```

Directory Structure:
```bash
ML-Model-For-Receipts
    |-- Dockerfile
    |-- README.md
    |-- app.py
    |-- app_images
        |-- April_pred.png
        |-- home_page.png
    |-- data
        |-- data_daily.csv
    |-- mlreceipts
    |-- model.py
    |-- model_weights
        |-- linear_regression_weights.npy
    |-- output_plots
        |-- linear_regression.jpg
        |-- neural_network.jpg
    |-- requirements.txt
    |-- templates
        |-- index.html
    |-- trained_weights.npz
```
