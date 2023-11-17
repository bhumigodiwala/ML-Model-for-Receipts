# ML-Model-for-Receipts

The problem statement provides data for the count of receipts for every month in the year 2021. Based on this knowledge we design a machine learning model that predicts the approximate number of the scanned receipts for each month for year 2022. The solution involves training a machine learning model from scratch, saving the weights of the trained model and based on that making predictions. The implementation is then deployed in form of a web application with the help of DOCKER.

### Home Page of Web Application:

<img src="https://github.com/bhumigodiwala/ML-Model-for-Receipts/blob/main/app_images/home_page.png" width="800">

The web application returns the approximate count of scanned receipts in 2022 as the user enters the number of the month (1-12) as shown below:

### Receipt Count Prediction for April (4) 2022:

<img src="https://github.com/bhumigodiwala/ML-Model-for-Receipts/blob/main/app_images/April_pred.png" width="800">

### Libraries and Packages to Install:
- Python 3.9.6
- Docker 20.10.17
- Flask==2.2.5
- Werkzeug==2.2.3

## Data

The dataset can be found [here]("https://github.com/bhumigodiwala/ML-Model-for-Receipts/blob/main/data/data_daily.csv") in the `data/` directory.

## Setup
To Create and Update the virtual environment RUN:

```
conda create -n mlreceipts
conda activate mlreceipts
conda env export > mlreceipts
```

### Directory Structure:
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
    |-- output.txt
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

## Usage

To RUN the ML model and get predictions locally run the model.py file 

Firstly ensure that you are in the current virtual environment by activating it using 

```
conda activate mlreceipts
```
Once you are in the virtual environment run the file using the below command
```
python3 model.py
```

The above command runs the designed machine learning model and the outputs are stored in 'model_output.txt' file and we can visualize it through plots stored in the `'output_plots/'` folder.

## Docker Setup

To RUN the web-application using Docker, firstly ensure Docker is installed in your system.

Firstly build the application using COMMAND:

```
docker build -t receipts-prediction-app . 
```

To check and RUN the docker:

```
docker ps
docker run -p 8080:5000 receipts-prediction-app
```

Once the docker is up and running open other terminal, activate the virtual environment and run the model.py file.
In another terminal:
```
conda activate mlreceipts
python3 app.py
```
Running the above command the outputs are also stored in 'app_output.txt' and 'model_output.txt' files.
The web application is now accessible on 1270.0.0.1:5000. The user can enter the number of the month ranging from 1-12 to get the approximate number of receipt counts for that month in the year 2022. 

## Contact
In case of any issues or queries, please feel free to create an issue or send an email to godiwala.bhumi@gmail.com