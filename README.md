# ML-Model-for-Receipts
Predicting the approximate number of the scanned receipts for each month for upcoming year

To Create and Update the virtual environment RUN:

```
conda create -n mlreceipts
conda activate mlreceipts
conda env export > mlreceipts
```

Directory Structure:

- Dockerfile
- README.md
- app.py
- data
   |-- data_daily.csv
- mlreceipts
- model.py
- model_weights
   |-- linear_regression_weights.npy
- output_plots
   |-- linear_regression.jpg
   |-- neural_network.jpg
- requirements.txt
- templates
   |-- index.html
- trained_weights.npz
