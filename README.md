# store-sales-forecasting
Building time series (SARIMA) models using `auto_arima()` to forecast store sales

### Dataset
  - The dataset is taken from kaggle competition on "Walmart Recruiting - Store Sales Forecasting" in 2014.
  - https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/overview

### Model (first iteration)
  - built SARIMA models using `auto_arima()`
  - one model for each store-department combination 
  - when there is no train data but the prediction is requested, just predicts 0
  - missing values are filled with `interpolate()`
  - when `auto_arima()` fails, predicts 0
  - used only sales data (time series)
    - additional data like `isHoliday`, store location, etc. is available but NOT USED in this iteration

### Result
  - the evaluation metric score is the weighted mean absolute error (WMAE).
  - my result is submitted to kaggle and the score is compared to those on the ranking at the time the competition ended
  - my score is 2979 which corresponds to the 122nd place in the compettition 
  - the winner's score is 2301 and the base line (all zeros benchmark) score is 22265
   
### The code to run auto_arima()
    python walmart_kaggle.py

### Notebook 
  - concat_results.ipynb
    - to gather all results and does a final touch to generate the submission file

### Outlook
  - there are cases when the current treatment is just predicting 0, this can be improved by assigning more reasonable values, e.g. look at the same department in the different stores and find the average value.
  - the model can be improved by incorporating additional information provided
