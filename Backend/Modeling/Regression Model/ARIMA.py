import pandas as pd
import numpy as np
import pmdarima as pmd
from statsmodels.tsa.stattools import adfuller
from Backend.Visualization.plotting import plot_sarima_model_line_plot
from Backend.Evaluation.metrics import compute_evaluation_metrics

#reading csv file including date, cases, class
df = pd.read_csv("data_smoothed.csv")

#creating an array with the dates
date = df["Date"]

#creating test and training data set
train = df[df["Class"] == "train"]
test = df[df["Class"] == "test"]

#create training array
train_array = train["Cases"]

#create test array
test_array = test["Cases"]

print("p-value:", adfuller(train_array.dropna())[1])

#differencing the training data to create stationarity
diff_1 = train_array.diff().dropna()

#test stationarity with augmented dickey-fuller test
print("p-value:", adfuller(diff_1.dropna())[1])

#creating SARIMA model
def sarimamodel(timeseriesarray, i):
        autoarima_model = pmd.auto_arima(timeseriesarray,
                                         start_p=1,
                                         start_q=1,
                                         max_p=3, max_q=3, m=i,
                                         start_P=1, max_P=1, seasonal=True, d=1,
                                         D=1, trace=True,
                                         error_action='ignore',
                                         suppress_warnings=True,
                                         stepwise=True, test = "adf")
        return autoarima_model

#model execution
def sarima_pipeline(train_array, test_array):
    m = 2000
    i = 1

    #iteration to find the best fitting length of a season
    while i < 15:
        arima_model = sarimamodel(train_array, i)
        predictions = arima_model.predict(len(test_array))
        evaluations = compute_evaluation_metrics(test_array, predictions)
        MAPE = evaluations["mape"]
        RMSE = evaluations["rmse"]
        print(MAPE)
        print(RMSE)

        #save best fitting model
        if MAPE < m:
            m = MAPE
            r = RMSE
            final_model = arima_model
        i +=1

    print(final_model.summary())
    print(final_model.predict(len(test)))
    print('MAPE = ', m)
    print('RMSE = ', r)
    plot_sarima_model_line_plot(train_array, test_array, final_model.predict(len(test)))
    return predictions

model = sarima_pipeline(train_array, test_array)