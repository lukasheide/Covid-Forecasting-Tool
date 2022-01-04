import pandas as pd
import numpy as np
import pmdarima as pmd
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
from Backend.Visualization.modeling_results import plot_train_fitted_and_validation
import matplotlib.pyplot as plt

#reading csv file including date, cases, class
df = pd.read_csv("data_long_older.csv")
print(df.shape)
print(df.info())

#creating an array with the dates
date = df["Date"]
print(date.shape)

#creating test and training data set
train = df[df["Class"] == "train"]
test = df[df["Class"] == "test"]
print(train.shape)
print(test.shape)

#create training array
train_array = train["Cases"]
print(train_array.shape)

#create test array
test_array = test["Cases"]
print(test_array.shape)

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

#creating ARIMA model
def arimamodel(timeseriesarray):
    autoarima_model = pmd.auto_arima(timeseriesarray,
                              start_p=1,
                              start_q=1,
                                     max_p=3,
                                     max_q=3,
                              test="adf",
                             trace=True)

    return autoarima_model

#evaluating arima model
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#evaluating arima model
def root_mean_square_error(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

#create plot for visualization
def plot_model(train_array, test_array, predictions: int):
    len_train = len(train_array)
    len_test = len(test_array)
    len_total = len_train + len_test
    t_grid_total = np.linspace(1, len_total, len_total)

    pred_array = np.concatenate((train_array, predictions))
    val_array = np.concatenate((train_array, test_array))
    plt.plot(t_grid_total, pred_array, 'b--')
    plt.plot(t_grid_total, val_array, 'g')
    print(pred_array)
    plt.show()

def run_model():
    m = 2000
    i = 1

    #iteration to find the best fitting length of a season
    while i < 12:
        arima_model = sarimamodel(train_array, i)
        predictions = arima_model.predict(len(test))
        MAPE = mean_absolute_percentage_error(test.Cases, predictions)
        RMSE = root_mean_square_error(test.Cases, predictions)
        print(MAPE)
        print(RMSE)

        #save best fitting model
        if MAPE < m:
            m = mean_absolute_percentage_error(test.Cases, predictions)
            r = root_mean_square_error(test.Cases, predictions)
            final_model = arima_model
        i +=1

    print(final_model.summary())
    print(final_model.predict(len(test)))
    print('MAPE = ', m)
    print('RMSE = ', r)
    plot_model(train_array, test_array, final_model.predict(len(test)))

model = run_model()
