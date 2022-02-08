import pandas as pd
import pmdarima as pmd

# Fitting ARIMA model
def arimamodel(timeseriesarray):
        # Find number of differencing required to make the time series stationary using ndiffs() function
        d = pmd.arima.ndiffs(timeseriesarray)

        # Fit ARIMA model with given training data
        autoarima_model = pmd.auto_arima(y=timeseriesarray,
                                         start_p=3,
                                         start_q=3,
                                         max_p=4, max_q=4, d=d,
                                         trace=False,
                                         error_action='ignore',
                                         suppress_warnings=True,
                                         stepwise=False, test = "adf",
                                         with_intercept=False, method='nm')
        return autoarima_model


# arima pipeline for predictions and validation
def sarima_pipeline(y_train, forecasting_horizon):

    # 1) Fit the ARIMA model with training data
    pred_arima = arimamodel(y_train)

    # 2) Create the forecasts, including prediction intervals
    predictions, conf_int = pred_arima.predict(forecasting_horizon, return_conf_int=True, alpha=0.1)

    # 3) Create DataFrames for upper and lower prediction intervals
    pred_int = pd.DataFrame(conf_int, columns=['lower', 'upper'])
    lower = pred_int['lower']
    upper = pred_int['upper']

    # 4a) Catch predictions <0
    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 0

    # 4b) Catch lower bound of intervals <0
    for i in range(len(lower)):
        if lower[i] < 0:
            lower[i] = 0

    # 4c) Catch upper bound of intervals <0
    for i in range(len(upper)):
        if upper[i] < 0:
            upper[i] = 0

    # Return prediction results as dictionary
    results_dict = {
        'predictions': predictions,
        'lower': lower,
        'upper': upper
    }
    return results_dict