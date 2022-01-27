import pandas as pd
import pmdarima as pmd
import numpy as np
from Backend.Evaluation.metrics import compute_evaluation_metrics

#creating SARIMA model
def sarimamodel(timeseriesarray, i):
        autoarima_model = pmd.auto_arima(timeseriesarray,
                                         start_p=1,
                                         start_q=1,
                                         max_p=3, max_q=3, m=i,
                                         start_P=1, max_P=1, seasonal=True, d=1,
                                         D=1, trace=False,
                                         error_action='ignore',
                                         suppress_warnings=True,
                                         stepwise=True, test = "adf")
        return autoarima_model

#model execution
def run_sarima(y_train, y_val):
    r = 2000000000
    i = 1

    #iteration to find the best fitting length of a season
    while i < 8:
        arima_model = sarimamodel(y_train, i)
        predictions = arima_model.predict(len(y_val))
        evaluations = compute_evaluation_metrics(y_val, predictions)
        RMSE = evaluations["rmse"]
        #save best fitting model
        if RMSE < r:
            r = RMSE
            final_model = arima_model
            j = i
        i +=1

    predictions = final_model.predict(len(y_val))
    # print(final_model.summary())
    # print(compute_evaluation_metrics(y_val, predictions))
    model_results = {'season': j, 'model': final_model}
    return model_results

def sarima_model_predictions(y_train, m, length):
    arima_model = sarimamodel(y_train, m)
    return arima_model

# create validation data for forecasting pipeline
def create_val_data(y_train, forecasting_horizon):
    length_train = len(y_train)-forecasting_horizon
    y_val_train, y_val_predict = np.split(y_train, [length_train])
    return y_val_train, y_val_predict

# sarima pipeline for predictions
def sarima_pipeline_pred(y_train, forecasting_horizon):
    y_val_train, y_val_predict = create_val_data(y_train, forecasting_horizon)
    sarima_model = run_sarima(y_train=y_train, y_val=y_val_predict)
    not_used, y_pred_train = np.split(y_train, [14])
    pred_sarima = sarimamodel(y_pred_train, sarima_model["season"])
    predictions, conf_int = pred_sarima.predict(forecasting_horizon, return_conf_int=True, alpha=0.1)
    pred_int = pd.DataFrame(conf_int, columns=['lower', 'upper'])
    results_dict = {
        'predictions': predictions,
        'lower': pred_int['lower'],
        'upper': pred_int['upper'],
        'season': sarima_model["season"]
    }
    return results_dict


# sarima pipeline for validation
def sarima_pipeline_val(y_train, forecasting_horizon):
    y_val_train, y_val_predict = create_val_data(y_train, forecasting_horizon)
    sarima_model = run_sarima(y_train=y_train, y_val=y_val_predict)
    predictions, conf_int = sarima_model["model"].predict(forecasting_horizon, return_conf_int=True, alpha=0.1)
    pred_int = pd.DataFrame(conf_int, columns=['lower', 'upper'])
    results_dict = {
        'predictions': predictions,
        'lower': pred_int['lower'],
        'upper': pred_int['upper'],
        'season': sarima_model["season"]
    }
    return results_dict