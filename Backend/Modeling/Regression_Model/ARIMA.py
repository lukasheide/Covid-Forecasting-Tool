import pmdarima as pmd
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
    #print(final_model.summary())
    print(compute_evaluation_metrics(y_val, predictions))
    model_results = {'season': j, 'model': final_model}
    return model_results

def sarima_model_predictions(y_train, m, length):
    arima_model = sarimamodel(y_train, m)
    return arima_model

