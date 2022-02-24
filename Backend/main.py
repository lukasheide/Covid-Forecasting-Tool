from Backend.Data.DataManager.data_pipeline import run_data_pipeline

from Backend.Modeling.forecast_pipeline import forecasting_pipeline
from Frontend.AnalyticsTool.forecast_dashboard import start_dashboard


def run_quick_forecast():
    run_data_pipeline()
    forecasting_pipeline(full_run=True)
    start_dashboard()


def run_forecast_with_eval():
    pass


if __name__ == '__main__':
    """
        Assigning different tasks name to the task variable will allow you to run different versions of forecasts
    
        task --> 'run_quick_forecast':
        
        this will only collect data from the scratch, SKIPS 'Evaluation Pipeline' 
        by directly downloading the latest existing prediction_intervals.csv from our remote server and run the forecast
        then start the dashboard app in your web browser.
        
        task --> 'run_forecast_with_eval':
        
        this will collect data from the scratch, RUNS 'Evaluation Pipeline' 
        generates prediction_intervals.csv and run the forecast
        then start the dashboard app in your web browser.
    """

    task = 'run_quick_forecast'

    if task == 'run_quick_forecast':
        run_quick_forecast()
    elif task == 'run_forecast_with_eval':
        run_forecast_with_eval()
