from Backend.Data.data_pipeline import run_data_pipeline

from Backend.Modeling.forecast_pipeline import forecasting_pipeline
from Backend.Modeling.model_validation_pipeline import model_validation_pipeline_v2_wrapper
from Frontend.AnalyticsTool.forecast_dashboard import start_dashboard


def run_quick_forecast():
    run_data_pipeline()
    forecasting_pipeline()
    start_dashboard()


def run_forecast_with_eval():
    run_data_pipeline()
    # run Evaluation Pipeline? --> evaluation_pipeline.R?
    forecasting_pipeline()


def run_model_validation():
    run_data_pipeline()
    model_validation_pipeline_v2_wrapper()


if __name__ == '__main__':
    """
        Assigning different tasks name to the task variable will allow you to run different versions of forecasts
    
        task --> 'generate_forecasts':
        
            this will only collect data from the scratch, by downloading the latest existing prediction_intervals.csv 
            from our remote server and run the forecast then start the dashboard app in your web browser after forecasts
            are generated.
            
            NOTE: to generate this prediction_intervals.csv by yourself, you need to execute the 
                  evaluation_pipeline.R script manually? TODO --> Lukas?
        
        task --> 'run_model_validation':
        
            this will execute a full validation run for all the models used in this project and 
            store validation results in corresponding .csv files
        
    """

    task = 'generate_forecasts'

    if task == 'generate_forecasts':
        run_quick_forecast()
    elif task == 'run_model_validation':
        run_model_validation()
