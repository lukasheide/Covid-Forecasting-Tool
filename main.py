from Backend.Data.data_pipeline import run_data_pipeline

from Backend.Modeling.forecast_pipeline import forecasting_pipeline
from Backend.Modeling.model_validation_pipeline import model_validation_pipeline_v2_wrapper
from Frontend.AnalyticsTool.forecast_dashboard import start_dashboard


# description can be found below
def run_quick_forecast():
    run_data_pipeline()
    forecasting_pipeline(full_run=True, debug=False, forecast_from=None, with_clean=False)
    """

    :param full_run: generates forecasts for all the districts
                    or districts in the variable 'manual_districts' will be used
    :param debug: execute additional step to visualize each district forecast
    :param forecast_from: can specify a date you need to forecast from. if None latest possible date will be used
    :param with_clean: used to drop the forecast store tables if exists or will only be created if not exist
    :return:
    """
    start_dashboard()


# description can be found below
def run_model_validation():
    run_data_pipeline()
    model_validation_pipeline_v2_wrapper()


if __name__ == '__main__':
    """
        Assigning different tasks name to the task variable will allow you to run different process of our tool
    
        # task --> 'generate_forecasts':
        
            This will only collect basic data from the scratch, Apart from the basic data inputs, 
            this pipeline requires an additional file input called prediction_intervals.csv. 
            These intervals are used for computing the prediction intervals for differential equation models for
            forecasting. 'generate_forecasts' will automatically download the latest existing prediction_intervals.csv 
            from our remote server during the pipeline run. After forecasts generated, 
            the dashboard app will automatically start in your web browser (this may take about approx.20-30sec)
            
            NOTE: to generate this prediction_intervals.csv by yourself with different configurations, 
                  you need to execute the evaluation_pipeline.R script manually(Backend/Evaluation/evaluation_pipeline.R). 
                  Where you can find further explanations to configure and execute the script.
        
        # task --> 'run_model_validation':
        
            This pipeline helps to execute a full validation run for all the models used in this project under flexible 
            different configurations. For instance, multiple intervals over which the pipeline is supposed to run can be 
            setup. For each interval the model validation pipeline will be called (includes all the models). 
            This is usually only done once, unless one wants to run two unconnected time intervals. 
            (Example: Run validation for Apr 2020 - Oct 2020 + Jun 2021 - Jan 2022) 
            
            After the configured pipeline run(s) are completed, it provides an output which contain all the forecast 
            results along with corresponding evaluation results which will be later. 
            The model_validation_data_metrics_(datetime).csv contains detailed information regarding the forecasts 
            of the different models and the correct data (validation data). Using the idx column this table is connected 
            to the model_validation_data_forecasts_(datetime).csv that evaluates the different approaches over each 
            forecasting horizon and district combination.
        
    """

    task = 'generate_forecasts'

    if task == 'generate_forecasts':
        run_quick_forecast()
    elif task == 'run_model_validation':
        run_model_validation()
