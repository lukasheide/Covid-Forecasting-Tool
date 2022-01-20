import pandas as pd
from datetime import date

from Backend.Data.DataManager.data_access_methods import get_smoothen_cases, get_starting_values, get_model_params
from Backend.Data.DataManager.data_util import Column, date_int_str, compute_end_date_of_validation_period
from Backend.Data.DataManager.db_calls import start_pipeline, insert_param_and_start_vals, insert_prediction_vals, \
    get_all_table_data
from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Modeling.Util.pipeline_util import train_test_split, get_list_of_random_dates, get_list_of_random_districts
from Backend.Visualization.modeling_results import plot_train_fitted_and_validation, plot_sarima_pred_plot, \
    plot_sarima_val_line_plot, plot_train_fitted_and_predictions
from Backend.Modeling.Regression_Model.ARIMA import run_sarima, sarima_model_predictions


def forecasting_pipeline():

    ### Pipeline Configuration:
    forecast_start_date = '2022-01-16'
    forecasting_horizon = 14

    train_length_diffeqmodel = 14
    train_length_sarima = 28
    training_period_max= max(train_length_diffeqmodel,train_length_sarima)

    opendata = get_all_table_data(table_name='district_list')
    districts = opendata['district'].tolist()


    # 1) Compute pipeline parameters:
    train_start_date_SEIRV = # todo
    train_start_date_SARIMA = # todo

    # Iterate over all districts:
    results_dict = {}
    for i, district in enumerate(districts):

        ### 2) Import Training Data

        ## 2a) Import Historical Infections

        y_train = 'myNumpyArray' # get training data for last four weeks

        ## 2b) Get Starting Values for SEIRV Model:
        start_vals = get_starting_values(district, train_start_date_SEIRV)
        fixed_model_params = get_model_params(district, train_start_date_SEIRV)

        ## 2c) Import Data for Machine Learning Matrix:

        # this is used for the machine learning layer later on

        ## 





if __name__ == '__main__':
    forecasting_pipeline()