import joblib
import pandas as pd
from datetime import date

from Backend.Data.DataManager.data_access_methods import get_smoothen_cases, get_starting_values, get_model_params
from Backend.Data.DataManager.data_util import Column, date_int_str, compute_end_date_of_validation_period
from Backend.Data.DataManager.db_calls import start_pipeline, insert_param_and_start_vals, insert_prediction_vals, \
    get_all_table_data
from Backend.Data.DataManager.matrix_data import get_predictors_for_ml_layer
from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline
from Backend.Modeling.Differential_Equation_Modeling.seirv_model_and_ml import seirv_ml_layer
from Backend.Modeling.Regression_Model.ARIMA import sarima_pipeline_pred, sarima_pipeline_val
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Modeling.Util.pipeline_util import train_test_split, get_list_of_random_dates, get_list_of_random_districts
from Backend.Modeling.forecast import forecast_all_models, convert_all_forecasts_to_incidences
from Backend.Modeling.model_validation import sarima_pipeline
from Backend.Visualization.modeling_results import plot_train_fitted_and_validation, plot_sarima_pred_plot, \
    plot_sarima_val_line_plot, plot_train_fitted_and_predictions
from Backend.Modeling.Regression_Model.ARIMA import run_sarima, sarima_model_predictions

import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def forecasting_pipeline():

    #################### Pipeline Configuration: ####################
    training_end_date = '2022-01-16'
    forecasting_horizon = 14

    train_length_diffeqmodel = 14
    train_length_sarima = 28
    training_period_max = max(train_length_diffeqmodel, train_length_sarima)

    # ML Layer:
    ml_model_path = '../Assets/MachineLearningLayer/Models/xgb_model_lukas.pkl'
    standardizer_model_path = '../Assets/MachineLearningLayer/Models/standardizer_model.pkl'

    # Ensemble Weights:
    ensemble_model_share = {
        'seirv_last_beta': 0.5,
        'seirv_ml_beta': 0,
        'sarima': 0.5
    }

    debug = False

    ################################################################

    # 1) Compute pipeline parameters:
    opendata = get_all_table_data(table_name='district_list')
    districts = opendata['district'].tolist()

    # Import ML-Model:
    with open(ml_model_path, 'rb') as fid:
        ml_model = joblib.load(fid)

    # Import Standardizer:
    with open(standardizer_model_path, 'rb') as fid:
        standardizer_obj = joblib.load(fid)


    # Iterate over all districts:
    results_dict = {}
    for i, district in enumerate(districts):
        print(f'Computing district {district}: {i+1} / {len(districts)}')

        ### 2) Import Training Data
        ## 2a) Import Historical Infections
        # Import Training Data:
        y_train_df = get_smoothen_cases(district, training_end_date, train_length_sarima)
        y_train_seirv_df = y_train_df[-train_length_diffeqmodel:].reset_index(drop=True)
        y_train_sarima_df = y_train_df

        # Get starting dates as strings:
        train_start_date_SEIRV = date_int_str(y_train_seirv_df[Column.DATE][0])
        train_start_date_SARIMA = date_int_str(y_train_sarima_df[Column.DATE][0])

        # Get arrays from training dataframe:
        y_train_seirv = y_train_seirv_df[Column.SEVEN_DAY_SMOOTHEN]
        y_train_sarima = y_train_sarima_df[Column.SEVEN_DAY_SMOOTHEN]

        ## 2b) Get Starting Values for SEIRV Model:
        start_vals_seirv = get_starting_values(district, train_start_date_SEIRV)
        fixed_model_params_seirv = get_model_params(district, train_start_date_SEIRV)

        ## 2c) Import Data for Machine Learning Matrix:
        ml_training_data = get_predictors_for_ml_layer(district, training_end_date)

        # this is used for the machine learning layer later on


        ### 3) Models
        # Run all four models:
        seirv_last_beta_only_results, seirv_ml_results, sarima_results, ensemble_results, all_combined_seven_day_average = \
            forecast_all_models(y_train_seirv, y_train_sarima, forecasting_horizon,
                                ml_training_data, start_vals_seirv, fixed_model_params_seirv,
                                standardizer_obj, ml_model, district, ensemble_model_share)

        ## 4) Debugging Visualization
        if debug:
            pass #todo

        ## 5) Convert 7-day average to 7-day-incident:
        all_combined_incidence = convert_all_forecasts_to_incidences(all_combined_seven_day_average, start_vals_seirv['N'])

        ## 6) Upload to DB

        # 6.1) If exists, delete existing data in table for current district

        # 6.2) Upload data for current district to table:
        # With the following Columns:
        # [1] date
        # [2] historical_infections (training data -> is NA in forecasting period)
        # [3-5] SEIRV-Model + Last Beta
        ## [3] Seirv_last_beta_MEAN_PREDICTION
        ## [4] Seirv_last_beta_UPPER_PREDICTION
        ## [5] Seirv_last_beta_LOWER_PREDICTION
        # [6-8] SEIRV-Model + Machine Learning layer
        ## [6] Seirv_ml_beta_MEAN_PREDICTION
        ## [7] Seirv_ml_beta_UPPER_PREDICTION
        ## [8] Seirv_ml_beta_LOWER_PREDICTION
        # [9-11] SArima
        ## ...
        # [12-14] Ensemble Model
        ## ...

        pass
        # Metadata - Table:
        # [1] Pipeline-ID
        # [2] Full_Run                          -> set to 1 if pipeline is run on all districts and 0 if only on a subset
        # [3] Timestamp of Start of Pipeline
        # [4] Training-Start-Day
        # [5] Training-End-Day
        # [6] Forecasting-Start-Day
        # [7] Forecasting-End-Day





if __name__ == '__main__':
    forecasting_pipeline()