import joblib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from datetime import date

from Backend.Data.DataManager.data_access_methods import get_smoothen_cases, get_starting_values, get_model_params
from Backend.Data.DataManager.data_util import Column, date_int_str, compute_end_date_of_validation_period, \
    create_dates_array, get_forecasting_df_columns, print_progress_with_computation_time_estimate
from Backend.Data.DataManager.db_calls import start_pipeline, insert_param_and_start_vals, insert_prediction_vals, \
    get_all_table_data, start_forecast_pipeline, update_db, end_forecast_pipeline
from Backend.Data.DataManager.matrix_data import get_predictors_for_ml_layer
from Backend.Modeling.Differential_Equation_Modeling.prediction_intervals import get_prediction_intervals
from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seiurv_pipeline
from Backend.Modeling.Differential_Equation_Modeling.seirv_model_and_ml import seirv_ml_layer
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Modeling.Util.pipeline_util import train_test_split, get_list_of_random_dates, get_list_of_random_districts
from Backend.Modeling.forecasting_wrapper_functions import forecast_all_models, convert_all_forecasts_to_incidences, \
    convert_seven_day_averages
from Backend.Modeling.model_validation import sarima_pipeline
from Backend.Visualization.plotting import plot_train_fitted_and_validation, plot_sarima_pred_plot, \
    plot_sarima_val_line_plot, plot_train_fitted_and_predictions, plot_all_forecasts

import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def forecasting_pipeline(full_run=False, debug=False):

    ######################################## Pipeline Configuration: ########################################
    ## Below important configurations of the forecasting pipeline parameters can be done:

    # Last day of training: (usually this should be set to the latest day for which RKI infection data is available)
    training_end_date = '2022-01-30'

    # Number of days to used for forecasting:
    forecasting_horizon = 14

    # Number of days prior to forecasting period that the models are supposed to be trained on:
    train_length_diffeqmodel = 14
    train_length_sarima = 42
    training_period_max = max(train_length_diffeqmodel, train_length_sarima)

    ### ML Layer:
    # For the machine learning layer the XGBoost Model provided by the beta estimation machine learning jupyter notebook
    # (Backend/Modeling/Differential_Equation_Modeling/Machine_Learning_Layer/beta_estimation) has to be imported
    # so that the model can be used for producing forecasts for our ML_Beta DiffEq model. As the model needs the
    # predictors to be standardized the standardizer model also has to be imported:

    # Both Models will only be created by running the above described jupyter notebook.
    ml_model_path = '../Assets/MachineLearningLayer/Models/xgb_model_lukas.pkl'
    standardizer_model_path = '../Assets/MachineLearningLayer/Models/standardizer_model.pkl'

    ### Ensemble Weights:
    # The ensemble model is computed as a weighted average of the other three models. The weights can be set here:
    ensemble_model_share = {
        'seirv_last_beta': 0,
        'seirv_ml_beta': 0.5,
        'sarima': 0.5
    }

    # Compute start/end dates for training/validation baased on the above set training_end_date variable:
    training_start_date = datetime.strptime(training_end_date, '%Y-%m-%d') - timedelta(days=training_period_max)
    forecast_start_date = datetime.strptime(training_end_date, '%Y-%m-%d') + timedelta(days=1)
    forecast_end_date = forecast_start_date + timedelta(days=forecasting_horizon)

    # Store starting values of the pipeline to DB, initialize a new pipeline in the database and get it's ID so that
    # the results of the pipeline can later be stored correctly to that pipeline ID:
    pipeline_id = start_forecast_pipeline(t_start_date=training_start_date.strftime('%Y-%m-%d'),
                                          t_end_date=training_end_date,
                                          f_start_date=forecast_start_date.strftime('%Y-%m-%d'),
                                          f_end_date=forecast_end_date.strftime('%Y-%m-%d'),
                                          full_run=full_run)

    ################################################################

    ### 1) Prepare Pipeline Run:
    ## Districts:
    # Option 1: Districts set manually (used for debugging only)
    manual_districts = ['Münster', 'Potsdam', 'Segeberg', 'Rosenheim, Kreis', 'Hochtaunus', 'Dortmund', 'Essen', 'Bielefeld',
                        'Warendorf', 'München, Landeshauptstadt']

    # Option 2: All districts
    # If full_run flag is set to true (standard case) the lines below retrieves the names of all 400 districts:
    if full_run:
        opendata = get_all_table_data(table_name='district_list')
        districts = opendata['district'].tolist()
        districts.sort()
    else:
        districts = manual_districts

    # Import ML-Model:
    with open(ml_model_path, 'rb') as fid:
        ml_model = joblib.load(fid)

    # Import Standardizer:
    with open(standardizer_model_path, 'rb') as fid:
        standardizer_obj = joblib.load(fid)

    ## Import Prediction Intervals:
    # These intervals are used for computing the prediction intervals for forecasting. Using them we can provide
    # prediction intervals for both differential equation models. The models are computed in the evaluation_pipeline.R
    # script. Here more information can be found regarding how exactly the values are computed.
    pred_intervals_df = get_prediction_intervals()

    # Save start time for printing the estimated time until when the function is finished.
    start_time_pipeline = datetime.now()

    # Iterate over all districts:
    for i, district in enumerate(districts):
        # Print progress and estimate time until pipeline is finished:
        print_progress_with_computation_time_estimate(completed=i + 1, total=len(districts),
                                                      start_time=start_time_pipeline)

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


        ### 3) Models
        # Get the predictions for all 4 models (at least if all flags are set to true):
        seirv_last_beta_only_results, seirv_ml_results, sarima_results, ensemble_results, all_combined_seven_day_average = \
            forecast_all_models(y_train_seirv, y_train_sarima, forecasting_horizon,
                                ml_training_data, start_vals_seirv, fixed_model_params_seirv,
                                standardizer_obj, ml_model, district, ensemble_model_share, pred_intervals_df)


        ## 4) Convert 7-day averages to 7-day-incidences:
        # As our models are trained and executed on 7-day averages instead of the 7-days incidences which we need as
        # our final outputs we have to convert them here:
        all_combined_incidence = convert_all_forecasts_to_incidences(all_combined_seven_day_average, start_vals_seirv['N'])
        y_train_sarima_incidence = convert_seven_day_averages(y_train_sarima, start_vals_seirv['N'])

        ## 5) Visualization (only for debugging)
        if debug:
            plot_all_forecasts(forecast_dictionary=all_combined_incidence, y_train=y_train_sarima_incidence,
                               start_date_str=training_start_date.strftime('%Y-%m-%d'), forecasting_horizon=forecasting_horizon,
                               district=district,
                               plot_diff_eq_last_beta=True,
                               plot_diff_eq_ml_beta=True,
                               plot_sarima=True,
                               plot_ensemble=False,
                               plot_predictions_intervals=False
                               )

        ### 6) Upload to DB
        column_names = get_forecasting_df_columns()
        final_forecast_df = pd.DataFrame(columns=column_names)

        # Create dates array from start_date_training to end_date_forecasting:
        dates_array = create_dates_array(start_date_str=train_start_date_SARIMA, num_days=training_period_max+forecasting_horizon)

        # Add dates:
        final_forecast_df['date'] = dates_array

        # Add cases:
        final_forecast_df['cases'] = pd.Series(y_train_sarima_incidence)

        # Add district name:
        final_forecast_df['district_name'] = district

        # Add district name:
        final_forecast_df['pipeline_id'] = pipeline_id

        # Add forecasts:
        for k, v in all_combined_incidence.items():
            final_forecast_df[k].iloc[-forecasting_horizon:] = v

        update_db(table_name='district_forecast', dataframe=final_forecast_df, replace=False)

    end_forecast_pipeline(pipeline_id)




if __name__ == '__main__':
    forecasting_pipeline(full_run=True)
