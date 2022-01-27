import numpy as np
import pandas as pd

from Backend.Data.DataManager.matrix_data import prepare_all_beta_predictors
from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline, forecast_seirv


def seirv_ml_layer(y_train_diffeq, start_vals_seirv, fixed_model_params_seirv, forecasting_horizon,
                   ml_matrix_predictors, standardizer_obj, ml_model):
    ## 1) Run Pipeline for training period:
    training_pipeline_results = seirv_pipeline(y_train=y_train_diffeq,
                                               start_vals_fixed=start_vals_seirv,
                                               fixed_model_params=fixed_model_params_seirv,
                                               allow_randomness_fixed_beta=False)

    ## 2) Prepare results for running again:
    # Get start values for rerunning:
    start_vals = {
        'S': training_pipeline_results['model_start_vals_forecast_period']['S'],
        'E0': training_pipeline_results['model_start_vals_forecast_period']['E'],
        'I0': training_pipeline_results['model_start_vals_forecast_period']['I'],
        'U0': training_pipeline_results['model_start_vals_forecast_period']['U'],
        'R0': training_pipeline_results['model_start_vals_forecast_period']['R'],
        'V0': training_pipeline_results['model_start_vals_forecast_period']['V'],
        'V_cum': 0,
    }

    # Create tuple with starting values:
    start_vals_tuple = tuple(start_vals.values())
    model_params_tuple = tuple(training_pipeline_results['model_params_forecast_period'].values())

    ## 3) Machine learning beta computation:
    # Prepare predictors for machine learning model:
    ml_matrix_predictors_all = \
        prepare_all_beta_predictors(ml_predictors=ml_matrix_predictors,
                                    y_train_last_two_weeks=y_train_diffeq,
                                    previous_beta=training_pipeline_results['model_params_forecast_period']['beta'])

    # Standardize predictors:
    ml_matrix_predictors_all_standardized = standardizer_obj.transform(ml_matrix_predictors_all)

    # Apply Machine Learning Model to obtain beta:
    beta_pred = ml_model.predict(ml_matrix_predictors_all)[0]

    ## 4) Forecast with obtained beta guess:
    # Rerun model with new beta guess:
    y_pred_point_estimate = forecast_seirv(all_model_params=model_params_tuple,
                                           y0=start_vals_tuple,
                                           forecast_horizon=forecasting_horizon)
