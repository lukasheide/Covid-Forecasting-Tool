import numpy as np
import pandas as pd

from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline
from Backend.Modeling.Differential_Equation_Modeling.seirv_model_and_ml import seirv_ml_layer
from Backend.Modeling.Regression_Model.ARIMA import sarima_pipeline


def forecast_all_models(y_train_diffeq, y_train_sarima, forecasting_horizon, ml_matrix_predictors,
                        start_vals_seirv, fixed_model_params_seirv,
                        standardizer_obj, ml_model, district, ensemble_model_share
                        ):
    ## 3.1) SEIRV + Last Beta
    seirv_last_beta_only_results = seirv_pipeline(y_train=y_train_diffeq,
                                                  start_vals_fixed=start_vals_seirv,
                                                  fixed_model_params=fixed_model_params_seirv,
                                                  forecast_horizon=forecasting_horizon,
                                                  allow_randomness_fixed_beta=False, district=district)

    ## 3.2) SEIRV + Machine Learning Layer
    seirv_ml_results = seirv_ml_layer(y_train_diffeq, start_vals_seirv, fixed_model_params_seirv,
                                      forecasting_horizon, ml_matrix_predictors, standardizer_obj, ml_model)

    ## 3.3) SARIMA
    # input: y_train_sarima (6 weeks), forecast_horizon (14 days)
    # output: {y_pred_mean, y_pred_upper, y_pred_lower, params}
    sarima_results = sarima_pipeline(y_train=y_train_sarima,
                                         forecasting_horizon=forecasting_horizon)

    ## 3.4) Ensemble Model
    # Ensemble Share:
    ensemble_point_y_pred = ensemble_model_share['seirv_last_beta'] * seirv_last_beta_only_results[
        'y_pred_without_train_period'] + \
                            ensemble_model_share['seirv_ml_beta'] * seirv_ml_results['y_pred_mean'] + \
                            ensemble_model_share['sarima'] * sarima_results['predictions']

    ensemble_results = {
        'y_pred_mean': ensemble_point_y_pred
    }

    all_combined = {
        # seirv_last_beta:
        'y_pred_seirv_last_beta_mean': seirv_last_beta_only_results['y_pred_without_train_period'],
        'y_pred_seirv_last_beta_upper': None,
        'y_pred_seirv_last_beta_lower': None,

        # seirv_ml_beta:
        'y_pred_seirv_ml_beta_mean': seirv_ml_results['y_pred_mean'],
        'y_pred_seirv_ml_beta_upper': None,
        'y_pred_seirv_ml_beta_lower': None,

        # sarima:
        'y_pred_sarima_mean': sarima_results['predictions'],
        'y_pred_sarima_upper': sarima_results['upper'],
        'y_pred_sarima_lower': sarima_results['lower'],

        # ensemble:
        'y_pred_ensemble_mean': ensemble_results['y_pred_mean'],
        'y_pred_ensemble_upper': None,
        'y_pred_ensemble_lower': None,
    }

    return seirv_last_beta_only_results, seirv_ml_results, sarima_results, ensemble_results, all_combined


def convert_all_forecasts_to_incidences(forecasts: dict, pop_size_district: int) -> dict:
    results_dict = {}

    for k, v in forecasts.items():
        if v is not None:
            results_dict[k] = convert_seven_day_averages(v, pop_size_district)
        else:
            results_dict[k] = None

    return results_dict


def convert_seven_day_averages(forecast_array: np.array, pop_size_district: int) -> np.array:
    # Multiply with 7 to go from 7 day average to 7 day sum
    # Then divide by population size and multiply with 100k to get incidences
    return np.array(forecast_array) * 7 / pop_size_district * 100_000
