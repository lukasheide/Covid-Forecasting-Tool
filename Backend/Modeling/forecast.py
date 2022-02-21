import numpy as np
import pandas as pd

from Backend.Modeling.Differential_Equation_Modeling.prediction_intervals import compute_prediction_intervals, \
    compute_ensemble_forecast
from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline
from Backend.Modeling.Differential_Equation_Modeling.seirv_model_and_ml import seirv_ml_layer
from Backend.Modeling.Regression_Model.ARIMA import sarima_pipeline


def forecast_all_models(y_train_diffeq, y_train_sarima, forecasting_horizon, ml_matrix_predictors,
                        start_vals_seirv, fixed_model_params_seirv,
                        standardizer_obj, ml_model, district, ensemble_model_share,
                        pred_intervals_df=None,
                        run_diff_eq_last_beta=True, run_diff_eq_ml_beta=True, run_sarima=True, run_ensemble=True
                        ):
    ## 3.1) SEIRV + Last Beta
    if run_diff_eq_last_beta:
        # Get point estimates from SEIRV Pipeline:
        seirv_last_beta_only_results = seirv_pipeline(y_train=y_train_diffeq,
                                                      start_vals_fixed=start_vals_seirv,
                                                      fixed_model_params=fixed_model_params_seirv,
                                                      forecast_horizon=forecasting_horizon,
                                                      allow_randomness_fixed_beta=False, district=district)
        ## Compute Upper and Lower Bound:
        if pred_intervals_df is not None:
            upper_bound, lower_bound = compute_prediction_intervals(
                y_pred=seirv_last_beta_only_results['y_pred_without_train_period'],
                intervals_residuals_df=pred_intervals_df,
                avg_pred=np.mean(seirv_last_beta_only_results['y_pred_without_train_period']),
                model_name='seirv_last_beta')

            seirv_last_beta_only_results['y_pred_without_train_period_upper_bound'] = upper_bound
            seirv_last_beta_only_results['y_pred_without_train_period_lower_bound'] = lower_bound

    ## 3.2) SEIRV + Machine Learning Layer
    if run_diff_eq_ml_beta:
        seirv_ml_results = seirv_ml_layer(y_train_diffeq, start_vals_seirv, fixed_model_params_seirv,
                                          forecasting_horizon, ml_matrix_predictors, standardizer_obj, ml_model,
                                          pred_intervals_df)

    ## 3.3) SARIMA
    if run_sarima:
        sarima_results = sarima_pipeline(y_train=y_train_sarima,
                                         forecasting_horizon=forecasting_horizon)

    ## 3.4) Ensemble Model
    if run_ensemble and run_diff_eq_last_beta and run_diff_eq_ml_beta and run_sarima:
        ensemble_results = compute_ensemble_forecast(ensemble_model_share=ensemble_model_share,
                                                     seirv_last_beta_only_results=seirv_last_beta_only_results,
                                                     seirv_ml_results=seirv_ml_results,
                                                     sarima_results=sarima_results)

    # Create results dicts with Nones in case a model is not run:
    if not run_diff_eq_last_beta:
        seirv_last_beta_only_results = {
            'y_pred_without_train_period': None,
            'y_pred_without_train_period_upper_bound': None,
            'y_pred_without_train_period_lower_bound': None,
        }
    if not run_diff_eq_ml_beta:
        seirv_ml_results = {
            'y_pred_mean': None,
            'y_pred_upper': None,
            'y_pred_lower': None,
        }
    if not run_sarima:
        sarima_results = {
            'predictions': None,
            'upper': None,
            'lower': None,
        }
    if not (run_ensemble and run_diff_eq_last_beta and run_diff_eq_ml_beta and run_sarima):
        ensemble_results = {
            'y_pred_mean': None,
            'y_pred_upper': None,
            'y_pred_lower': None,
        }

    all_combined = {
        # seirv_last_beta:
        'y_pred_seirv_last_beta_mean': seirv_last_beta_only_results['y_pred_without_train_period'],
        'y_pred_seirv_last_beta_upper': seirv_last_beta_only_results['y_pred_without_train_period_upper_bound'],
        'y_pred_seirv_last_beta_lower': seirv_last_beta_only_results['y_pred_without_train_period_lower_bound'],

        # seirv_ml_beta:
        'y_pred_seirv_ml_beta_mean': seirv_ml_results['y_pred_mean'],
        'y_pred_seirv_ml_beta_upper': seirv_ml_results['y_pred_upper'],
        'y_pred_seirv_ml_beta_lower': seirv_ml_results['y_pred_lower'],

        # sarima:
        'y_pred_sarima_mean': sarima_results['predictions'],
        'y_pred_sarima_upper': sarima_results['upper'],
        'y_pred_sarima_lower': sarima_results['lower'],

        # ensemble:
        'y_pred_ensemble_mean': ensemble_results['y_pred_mean'],
        'y_pred_ensemble_upper': ensemble_results['y_pred_upper'],
        'y_pred_ensemble_lower': ensemble_results['y_pred_lower'],
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


def convert_seven_day_averages(array: np.array, pop_size_district: int) -> np.array:
    # Multiply with 7 to go from 7 day average to 7 day sum
    # Then divide by population size and multiply with 100k to get incidences
    return np.array(array) * 7 / pop_size_district * 100_000
