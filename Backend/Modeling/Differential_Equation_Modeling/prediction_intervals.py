import numpy as np
import pandas as pd

from Backend.Data.DataManager.remote_db_manager import download_pred_intervals_file


def compute_prediction_intervals(y_pred, intervals_residuals_df, avg_pred, model_name):
    # Get classes depending on how the infections currently are:
    classes = list(set(intervals_residuals_df['class'].to_list()))

    # Find class:
    inf_idx = [n for n, i in enumerate(classes) if i > avg_pred][0]
    inf_class = classes[inf_idx]

    # Transform y_pred to numpy array:
    y_target = np.array(y_pred)

    df = intervals_residuals_df[intervals_residuals_df['class'] == inf_class]

    if model_name == 'seirv_last_beta':
        percentages_upper = np.array(df['upper_perc_diff_eq_last_beta'])
        multipliers_upper = percentages_upper + 1

        percentages_lower = np.array(df['lower_perc_diff_eq_last_beta'])
        multipliers_lower = percentages_lower + 1

    elif model_name == 'seirv_ml_beta':
        percentages_upper = np.array(df['upper_perc_diff_eq_ml_beta'])
        multipliers_upper = percentages_upper + 1

        percentages_lower = np.array(df['lower_perc_diff_eq_ml_beta'])
        multipliers_lower = percentages_lower + 1

    # Apply Multipliers to target:
    y_pred_upper = y_target * multipliers_upper
    y_pred_lower = y_target * multipliers_lower

    return y_pred_upper, y_pred_lower


def compute_ensemble_forecast(ensemble_model_share, seirv_last_beta_only_results, seirv_ml_results, sarima_results):

    # Compute weighted average:
    ensemble_y_pred_point = ensemble_model_share['seirv_last_beta'] * seirv_last_beta_only_results['y_pred_without_train_period'] + \
                            ensemble_model_share['seirv_ml_beta'] * seirv_ml_results['y_pred_mean'] + \
                            ensemble_model_share['sarima'] * sarima_results['predictions']

    ensemble_y_pred_upper = ensemble_model_share['seirv_last_beta'] * seirv_last_beta_only_results['y_pred_without_train_period_upper_bound'] + \
                            ensemble_model_share['seirv_ml_beta'] * seirv_ml_results['y_pred_upper'] + \
                            ensemble_model_share['sarima'] * sarima_results['upper']

    ensemble_y_pred_lower = ensemble_model_share['seirv_last_beta'] * seirv_last_beta_only_results['y_pred_without_train_period_lower_bound'] + \
                            ensemble_model_share['seirv_ml_beta'] * seirv_ml_results['y_pred_lower'] + \
                            ensemble_model_share['sarima'] * sarima_results['lower']

    ensemble_results_dict = {
        'y_pred_mean': ensemble_y_pred_point,
        'y_pred_upper': ensemble_y_pred_upper,
        'y_pred_lower': ensemble_y_pred_lower,
    }

    return ensemble_results_dict


def get_prediction_intervals():
    # get the latest version of the file from the server before reading
    # download_pred_intervals_file()
    return pd.read_csv('../Assets/Forecasts/PredictionIntervals/prediction_intervals.csv', delimiter=",")