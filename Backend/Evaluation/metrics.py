import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error


def compute_evaluation_metrics(y_val, y_pred):
    actual = y_val.copy()
    pred = y_pred.copy()

    if len(actual) != len(pred):
        # use only predictions for past n days:
        pred = pred[-len(actual):]


    # Compute metrics:
    metrics = {
        'rmse': mean_squared_error(y_true=actual, y_pred=pred, squared=False),
        'mape': mean_absolute_percentage_error(y_true=actual, y_pred=pred),
        'mae': mean_absolute_error(y_true=actual, y_pred=pred)
    }

    return metrics


def compute_rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def compute_mape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)

