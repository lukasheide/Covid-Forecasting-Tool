import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def compute_evaluation_metrics(y_true, y_pred):
    actual = y_true.copy()
    pred = y_pred.copy()

    if len(actual) != len(pred):
        # handle case in which y_true also contains starting value in t=0 and y_pred doesn't.
        if len(actual) - 1 == len(y_pred):
            # crop first value:
            actual = actual[1:]

    # Compute metrics:
    metrics = {
        'rmse': mean_squared_error(y_true=actual, y_pred=pred),
        'mape': mean_absolute_percentage_error(y_true=actual, y_pred=pred)
    }

    return metrics


def compute_rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def compute_mape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)

