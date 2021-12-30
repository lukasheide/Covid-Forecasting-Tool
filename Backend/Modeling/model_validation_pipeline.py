import pandas as pd
import numpy as np
from datetime import date, time, datetime

from Backend.Data.data_access_methods import get_smoothen_cases, get_starting_values
from Backend.Data.data_util import Column, date_int_str
from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Modeling.Util.pipeline_util import train_test_split
from Backend.Visualization.modeling_results import plot_train_and_fitted_infections_line_plot, \
    plot_train_and_fitted_infections_bar_plot, plot_train_infections, plot_train_fitted_and_validation
from Backend.Data.db_functions import get_table_data


def diff_eq_model_validation_pipeline(end_date: date, duration: int, districts: list, validation_duration: int,
                                      visualize=False, verbose=False) -> None:
    # iterate over districts(list) of interest
    for district in districts:
        ## 1) Import Data
        # 1a) get_smoothed_infection_counts() -> directly from Database
        smoothen_cases = get_smoothen_cases(district, end_date, duration)

        # 1b) split_into_train_and_validation()
        y_train, y_val = train_test_split(data=smoothen_cases[Column.SEVEN_DAY_SMOOTHEN.value],
                                          validation_duration=validation_duration)

        # 1c) get_starting_values() -> N=population, R0=recovered to-the-date, V0=vaccinated to-the-date
        train_start_date = date_int_str(y_train[Column.DATE.value][0])
        start_vals = get_starting_values(district, train_start_date)

        ## 2) Run model_pipeline

        pipeline_result = seirv_pipeline(y_train=y_train, start_vals_fixed=start_vals)
        y_pred = pipeline_result['y_pred']

        # returned:
        # I) y_pred for both training and validation period,
        # II) Model Params (greek letters)
        # III) Model Starting values (number of people for each compartment at time t)

        ## 3) Evaluate the results

        # 3a) Visualize results (mainly for debugging)
        plot_train_fitted_and_validation(y_train=y_train, y_val=y_val, y_pred=y_pred)

        # 3b) Compute metrics (RMSE, MAPE, ...)
        scores = compute_evaluation_metrics(y_pred=y_pred, y_val=y_val)

        ## 4) Store results in database:

        ## 4a) Meta parameters
        # 1) which model?
        # 2) what period?
        # 3) with what parameters?

        ## 4b) Predictions

        # -> basically all parameters that are set


if __name__ == '__main__':
    smoothen_cases = get_smoothen_cases('Münster', '2021-03-31', 40)
    # y_train, y_val = train_test_split(smoothen_cases)
    # train_start_date = date_int_str(y_train[Column.DATE.value][0])
    # start_vals = get_starting_values('Münster', train_start_date)
    # print(start_vals)
