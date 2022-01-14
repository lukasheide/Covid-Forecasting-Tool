import pandas as pd
import numpy as np

from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline
from Backend.Modeling.Simulate_Infection_Cases.simulate_infection_counts import produce_simulated_infection_counts, \
    set_starting_values, set_starting_values_e0_fitted, set_starting_values_e0_and_i0_fitted
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Visualization.modeling_results import plot_train_and_fitted_infections_line_plot, \
    plot_train_and_fitted_infections_bar_plot, plot_train_infections, plot_train_fitted_and_validation
from Backend.Modeling.model_validation_pipeline import diff_eq_pipeline, diff_eq_pipeline_wrapper

from Backend.Data.db_functions import get_table_data

from datetime import date, time, datetime
import matplotlib.pyplot as plt
import matplotlib

matplotlib.interactive(True)


def main():
    # Call wrapper function used for finding optimal training period length:
    diff_eq_pipeline_wrapper()

    # Call differential equation model validation pipeline:
    end_date = '2021-12-14'
    time_frame_train_and_validation = 28
    forecasting_horizon = 14
    districts = ['Essen', 'Münster', 'Herne', 'Bielefeld', 'Dortmund', 'Leipzig, Stadt', 'Berlin']

    # Call differential equation model validation pipeline:
    diff_eq_pipeline(train_end_date=end_date,
                     duration=time_frame_train_and_validation,
                     districts=districts,
                     validation_duration=forecasting_horizon,
                     visualize=True,
                     verbose=False,
                     validate=True, # should be similar to 'visualize' boolean value
                     store_results_to_db=True)


    ##### Stuff below will be refactored soon #####
    # end_date = 20210804
    # start_date = 20210901
    #
    # muenster_last_28_days = get_table_data(table='Essen', date1=end_date, date2=start_date,
    #                                        attributes=['date', 'seven_day_infec'], with_index=False)
    #
    # # Split into train and validation set:
    # y_train_actual = np.array(muenster_last_28_days['seven_day_infec'])[0:15]
    # y_val_actual = np.array(muenster_last_28_days['seven_day_infec'])[15:]
    #
    # # Get simulated infection cases:
    # y_train_simulation = produce_simulated_infection_counts()
    #
    # # Get starting values for compartmental model (Should come from the data pipeline later on)
    # start_vals = set_starting_values_e0_and_i0_fitted()
    #
    # # Call seirv_model pipeline:
    # pipeline_result = seirv_pipeline(y_train=y_train_actual, start_vals_fixed=start_vals)
    # y_pred = pipeline_result['y_pred_without_train_period']
    #
    # # Visualize model pipeline run:
    # plot_train_fitted_and_validation(y_train=y_train_actual, y_val=y_val_actual, y_pred=y_pred)
    #
    # # Compute metrics:
    # scores = compute_evaluation_metrics(y_pred=y_pred, y_val=y_val_actual)

    print('end reached')


if __name__ == '__main__':
    main()
