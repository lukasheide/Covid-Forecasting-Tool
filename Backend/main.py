import pandas as pd
import numpy as np

from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_model_pipeline_DEPRECATED, \
    fit_seirv_model, seirv_pipeline
from Backend.Modeling.Simulate_Infection_Cases.simulate_infection_counts import produce_simulated_infection_counts, \
    set_starting_values, set_starting_values_e0_fitted, set_starting_values_e0_and_i0_fitted
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Visualization.modeling_results import plot_train_and_fitted_infections, plot_train_infections

import matplotlib.pyplot as plt


def main():

    ### Insert import data part here: ###
    rki_data = pd.read_csv('./Assets/Data/rki_data_161121.csv', index_col=0)


    ######################################################################
    # Currently the SEIRV model is not connected with the data pipeline. #
    # Instead simulated data was used for building the model pipeline.   #
    ######################################################################

    
    # Get simulated infection cases:
    simulated_inf_cases = produce_simulated_infection_counts()

    # Get starting values for compartmental model (Should come from the data pipeline later on)
    start_vals = set_starting_values_e0_and_i0_fitted(simulated_inf_cases)

    # Call seirv_model pipeline:
    pipeline_result = seirv_pipeline(y_train=simulated_inf_cases, start_vals_fixed=start_vals)
    y_pred = pipeline_result['y_pred']

    # Visualize model pipeline run:
    plot_train_and_fitted_infections(y_train=simulated_inf_cases, y_pred=y_pred)

    # Compute metrics:
    scores = compute_evaluation_metrics(y_pred=y_pred, y_true=simulated_inf_cases)



    print('end reached')





if __name__ == '__main__':
    main()
