import pandas as pd
import numpy as np

from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_model_pipeline
from Backend.Modeling.Simulate_Infection_Cases.simulate_infection_counts import produce_simulated_infection_counts, set_starting_values
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Visualization.modeling_results import plot_train_and_fitted_infections

import matplotlib.pyplot as plt


def main():

    ### Insert import data part here: ###
    rki_data = pd.read_csv('./Assets/Data/rki_data_161121.csv', index_col=0)

    # Get simulated infection cases:
    simulated_inf_cases = produce_simulated_infection_counts()

    # Get starting values for compartmental model (Should come from the data pipeline later on)
    start_vals = set_starting_values(simulated_inf_cases)

    # Call seirv_model pipeline:
    y_pred = seirv_model_pipeline(simulated_inf_cases, start_vals)

    # Visualize model pipeline run:
    plot_train_and_fitted_infections(y_train=simulated_inf_cases, y_pred=y_pred)

    # Compute metrics:
    scores = compute_evaluation_metrics(y_pred=y_pred, y_true=simulated_inf_cases)



    print('end reached')





if __name__ == '__main__':
    main()
