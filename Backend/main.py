import pandas as pd
import numpy as np

from Backend.Modeling.Differential_Equation_Modeling.model import run_SEIRV_model, fit_SEIRV_model
from Backend.Modeling.Differential_Equation_Modeling.diffEqPlayground import model_fit
from Backend.Modeling.Differential_Equation_Modeling.diffEqFittingAttempt import outer_function
from Backend.Modeling.Differential_Equation_Modeling.diffEqPlayground2 import outer_function
from Backend.Modeling.Differential_Equation_Modeling.diffEqPlayground3 import runner
from Backend.Modeling.Differential_Equation_Modeling.diffEqPlayground4 import runner

def main():

    ##### to be deleted later on #####
    rki_data = pd.read_csv('./Assets/Data/rki_data_161121.csv', index_col=0)

    # fake infection cases for 28 days:
    inf_cases = [1000]
    inc_factor = 1.03
    num_days = 28

    # set seed for drawing random numbers:
    np.random.seed(42)

    # create list with infection cases for 28 days
    for d in range(num_days):
        noise = np.random.normal(0, 0.05)

        # use previous day as base for next day and multiply it with a factor to add some noise
        new_val = inf_cases[-1] * (inc_factor+noise)

        inf_cases.append(new_val)

    # transform list to numpy array
    inf_cases = np.array(inf_cases)

    # get starting values for compartmental model and forward them
    start_vals = set_starting_values()

    runner(inf_cases)



    # fit_SEIRV_model(num_days_train=num_days, train_data=inf_cases, starting_values=start_vals)
    #run_SEIRV_model()

    print('end reached')


# also to be deleted later on:
def set_starting_values():
    S0 = 120000
    E0 = 250
    I0 = 1000
    R0 = 300000 - S0 - E0 - I0
    I_cum = I0 + R0
    V0 = 150000

    return S0, E0, I0, R0, I_cum, V0


if __name__ == '__main__':
    main()
