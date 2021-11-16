import pandas as pd
import json

from Backend.Modeling.Differential_Equation_Modeling.model import run_SEIRV_model

def main():
    run_SEIRV_model()

    print('end reached')


if __name__ == '__main__':
    main()