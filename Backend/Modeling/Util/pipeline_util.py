import numpy as np
from datetime import date, time, datetime, timedelta

from Backend.Data.opendata_db_calls import get_list_of_districts
import random


def train_test_split(data, validation_duration=14):
    total_data = len(data)

    if total_data > validation_duration:
        train_data, test_data = data[:total_data - validation_duration], data[total_data - validation_duration:]
        return train_data, test_data

    else:
        validation_duration = int(total_data * 0.3)
        train_data, test_data = data[:total_data - validation_duration], data[total_data - validation_duration:]
        return train_data, test_data  # TODO exception


def models_params_to_dictionary(params: np.array):
    param_dict = {
        'beta': params[0],
        'gamma_I': params[1],
        'gamma_U': params[2],
        'delta': params[3],
        'theta': params[4],
        'rho': params[5],
    }
    return param_dict


def models_compartment_values_to_dictionary(vals: np.array):
    vals_dict = {
        'S': vals[0],
        'E': vals[1],
        'I': vals[2],
        'U': vals[3],
        'R': vals[4],
        'V': vals[5],
    }
    return vals_dict


def get_list_of_random_dates(num: int, lower_bound: str, upper_bound: str, seed=42) -> list:

    np.random.seed(seed)

    lower_bound_day = datetime.strptime(lower_bound, '%Y-%m-%d')
    upper_bound_day = datetime.strptime(upper_bound, '%Y-%m-%d')

    diff_bounds = abs((lower_bound_day-upper_bound_day).days)

    # discrete uniform distribution:
    random_numbers = list(range(0, diff_bounds+1))
    random.shuffle(random_numbers)


    # create list of random dates as datetimes:
    random_dates_as_datetimes = [lower_bound_day+timedelta(days=d) for d in random_numbers[0:num]]
    # and then convert to strings:
    random_dates_as_strings = [d.strftime('%Y-%m-%d') for d in random_dates_as_datetimes]

    return random_dates_as_strings


def get_list_of_random_districts(num, seed=42):
    np.random.seed(seed)

    ### temporary solution:
    if True:
        all_districts_lst = [
            'Essen', 'Münster', 'Herne', 'Bielefeld', 'Dortmund', 'Berlin', 'Ingolstadt', 'Odenwaldkreis', 'Salzgitter',
            'Verden', 'Heidenheim', 'Bremen', 'Erding', 'Friesland', 'Celle', 'Northeim', 'Braunschweig'
            # 'Vorpommern_Greifswald', 'Suhl', 'Goerlitz', 'Chemnitz', 'Magdeburg'
        ]


    #### Code below will be updated:
    if False:
        all_districts_lst = get_list_of_districts()

    # Ensure that number of desired districts does not exceed number of districts:
    if num > len(all_districts_lst):
        raise Exception(f'Number of desired districts {num} exceeds number of districts in list {len(all_districts_lst)}')


    # discrete uniform distribution:
    random_indices = list(range(0, len(all_districts_lst)))
    random.shuffle(random_indices)

    random_districts = [all_districts_lst[idx] for idx in random_indices[0:num]]

    return random_districts

