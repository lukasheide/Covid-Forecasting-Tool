import numpy as np


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
