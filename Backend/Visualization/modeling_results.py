import numpy as np
import matplotlib.pyplot as plt


def plot_train_and_fitted_infections(y_train: np.array, y_pred: np.array):
    len_train = len(y_train)
    len_pred = len(y_pred)
    len_total = len_train + len_pred
    t_grid_train = np.linspace(1, len_train, len_train)
    t_grid_pred = np.linspace(1 + len_train, len_pred+len_train, len_pred)
    t_grid_total = np.linspace(1, len_total, len_total)

    plt.plot(t_grid_train, y_train)
    plt.plot(t_grid_pred, y_pred)
    plt.show()

    pass


def plot_train_and_val_infections(y_train:np.array, y_val:np.array):

    assert len(y_train) == len(y_val)
    len_total = len(y_train)

    plt.plot(y_train)
    plt.plot(y_val)
    plt.show()

    pass