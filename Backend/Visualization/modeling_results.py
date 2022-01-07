import numpy as np
import matplotlib.pyplot as plt


def plot_train_infections(y_train: np.array):
    plt.plot(y_train)
    plt.show()
    plt.clf()


def plot_train_fitted_and_validation(y_train: np.array, y_val: np.array, y_pred: np.array):
    len_train = len(y_train)
    len_val = len(y_val)
    len_pred = len(y_pred)

    # Split y_pred into train and val period:
    y_pred_train = y_pred[:len_train-1]
    y_pred_val = y_pred[len_train-1:]
    y_pred_val_plus_one = y_pred[len_train-2:]

    # Cut of first day of y_train
    t_grid_train = np.linspace(1, len_train-1, len_train-1)
    t_grid_val = np.linspace(len_train, len_pred, len_val)
    t_grid_val_plus_one = np.linspace(len_train-1, len_pred, len_val+1)
    t_grid_pred = np.linspace(1, len_pred, len_pred)

    plt.plot(t_grid_train, y_pred_train, color='#2077b4', zorder=5)
    plt.plot(t_grid_val_plus_one, y_pred_val_plus_one, color='#76324e', zorder=5)
    plt.scatter(x=t_grid_val, y=y_val, s=40, color='#0f0f0f', zorder=10)
    plt.scatter(x=t_grid_train, y=y_train[1:], s=40, color='#ff7f0f', zorder=15)

    plt.show()
    plt.clf()


def plot_train_and_fitted_infections_line_plot(y_train: np.array, y_pred: np.array):
    len_train = len(y_train)
    len_total = len(y_pred)
    t_grid_train = np.linspace(1, len_train, len_train)
    t_grid_total = np.linspace(1, len_total, len_total)

    plt.scatter(x=t_grid_train, y=y_train, s=15)
    plt.plot(t_grid_total, y_pred)

    plt.show()
    plt.clf()


def plot_train_and_fitted_infections_bar_plot(y_train: np.array, y_pred: np.array):
    len_train = len(y_train)
    len_total = len(y_pred)
    t_grid_train = np.linspace(1, len_train, len_train)
    t_grid_total = np.linspace(1, len_total, len_total)

    plt.bar(x=t_grid_total, height=y_pred, color='#2077b4', zorder=5)
    plt.scatter(x=t_grid_train, y=y_train, s=40, color='#ff7f0f', zorder=10)

    plt.show()
    plt.clf()


def plot_train_and_fitted_infections_DEPRECATED(y_train: np.array, y_pred: np.array):
    len_train = len(y_train)
    len_pred = len(y_pred)
    len_total = len_train + len_pred
    t_grid_train = np.linspace(1, len_train, len_train)
    t_grid_pred = np.linspace(len_train, len_pred+len_train-1, len_pred)
    t_grid_total = np.linspace(1, len_total-1, len_total-1)

    plt.plot(t_grid_train, y_train)
    plt.plot(t_grid_pred, y_pred)
    plt.show()
    plt.clf()


def plot_train_and_val_infections(y_train:np.array, y_val:np.array):

    if len(y_val) > len(y_train):
        plt.plot(y_train)
        plt.plot(y_val[1:])
        plt.show()
        plt.clf()

    else:
        plt.plot(y_train)
        plt.plot(y_val)
        plt.show()
        plt.clf()
