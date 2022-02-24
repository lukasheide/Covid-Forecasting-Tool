import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Backend.Data.DataManager.data_util import create_dates_array

## True Data:
# train:

orange = '#ff7f0f'
# validation:
black = '#0f0f0f'

## Predictions
# train:
lightblue = '#2077b4'
# forecast:
purple = '#76324e'


def plot_train_infections(y_train: np.array):
    plt.clf()
    plt.plot(y_train)
    plt.show()


def plot_train_fitted_and_predictions(y_train_fitted: np.array, y_train_true: np.array, y_pred_full: np.array,
                                      district=None, pred_start_date=None,
                                      save_results=False):
    plt.clf()

    len_train = len(y_train_true)
    len_total = len(y_pred_full)

    # Create timegrids:
    t_grid_train = np.linspace(1, len_train, len_train)
    t_grid_pred = np.linspace(len_train + 1, len_total, len_train)
    t_grid_full = np.linspace(1, len_total, len_total)

    # Training data:
    plt.scatter(x=t_grid_train, y=y_train_true, s=40, color=orange, zorder=15, label='Training Data')

    ## Fitted data:
    # Training:
    plt.plot(t_grid_full, y_pred_full, color=lightblue, zorder=5, linewidth=2.5, label='Fitted Line')
    # Prediction:
    plt.plot(t_grid_pred, y_pred_full[len_train:], color=purple, zorder=10, linewidth=2.5, label='Forecast')

    # Axis description:
    plt.title(f'Forecast for {district} starting on {pred_start_date}')
    plt.ylabel('7-day average infections')
    plt.xlabel('Days')
    plt.legend(loc="upper left")

    if save_results:
        plt.savefig(f'Assets/Forecasts/Plots/Forecast_{district}_StartDate_{pred_start_date}.png')
        temp_df = pd.DataFrame({'Training': pd.Series(y_train_true),
                                'Prediction': pd.Series(y_pred_full)
                                })

        temp_df.to_csv(
            path_or_buf=f'Assets/Forecasts/CSV_files/Forecast_{district}_StartDate_{pred_start_date}.csv',
            sep=';'
        )

    plt.show()


def plot_train_fitted_and_validation(y_train: np.array, y_val: np.array, y_pred: np.array, y_pred_upper=None,
                                     y_pred_lower=None):
    plt.clf()
    len_train = len(y_train)
    len_val = len(y_val)
    len_pred = len(y_pred)

    # Split y_pred into train and val period:
    y_pred_train = y_pred[:len_train]
    y_pred_val = y_pred[len_train:]
    y_pred_val_plus_one = y_pred[len_train - 1:]

    t_grid_train = np.linspace(1, len_train, len_train)
    t_grid_val = np.linspace(len_train + 1, len_pred, len_val)
    t_grid_val_plus_one = np.linspace(len_train, len_pred, len_val + 1)
    t_grid_pred = np.linspace(1, len_pred, len_pred)

    plt.plot(t_grid_train, y_pred_train, color=lightblue, zorder=5, linewidth=2.5)
    plt.plot(t_grid_val_plus_one, y_pred_val_plus_one, color=purple, zorder=5, linewidth=2.5)
    plt.scatter(x=t_grid_val, y=y_val, s=40, color=black, zorder=10)
    plt.scatter(x=t_grid_train, y=y_train, s=40, color=orange, zorder=15)

    if y_pred_upper is not None and y_pred_lower is not None:
        y_pred_upper_plus_one = np.insert(y_pred_upper, 0, y_pred_val_plus_one[0])
        y_pred_lower_plus_one = np.insert(y_pred_lower, 0, y_pred_val_plus_one[0])
        plt.plot(t_grid_val_plus_one, y_pred_upper_plus_one, color=purple, zorder=5, linestyle='dashed')
        plt.plot(t_grid_val_plus_one, y_pred_lower_plus_one, color=orange, zorder=5, linestyle='dashed')

    plt.show()


def plot_train_and_fitted_infections_line_plot(y_train: np.array, y_pred: np.array):
    plt.clf()
    len_train = len(y_train)
    len_pred = len(y_pred)

    corrected_train_length = min(len_train, len_pred)

    y_train_copy = y_train.copy()
    y_pred_copy = y_pred.copy()

    if len_train != len_pred:
        if len_pred == len_train - 1:
            y_train_copy = y_train_copy[1:len_train]
            y_pred_copy = y_pred_copy[0:len_pred]
        else:
            y_train_copy = y_train_copy[0:corrected_train_length]
            y_pred_copy = y_pred_copy[0:corrected_train_length]

    t_grid = np.linspace(0, corrected_train_length - 1, corrected_train_length)

    plt.scatter(x=t_grid, y=y_train_copy, s=40, color='#0f0f0f', zorder=10)
    plt.plot(t_grid, y_pred_copy, color='#2077b4', zorder=5, linewidth=2.5)

    plt.show()


def plot_train_and_fitted_infections_bar_plot(y_train: np.array, y_pred: np.array):
    plt.clf()
    len_train = len(y_train)
    len_total = len(y_pred)
    t_grid_train = np.linspace(1, len_train, len_train)
    t_grid_total = np.linspace(1, len_total, len_total)

    plt.bar(x=t_grid_total, height=y_pred, color='#2077b4', zorder=5)
    plt.scatter(x=t_grid_train, y=y_train, s=40, color='#ff7f0f', zorder=10)

    plt.show()


def plot_train_and_fitted_infections_DEPRECATED(y_train: np.array, y_pred: np.array):
    plt.clf()
    len_train = len(y_train)
    len_pred = len(y_pred)
    len_total = len_train + len_pred
    t_grid_train = np.linspace(1, len_train, len_train)
    t_grid_pred = np.linspace(len_train, len_pred + len_train - 1, len_pred)
    t_grid_total = np.linspace(1, len_total - 1, len_total - 1)

    plt.plot(t_grid_train, y_train)
    plt.plot(t_grid_pred, y_pred)
    plt.show()


def plot_train_and_val_infections(y_train: np.array, y_val: np.array):
    plt.clf()

    if len(y_val) > len(y_train):
        plt.plot(y_train)
        plt.plot(y_val[1:])
        plt.show()

    else:
        plt.plot(y_train)
        plt.plot(y_val)
        plt.show()


# create line plot for SARIMA visualization
def plot_sarima_val_line_plot(train_array, test_array, predictions: int, pred_start_date, district):
    plt.clf()

    len_train = len(train_array)
    len_test = len(test_array)
    len_total = len_train + len_test
    t_grid_val = np.linspace(len_train, len_total, len_test + 1)
    t_grid_train = np.linspace(1, len_train, len_train)

    pred_array = np.concatenate((train_array[len_train - 1:], predictions))
    val_array = np.concatenate((train_array[len_train - 1:], test_array))

    plt.plot(t_grid_val, pred_array, color=purple, zorder=5, linewidth=2.5, label='Forecast')
    plt.plot(t_grid_train, train_array, color=lightblue, zorder=5, linewidth=2.5, label='Training Line')
    plt.scatter(x=t_grid_val, y=val_array, s=40, zorder=10, color=black, label='Real Cases')

    plt.title(f'Validation for {district} starting on {pred_start_date}')
    plt.xlabel("Days")
    plt.ylabel("7-day average infections")
    plt.legend(loc="upper left")

    # plt.show()

    plt.savefig(f'Assets/Forecasts/Plots/Sarima_Evaluate_{district}_StartDate_{pred_start_date}.png')


def plot_sarima_pred_plot(y_train, predictions: int, district, pred_start_date):
    plt.clf()

    len_train = len(y_train)
    len_test = len(predictions)
    len_total = len_train + len_test
    t_grid_val = np.linspace(len_train, len_total, len_test + 1)
    t_grid_train = np.linspace(1, len_train, len_train)

    pred_array = np.concatenate((y_train[len_train - 1:], predictions))

    ## Fitted data:
    # Training:
    plt.plot(t_grid_train, y_train, color=lightblue, zorder=5, linewidth=2.5, label='Training Line')
    # Prediction:
    plt.plot(t_grid_val, pred_array, color=purple, zorder=10, linewidth=2.5, label='Forecast')

    # Axis description:
    plt.title(f'Forecast for {district} starting on {pred_start_date}')
    plt.ylabel('7-day average infections')
    plt.xlabel('Days')
    plt.legend(loc="upper left")

    plt.show()

    plt.savefig(f'Assets/Forecasts/Plots/Sarima_Forecast_{district}_StartDate_{pred_start_date}.png')


def plot_evaluation_metrics(rmse, districts, i, round):
    name = (districts[i] + str(round))
    plt.bar(name, rmse)
    plt.xticks(color='orange', rotation=20, horizontalalignment='right')

    plt.title("Metrics Plot")
    plt.xlabel("District & Date")
    plt.ylabel("RMSE")

    plt.show()


def plot_beta_matrix_estimation(y_train_true, y_val_true, y_train_pred_full, y_val_pred, district, start_date=None,
                                end_date=None):
    plt.clf()

    len_train = len(y_train_true)
    len_val = len(y_val_true)
    len_total = len_train + len_val

    t_grid_train = np.linspace(1, len_train, len_train)
    t_grid_val = np.linspace(len_train + 1, len_total, len_val)
    t_grid_total = np.linspace(1, len_total, len_total)

    # Training data:
    plt.scatter(x=t_grid_train, y=y_train_true, s=40, color=orange, zorder=15, label='Training Data')
    plt.plot(t_grid_total, y_train_pred_full, color=lightblue, zorder=5, linewidth=2.5,
             label='Fitted Line (with beta t-1)')

    # Validation data:
    plt.scatter(x=t_grid_val, y=y_val_true, s=40, color=black, zorder=15, label='Validation Data')
    plt.plot(t_grid_val, y_val_pred, color=purple, zorder=5, linewidth=2.5, label='Fitted Line (best fit)')

    # Axis description:
    plt.title(f'Forecast for {district} - from {start_date} to {end_date}')
    plt.ylabel('7-day average infections')
    plt.xlabel('Days')
    plt.legend(loc="upper right")

    plt.show()


def plot_all_forecasts(forecast_dictionary, y_train, start_date_str, forecasting_horizon, district,
                       y_val=None,
                       y_train_fitted=None,
                       plot_val=False,
                       plot_y_train_fitted=False,
                       plot_y_train_fitted_all=False,
                       plot_diff_eq_last_beta=True,
                       plot_diff_eq_ml_beta=True,
                       plot_sarima=True,
                       plot_ensemble=True,
                       plot_predictions_intervals=False
                       ):

    """
    Flexible function to combine all sorts of combinations of the different models + prediction intervals, validation
    and training data into one plot.
    """


    plt.rcParams['figure.figsize'] = 10, 5
    plt.tight_layout()
    plt.clf()

    # Get dates array:
    dates_array = create_dates_array(start_date_str=start_date_str, num_days=len(y_train) + forecasting_horizon)

    # Colorcodes for each model:
    y_train_color = '#122499' # blue
    y_val_color = '#111111' # black
    diff_eq_last_beta_color = '#043b05'  # dark green
    diff_eq_ml_beta_color = '#1a4b75'  # light blue
    sarima_color = '#a62189'  # pink
    ensemble_color = '#d1930d'  # yellow

    # ...

    # Get length of periods:
    len_train = len(y_train)
    len_forecast = forecasting_horizon
    len_total = len_train + len_forecast

    t_grid_train = dates_array[:len_train]
    t_grid_forecasting = dates_array[-forecasting_horizon:]
    t_grid_forecasting_plus_one = dates_array[-(forecasting_horizon+1):]
    t_grid_all = dates_array

    ## Create plots:
    # Training Data:
    plt.scatter(x=t_grid_train, y=y_train, s=20, color=y_train_color, zorder=15, label='Training Data')
    plt.plot(t_grid_train, y_train, color=y_train_color, linewidth=1.5, zorder=12)

    # Validation Data:
    if plot_val:
        # Add last train point to y_val:
        y_val = np.append(np.array(y_train[-1]), y_val)
        plt.scatter(x=t_grid_forecasting_plus_one, y=y_val, s=20, color=y_val_color, zorder=15, label='Validation Data')
        plt.plot(t_grid_forecasting_plus_one, y_val, color=y_val_color, linewidth=1.5, zorder=12)

    ## Forecasts:

    # Fitted line:
    if y_train_fitted is not None and plot_y_train_fitted is True:
        if not plot_y_train_fitted_all:
            y_train_fitted_only_train = y_train_fitted[:len_train]
            plt.plot(t_grid_train, y_train_fitted_only_train, color=diff_eq_last_beta_color, zorder=5, linewidth=2.5,
                     label='SEIURV_LastBeta')
        else:
            plt.plot(t_grid_all, y_train_fitted, color=diff_eq_last_beta_color, zorder=5, linewidth=2.5,
                     label='SEIURV_LastBeta')



    # Diff Eq Last Beta:
    if plot_diff_eq_last_beta:
        y_pred_mean_diff_eq_last_beta = np.append(y_train[-1], forecast_dictionary['y_pred_seirv_last_beta_mean'])
        y_pred_upper_diff_eq_last_beta = np.append(y_train[-1], forecast_dictionary['y_pred_seirv_last_beta_upper'])
        y_pred_lower_diff_eq_last_beta = np.append(y_train[-1], forecast_dictionary['y_pred_seirv_last_beta_lower'])

        plt.plot(t_grid_forecasting_plus_one, y_pred_mean_diff_eq_last_beta, color=diff_eq_last_beta_color, zorder=5, linewidth=2.5,
                 label='SEIURV_LastBeta')

        if plot_predictions_intervals:
            plt.plot(t_grid_forecasting_plus_one, y_pred_upper_diff_eq_last_beta, linestyle='dashed', color=diff_eq_ml_beta_color, zorder=5, linewidth=1)
            plt.plot(t_grid_forecasting_plus_one, y_pred_lower_diff_eq_last_beta, linestyle='dashed', color=diff_eq_ml_beta_color, zorder=5, linewidth=1)


    # Diff Eq ML Beta:
    if plot_diff_eq_ml_beta:
        y_pred_mean_diff_eq_ml_beta = np.append(y_train[-1], forecast_dictionary['y_pred_seirv_ml_beta_mean'])
        y_pred_upper_diff_eq_ml_beta = np.append(y_train[-1], forecast_dictionary['y_pred_seirv_ml_beta_upper'])
        y_pred_lower_diff_eq_ml_beta = np.append(y_train[-1], forecast_dictionary['y_pred_seirv_ml_beta_lower'])

        plt.plot(t_grid_forecasting_plus_one, y_pred_mean_diff_eq_ml_beta, color=diff_eq_ml_beta_color, zorder=5, linewidth=2.5,
                 label='SEIURV_MLBeta')

        if plot_predictions_intervals:
            plt.plot(t_grid_forecasting_plus_one, y_pred_upper_diff_eq_ml_beta, linestyle='dashed', color=diff_eq_ml_beta_color, zorder=5, linewidth=1)
            plt.plot(t_grid_forecasting_plus_one, y_pred_lower_diff_eq_ml_beta, linestyle='dashed', color=diff_eq_ml_beta_color, zorder=5, linewidth=1)


    # SArima:
    if plot_sarima:
        y_pred_sarima_mean = np.append(y_train[-1], forecast_dictionary['y_pred_sarima_mean'])
        y_pred_sarima_upper = np.append(y_train[-1], forecast_dictionary['y_pred_sarima_upper'])
        y_pred_sarima_lower = np.append(y_train[-1], forecast_dictionary['y_pred_sarima_lower'])
        plt.plot(t_grid_forecasting_plus_one, y_pred_sarima_mean, color=sarima_color, zorder=5, linewidth=2.5,
                 label='Arima')

        if plot_predictions_intervals:
            plt.plot(t_grid_forecasting_plus_one, y_pred_sarima_upper, linestyle='dashed', color=sarima_color, zorder=5, linewidth=1)
            plt.plot(t_grid_forecasting_plus_one, y_pred_sarima_lower, linestyle='dashed', color=sarima_color, zorder=5, linewidth=1)

    # Ensemble:
    if plot_ensemble:
        y_pred_ensemble_mean = np.append(y_train[-1], forecast_dictionary['y_pred_ensemble_mean'])
        y_pred_ensemble_upper = np.append(y_train[-1], forecast_dictionary['y_pred_ensemble_upper'])
        y_pred_ensemble_lower = np.append(y_train[-1], forecast_dictionary['y_pred_ensemble_lower'])

        plt.plot(t_grid_forecasting_plus_one, y_pred_ensemble_mean, color=ensemble_color, zorder=5, linewidth=2.5,
                 label='Ensemble')

        if plot_predictions_intervals:
            plt.plot(t_grid_forecasting_plus_one, y_pred_ensemble_upper, linestyle='dashed', color=ensemble_color, zorder=5, linewidth=1)
            plt.plot(t_grid_forecasting_plus_one, y_pred_ensemble_lower, linestyle='dashed', color=ensemble_color, zorder=5, linewidth=1)



    ## Making stuff pretty:
    # Axis description:
    plt.title(f'Forecast for {district} starting on {t_grid_forecasting[0]}')
    plt.ylabel('7-day Incidence')
    plt.xlabel('Days')


    # If last value of y_train is larger than the first:
    if y_train[0] < y_train[-1]:
        trend ='increasing'
    else:
        trend = 'decreasing'

    # if numbers are decreasing: legend to lower left:
    if trend == 'decreasing':
        plt.legend(bbox_to_anchor=(0.02, 0.02), loc='lower left', borderaxespad=0.)

    # else if numbers are increasing: legend to upper right
    else:
        plt.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left', borderaxespad=0.)

    # Display every 7th day and the last day:
    plt.xticks(np.append(t_grid_all[::7], t_grid_all[-1]), rotation=20)

    # Change y-axis lower bound to 0 and extend upper bound of y-axis by a factor of 1.35:
    x1, x2, y1, y2 = plt.axis()

    # plt.axis((x1, x2, 0, y2*1.35))
    # plt.axis((x1, x2, 0, 650))

    plt.show()
