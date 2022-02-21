import datetime
import random
import statistics

import joblib
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from Backend.Data.DataManager.data_access_methods import get_smoothen_cases, get_starting_values, get_model_params
from Backend.Data.DataManager.data_util import Column, date_int_str, compute_end_date_of_validation_period, \
    create_dates_array, print_progress_with_computation_time_estimate
from Backend.Data.DataManager.db_calls import start_pipeline, insert_param_and_start_vals, insert_prediction_vals, \
    get_all_table_data
from Backend.Data.DataManager.matrix_data import get_weekly_intervals_grid, get_predictors_for_ml_layer, \
    prepare_all_beta_predictors
from Backend.Data.DataManager.remote_db_manager import download_db_file
from Backend.Modeling.Differential_Equation_Modeling.prediction_intervals import get_prediction_intervals
from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline, forecast_seirv
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Modeling.Differential_Equation_Modeling.seirv_model_and_ml import seirv_ml_layer
from Backend.Modeling.Util.pipeline_util import train_test_split, get_list_of_random_dates, \
    get_list_of_random_districts, date_difference_strings
from Backend.Modeling.forecast import forecast_all_models, convert_all_forecasts_to_incidences, convert_seven_day_averages
from Backend.Visualization.plotting import plot_train_fitted_and_validation, plot_sarima_pred_plot, \
    plot_sarima_val_line_plot, plot_train_fitted_and_predictions, plot_all_forecasts
from Backend.Modeling.Regression_Model.ARIMA import run_sarima, sarima_model_predictions, sarima_pipeline_val
import copy
from Backend.Modeling.Regression_Model.ARIMA import run_sarima, sarima_model_predictions, sarima_pipeline

import xgboost as xgb
from sklearn.preprocessing import StandardScaler


# from Backend.Modeling.Regression Model.ARIMA import sarima_pipeline


def diff_eq_pipeline(train_end_date: date, duration: int, districts: list, validation_duration: int,
                     visualize=False, verbose=False, validate=True, store_results_to_db=True,
                     with_db_update=False) -> None:
    # iterate over districts(list) of interest
    # results_dict = []

    # retrieve the latest db file from the server
    if with_db_update:
        download_db_file()

    # store pipeline data in the DB
    pipeline_id = start_pipeline(train_end_date, validation_duration, visualize, validate, verbose)

    # set up results dictionary:
    results_dict = {}

    for i, district in enumerate(districts):
        # 1) Import Data
        # 1a) get_smoothed_infection_counts() -> directly from Database

        if validate:
            val_end_date = compute_end_date_of_validation_period(train_end_date, validation_duration)
            smoothen_cases = get_smoothen_cases(district, val_end_date, duration)
            # 1b) split_into_train_and_validation()
            y_train, y_val = train_test_split(data=smoothen_cases[Column.SEVEN_DAY_SMOOTHEN],
                                              validation_duration=validation_duration)
            train_start_date = date_int_str(smoothen_cases[Column.DATE][0])

        else:
            forecast_length = duration - validation_duration
            y_train = get_smoothen_cases(district, train_end_date, forecast_length)
            train_start_date = date_int_str(y_train[Column.DATE][0])
            y_train = y_train[Column.SEVEN_DAY_SMOOTHEN]

        # 1c) get_starting_values() -> N=population, R0=recovered to-the-date, V0=vaccinated to-the-date
        start_vals = get_starting_values(district, train_start_date)
        fixed_model_params = get_model_params(district, train_start_date)

        ## 2) Run model_pipeline
        pipeline_result = seirv_pipeline(y_train=y_train, start_vals_fixed=start_vals,
                                         fixed_model_params=fixed_model_params,
                                         allow_randomness_fixed_beta=False, random_runs=100, district=district)
        y_pred_without_train_period = pipeline_result['y_pred_without_train_period']

        ## 3) Evaluate the results

        # 3a) Visualize results (mainly for debugging)
        if visualize:
            if validate:
                plot_train_fitted_and_validation(y_train=y_train, y_val=y_val,
                                                 y_pred=pipeline_result['y_pred_including_train_period'],
                                                 y_pred_upper=pipeline_result[
                                                     'y_pred_without_train_period_upper_bound'],
                                                 y_pred_lower=pipeline_result[
                                                     'y_pred_without_train_period_lower_bound'])

            else:
                plot_train_fitted_and_predictions(
                    y_train_fitted=pipeline_result['y_pred_including_train_period'][0:duration],
                    y_train_true=y_train,
                    y_pred_full=pipeline_result['y_pred_including_train_period'],
                    district=district,
                    pred_start_date=train_end_date)

        # 3b) Compute metrics (RMSE, MAPE, ...)
        if validate:
            scores = compute_evaluation_metrics(y_pred=y_pred_without_train_period, y_val=y_val)

            # collecting pipeline results to a list to be used in step four
            # results_dict.append({
            #     'district': district,
            #     'pipeline_results': pipeline_result,
            #     'scores': scores,
            # })

        # 4) Store results in database:
        if store_results_to_db:
            insert_param_and_start_vals(pipeline_id, district, start_vals,
                                        pipeline_result['model_params_forecast_period'])
            insert_prediction_vals(pipeline_id, district, pipeline_result['y_pred_without_train_period'],
                                   train_end_date)

        #  5) Append results to dictionary:
        if validate:
            results_dict[district] = {
                'scores': scores
            }

    if validate:
        return results_dict
    else:
        return None


def diff_eq_pipeline_wrapper(**kwargs):
    end_date = '2021-12-14'
    time_frame_train_and_validation = 28
    forecasting_horizon = 14
    districts = ['Essen', 'Münster', 'Herne', 'Bielefeld', 'Dortmund', 'Leipzig_Stadt', 'Berlin']

    SEED = 42

    random_train_end_dates = get_list_of_random_dates(num=20, lower_bound='2020-03-15', upper_bound='2021-12-15',
                                                      seed=SEED)
    random_districts = get_list_of_random_districts(num=10, seed=SEED)

    ####################################################################
    ### Part below is for finding optimal length of training period: ###
    ####################################################################

    duration_param_grid = [7, 10, 14, 18, 21, 28]

    results_level1 = []
    ## Iterate over duration_param_grid:
    for round_lvl1, train_period_length in enumerate(duration_param_grid):

        ## Iterate over train dates:
        # set_up dictionary for storing results at run time:
        results_lvl2 = []
        for round_lvl2, random_train_end_date in enumerate(random_train_end_dates):
            res = diff_eq_pipeline(train_end_date=random_train_end_date,
                                   duration=train_period_length + forecasting_horizon,
                                   districts=random_districts,
                                   validation_duration=forecasting_horizon,
                                   visualize=False,
                                   verbose=False,
                                   validate=True,
                                   store_results_to_db=False)

            # Print progess:
            print(f'Round {1 + round_lvl2 + round_lvl1 * len(random_train_end_dates)} / '
                  f'{len(duration_param_grid) * len(random_train_end_dates)} completed!')

            results_lvl2.append({
                'date': random_train_end_date,
                'pipeline_results': res
            })

        results_level1.append({
            'train_period_length': train_period_length,
            'dict': results_lvl2
        })

    # build dataframe from results:
    results_transformed = []

    for lvl1 in results_level1:
        for lvl2 in lvl1['dict']:
            for city, result in lvl2['pipeline_results'].items():
                results_transformed.append({
                    'train_length': lvl1['train_period_length'],
                    'training_end_date': lvl2['date'],
                    'city': city,
                    'rmse': result['scores']['rmse'],
                    'mape': result['scores']['mape'],
                    'mae': result['scores']['mae'],
                })

    results_df = pd.DataFrame(results_transformed)

    grouped_df = results_df.groupby('train_length')
    temp = grouped_df[['rmse', 'mae', 'mape']].median()

    pass


# SARIMA Model
def sarima_pipeline_old(train_end_date: date, duration: int, districts: list, validation_duration: int,
                    visualize=False, verbose=False, validate=True, evaluate=False, with_db_update=False) -> None:
    if with_db_update:
        download_db_file()

    # iterate over districts(list) of interest
    # results_dict = []
    # store pipeline data in the DB
    # pipeline_id = start_pipeline(train_end_date, validation_duration, visualize, verbose)
    # pipeline_id = start_pipeline(train_end_date, validation_duration, visualize, verbose)
    predictions_list = []

    if evaluate:
        rmse_list = []

    for i, district in enumerate(districts):

        val_end_date = compute_end_date_of_validation_period(train_end_date, validation_duration)
        smoothen_cases = get_smoothen_cases(district, val_end_date, duration)

        y_train = smoothen_cases['seven_day_infec']

        if validate == False:
            #sarima_pipeline_val(y_train, validation_duration)

            format = "%Y-%m-%d"
            train_end_date = datetime.datetime.strptime(train_end_date, format)
            train_end_date = train_end_date - datetime.timedelta(days=validation_duration)
            train_end_date = str(train_end_date)
            train_end_date = train_end_date[:10]

        # 1) Import Data
        # 1a) get_smoothed_infection_counts() -> directly from Database

        # 1b) split_into_train_and_validation()
        y_train, y_val = train_test_split(data=smoothen_cases[Column.SEVEN_DAY_SMOOTHEN],
                                          validation_duration=validation_duration)

        ## 2) Run model_pipeline
        sarima_model = run_sarima(y_train=y_train, y_val=y_val)
        season = sarima_model["season"]
        predictions_val = sarima_model["model"].predict(validation_duration)

        # 2b) Run model without validation data
        if validate == False:
            format = "%Y-%m-%d"
            train_end_date = datetime.datetime.strptime(train_end_date, format)
            train_end_date = train_end_date + datetime.timedelta(days=validation_duration)
            train_end_date = str(train_end_date)
            train_end_date = train_end_date[:10]

            y_train_pred = get_smoothen_cases(district, train_end_date, duration - validation_duration)
            y_train_pred = y_train_pred[Column.SEVEN_DAY_SMOOTHEN]
            sarima_model_without_val = sarima_model_predictions(y_train=y_train_pred, m=season,
                                                                length=validation_duration)
            predictions = sarima_model_without_val.predict(validation_duration)

        # returned:
        # I) sarima_model: season, model
        # II) sarima_model_without_val: model

        ## 3) Evaluate the results

        # 3a) Visualize results (mainly for debugging)
        if visualize:
            if validate:
                plot_sarima_val_line_plot(y_train, y_val, predictions_val, pred_start_date=train_end_date,
                                          district=district)
            else:
                plot_sarima_pred_plot(y_train_pred, predictions, district, pred_start_date=train_end_date)

        # 3b) Compute metrics (RMSE, MAPE, ...)
        if validate:
            scores = compute_evaluation_metrics(y_pred=predictions_val, y_val=y_val)
            # collecting pipeline results to a list to be used in step four
            # results_dict.append({
            #     'district': district,
            #     'pipeline_results': pipeline_result,
            #     'scores': scores,
            # })
        if validate == False:
            predictions_list.append(predictions)
        else:
            predictions_list.append(predictions_val)

        if evaluate:
            rmse_list.append(scores["rmse"])

        # 4) Store results in database:
        # insert_param_and_start_vals(pipeline_id, district, start_vals, pipeline_result['model_params'])
        # insert_prediction_vals(pipeline_id, district, pipeline_result['y_pred_without_train_period'], train_end_date)

    if evaluate:
        return rmse_list

    return predictions_list

    pass

    ## 4a) Meta parameters
    # 1) which model?
    # 2) what period?
    # 3) with what parameters?

    ## 4b) Predictions

    # -> basically all parameters that are set


def model_validation_pipeline_v2_wrapper():
    # Small helper function for determining date shifted by a given number of weeks
    target_date = '2021-11-03'
    num_days_shift = 14
    shifted_date_obj = datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=num_days_shift)
    shifted_date_str = shifted_date_obj.strftime('%Y-%m-%d')
    # print(f'Date 8 weeks after target_date: {shifted_date_str}')

    #################### Pipeline Configuration: ####################
    # # For each interval

    pipeline_intervals = [
        ('2020-11-12', '2022-01-29'),
    ]

    forecasting_horizon = 14

    train_length_diffeqmodel = 14
    train_length_sarima = 42
    training_period_max = max(train_length_diffeqmodel, train_length_sarima)

    opendata = get_all_table_data(table_name='district_list')
    districts = opendata['district'].tolist()

    # Only sample:
    np.random.seed(420)
    # districts = random.sample(districts, 80)

    districts.sort()

    # districts = ['Aachen', 'Hannover', 'Münster', 'Bielefeld']

    ensemble_model_share = {
        'seirv_last_beta': 0,
        'seirv_ml_beta': 0.5,
        'sarima': 0.5
    }

    # ML Layer:
    ml_model_path = '../Assets/MachineLearningLayer/Models/xgb_model_lukas.pkl'
    standardizer_model_path = '../Assets/MachineLearningLayer/Models/standardizer_model.pkl'

    ################################################################

    results_dict = {}
    for num, pipeline_interval in enumerate(pipeline_intervals):
        print(f'################## Starting Model Validation Pipeline for pipeline interval {num+1}/{len(pipeline_intervals)} '
              f'for time interval: {pipeline_interval[0]} until {pipeline_interval[1]} ################## ')

        results_dict[num] = (
            model_validation_pipeline_v2(pipeline_start_date=pipeline_interval[0], pipeline_end_date=pipeline_interval[1],
                                         forecasting_horizon=forecasting_horizon,
                                         train_length_diffeqmodel=train_length_diffeqmodel,
                                         train_length_sarima=train_length_sarima, training_period_max=training_period_max,
                                         ml_model_path=ml_model_path, standardizer_model_path=standardizer_model_path,
                                         ensemble_model_share=ensemble_model_share,
                                         districts=districts)
        )


    ### Prepare results for exporting them as a dataframe:
    # Very complicated unpacking of multiple levels deep dictionary:

    # Create Lvl1-DataFrame:
    ## Unbox dictionary:

    unpacked_1 = []

    for pipeline_num, v1 in results_dict.items():
        for district, v2 in v1.items():
            for week, v3 in enumerate(v2):
                unpacked_1.append({
                    'pipeline_num': pipeline_num,
                    'district': district,
                    'week': week,

                    # unpack everything on level 3:
                    **v3
                })


    unpacked_2 = copy.deepcopy(unpacked_1)


    # Delete all time series data and keep only the rest:
    time_series_keys = ['y_train_sarima', 'y_train_diffeq', 'y_val', 'y_pred', 'residuals',
                        'dates_training_diffeq', 'dates_training_sarima', 'dates_validation', 'dates_full']

    for item in unpacked_2:
        for k in list(item.keys()):
            if k in time_series_keys:
                del item[k]

    unpacked_3 = []
    for idx, item in enumerate(unpacked_2):
        temp_dict = {}
        for k, v in item.items():
            if not k == 'metrics':
                temp_dict[k] = v
            else:
                for model, metrics in v.items():
                    for metric_name, metric_value in metrics.items():
                        temp_dict[model+'-'+metric_name] = metric_value
        temp_dict['idx'] = idx
        unpacked_3.append(temp_dict)

    # Create Tier1 DataFrame:
    df_lvl1 = pd.DataFrame(unpacked_3)

    # Day-specific data:
    unpacked_2b = copy.deepcopy(unpacked_1)
    for item in unpacked_2b:
        for k in list(item.keys()):
            if k not in time_series_keys:
                del item[k]

    # unpack y_pred and residuals:
    unpacked_3b = []
    for idx, item in enumerate(unpacked_2b):
        temp_dict = {}
        for k, v in item.items():
            if k not in ['y_pred', 'residuals']:
                temp_dict[k] = v
            else:
                for model, values in v.items():
                    temp_dict[f'{k}_{model}'] = values
        temp_dict['idx'] = idx
        unpacked_3b.append(temp_dict)

    # Delete non forecasting related keys:
    del_keys = ['y_train_sarima', 'y_train_diffeq', 'dates_training_diffeq', 'dates_training_sarima', 'dates_full']
    for item in unpacked_3b:
        for k in list(item.keys()):
            if k in del_keys:
                del item[k]

    ## Create Tier2 DataFrame:
    df_list = []
    for data_interval in unpacked_3b:
        df_list.append(
            pd.DataFrame(data_interval)
        )

    df_lvl2 = pd.concat(df_list, axis=0)


    #### Export both DataFrames for further analysis as CSVs:

    # Run-Information (including metrics)
    df_lvl1.to_csv(
        path_or_buf=f'../Assets/Data/Evaluation/model_validation_data_metrics_{datetime.now().strftime("%d_%m_%H:%M")}.csv')

    # Estimates
    df_lvl2.to_csv(
        path_or_buf=f'../Assets/Data/Evaluation/model_validation_data_forecasts_{datetime.now().strftime("%d_%m_%H:%M")}.csv')





def model_validation_pipeline_v2(pipeline_start_date, pipeline_end_date, forecasting_horizon, train_length_diffeqmodel,
                                 train_length_sarima, training_period_max,
                                 ml_model_path, standardizer_model_path, ensemble_model_share, districts,
                                 run_diff_eq_last_beta=True,
                                 run_diff_eq_ml_beta=True,
                                 run_sarima=True,
                                 run_ensemble=True,
                                 debug=False):
    # Create time_grid:
    intervals_grid = get_weekly_intervals_grid(pipeline_start_date, pipeline_end_date, training_period_max,
                                               forecasting_horizon)

    # Compute training duration:
    duration_full = date_difference_strings(d1=pipeline_start_date, d2=pipeline_end_date)

    # Import ML-Model:
    if run_diff_eq_ml_beta:
        with open(ml_model_path, 'rb') as fid:
            ml_model = joblib.load(fid)

        # Import Standardizer:
        with open(standardizer_model_path, 'rb') as fid:
            standardizer_obj = joblib.load(fid)
    else:
        ml_model = None
        standardizer_obj = None

    # Import Prediction Intervals:
    pred_intervals_df = get_prediction_intervals()

    # Iterate over districts:
    results_dict = {}
    start_time = datetime.now()
    for i, district in enumerate(districts):
        print_progress_with_computation_time_estimate(completed=i + 1, total=len(districts), start_time=start_time)

        ## 2a) Import Historical Infections
        # Import Training Data:

        infections_df = get_smoothen_cases(district=district, duration=duration_full + 1, end_date=pipeline_end_date)
        # append one column for formatted dates:
        infections_df['date_str'] = infections_df[Column.DATE].apply(
            lambda row: datetime.strptime(row, '%Y%m%d').strftime('%Y-%m-%d'))

        # Iterate over weeks:
        weekly_results = []
        for week_num, current_interval in enumerate(intervals_grid):
            ### 2) Import Training Data
            ## 2a) Import Historical Infections

            # Training indices:
            idx_train_start_sarima = infections_df.loc[infections_df['date_str'] == current_interval['start_day_train_str']].index[0]
            idx_train_start_diffeq = idx_train_start_sarima + (train_length_sarima - train_length_diffeqmodel)
            idx_train_end = infections_df.loc[infections_df['date_str'] == current_interval['end_day_train_str']].index[0]

            # Validation indices:
            idx_val_start = infections_df.loc[infections_df['date_str'] == current_interval['start_day_val_str']].index[0]
            idx_val_end = infections_df.loc[infections_df['date_str'] == current_interval['end_day_val_str']].index[0]

            # Get arrays from dataframe:
            y_train_sarima = infections_df.loc[idx_train_start_sarima:idx_train_end,
                             Column.SEVEN_DAY_SMOOTHEN].reset_index(drop=True)
            y_train_diffeq = infections_df.loc[idx_train_start_diffeq:idx_train_end,
                             Column.SEVEN_DAY_SMOOTHEN].reset_index(drop=True)
            y_val = infections_df.loc[idx_val_start:idx_val_end, Column.SEVEN_DAY_SMOOTHEN].reset_index(drop=True)

            ## 2b) Get Starting Values for SEIRV Model:
            train_start_date_diffeq_obj = current_interval['start_day_train_obj'] + timedelta(days=14)
            train_start_date_diff_eq_str = train_start_date_diffeq_obj.strftime('%Y-%m-%d')

            if run_diff_eq_last_beta or run_diff_eq_ml_beta:
                start_vals_seirv = get_starting_values(district, train_start_date_diff_eq_str)
                fixed_model_params_seirv = get_model_params(district, train_start_date_diff_eq_str)
            else:
                start_vals_seirv = None
                fixed_model_params_seirv = None

            ## 2c) Import Data for Machine Learning Matrix:
            if run_diff_eq_ml_beta:
                ml_matrix_predictors = get_predictors_for_ml_layer(district, train_start_date_diff_eq_str)
            else:
                ml_matrix_predictors = None

            ### 3) Models
            seirv_last_beta_only_results, seirv_ml_results, sarima_results, ensemble_results, all_combined = \
                forecast_all_models(y_train_diffeq, y_train_sarima, forecasting_horizon,
                                    ml_matrix_predictors, start_vals_seirv, fixed_model_params_seirv,
                                    standardizer_obj, ml_model, district, ensemble_model_share,
                                    pred_intervals_df,
                                    run_diff_eq_last_beta, run_diff_eq_ml_beta, run_sarima, run_ensemble,
                                    )
            ## 3a) Try to catch bad ARIMA results
            if sarima_results['predictions'][0] == sarima_results['predictions'][1]:
                y_train_short = y_train_sarima.loc[14:idx_train_end]
                sarima_results = sarima_pipeline(y_train=y_train_short, forecasting_horizon=forecasting_horizon)
                all_combined['y_pred_sarima_mean'] = sarima_results['predictions']
                all_combined['y_pred_sarima_upper'] = sarima_results['upper']
                all_combined['y_pred_sarima_lower'] = sarima_results['lower']


            # Combine results:
            y_pred = {
                'Diff_Eq_Last_Beta': seirv_last_beta_only_results['y_pred_without_train_period'],
                'Diff_Eq_ML_Beta': seirv_ml_results['y_pred_mean'],
                'Sarima': sarima_results['predictions'],
                'Ensemble': ensemble_results['y_pred_mean'],
            }

            residuals = {
                'Diff_Eq_Last_Beta': y_pred['Diff_Eq_Last_Beta'] - y_val,
                'Diff_Eq_ML_Beta': y_pred['Diff_Eq_ML_Beta'] - y_val,
                'Sarima': y_pred['Sarima'] - y_val,
                'Ensemble': y_pred['Ensemble'] - y_val,
            }

            ## 4) Convert 7-day average to 7-day-incident:
            all_combined_incidence = convert_all_forecasts_to_incidences(all_combined,
                                                                         start_vals_seirv['N'])
            y_train_sarima_incidence = convert_seven_day_averages(y_train_sarima, start_vals_seirv['N'])
            y_train_diffeq_incidence = convert_seven_day_averages(y_train_diffeq, start_vals_seirv['N'])
            y_val_incidence = convert_seven_day_averages(y_val, start_vals_seirv['N'])

            y_pred_incidence = {
                'Diff_Eq_Last_Beta': all_combined_incidence['y_pred_seirv_last_beta_mean'],
                'Diff_Eq_ML_Beta': all_combined_incidence['y_pred_seirv_ml_beta_mean'],
                'Sarima': all_combined_incidence['y_pred_sarima_mean'],
                'Ensemble': all_combined_incidence['y_pred_ensemble_mean'],
            }


            ## 5) Visualization:
            if debug:
                plot_all_forecasts(forecast_dictionary=all_combined_incidence, y_train=y_train_diffeq_incidence,
                                   start_date_str=current_interval['start_day_train_str'],
                                   forecasting_horizon=forecasting_horizon,
                                   district=district,
                                   y_val=y_val_incidence,
                                   y_train_fitted=convert_seven_day_averages(seirv_last_beta_only_results['y_pred_including_train_period'],start_vals_seirv['N']),
                                   plot_y_train_fitted=False,
                                   plot_y_train_fitted_all=False,
                                   plot_val=False,
                                   plot_diff_eq_last_beta=True,
                                   plot_diff_eq_ml_beta=True,
                                   plot_sarima=True,
                                   plot_ensemble=True,
                                   plot_predictions_intervals=False
                                   )

                # Train + VAL - SEIRV
                # plot_train_fitted_and_validation(y_train=y_train_diffeq,
                #                                  y_pred=seirv_last_beta_only_results['y_pred_including_train_period'],
                #                                  y_val=y_val)

                #Train + VAL - SARima
               # plot_sarima_val_line_plot(train_array=y_train_sarima, test_array=y_val,
                #                          predictions=sarima_results["predictions"],
                 #                         pred_start_date=current_interval['start_day_val_str'], district=district)


            ## 6) Evaluation - Compute metrics:
            metrics = {
                'Diff_Eq_Last_Beta': compute_evaluation_metrics(y_pred=y_pred['Diff_Eq_Last_Beta'], y_val=y_val),
                'Diff_Eq_ML_Beta': compute_evaluation_metrics(y_pred=y_pred['Diff_Eq_ML_Beta'], y_val=y_val),
                'Sarima': compute_evaluation_metrics(y_pred=y_pred['Sarima'], y_val=y_val),
                'Ensemble': compute_evaluation_metrics(y_pred=y_pred['Ensemble'], y_val=y_val),
            }

            ## 7) Append everything to result list:
            weekly_results.append({
                'week_num': week_num,
                'year_start_forecast': current_interval['start_day_val_obj'].isocalendar()[0],
                'calendar_week_start_forecast': current_interval['start_day_val_obj'].isocalendar()[1],
                'start_day_train_sarima': current_interval['start_day_train_str'],
                'start_day_train_diffeq': train_start_date_diff_eq_str,
                'end_day_train': current_interval['end_day_train_str'],
                'start_day_val': current_interval['start_day_val_str'],
                'end_day_val': current_interval['end_day_val_str'],
                'y_train_sarima': y_train_sarima_incidence,
                'y_train_diffeq': y_train_diffeq_incidence,
                'y_val': y_val_incidence,
                'y_pred': y_pred_incidence,
                'mean_y_train': np.mean(y_train_diffeq_incidence),
                'mean_y_val': np.mean(y_val_incidence),
                'residuals': residuals,
                'metrics': metrics,
                'dates_training_diffeq': create_dates_array(start_date_str=train_start_date_diff_eq_str, num_days=train_length_diffeqmodel),
                'dates_training_sarima': create_dates_array(start_date_str=current_interval['start_day_train_str'], num_days=train_length_sarima),
                'dates_validation': create_dates_array(start_date_str=current_interval['start_day_val_str'], num_days=forecasting_horizon),
                'dates_full': create_dates_array(start_date_str=current_interval['start_day_train_str'], num_days=train_length_sarima+forecasting_horizon),
            })

        # Append everything to results dict:
        results_dict[district] = weekly_results

    return results_dict


if __name__ == '__main__':
    # smoothen_cases = get_smoothen_cases('Münster', '2021-03-31', 40)
    # y_train, y_val = train_test_split(smoothen_cases)
    # train_start_date = date_int_str(y_train[Column.DATE][0])
    # start_vals = get_starting_values('Münster', train_start_date)
    # print(start_vals)

    model_validation_pipeline_v2_wrapper()
