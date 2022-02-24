import datetime

import joblib
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from Backend.Data.DataManager.data_access_methods import get_smoothen_cases, get_starting_values, get_model_params
from Backend.Data.DataManager.data_util import Column, date_int_str, compute_end_date_of_validation_period, \
    create_dates_array, print_progress_with_computation_time_estimate
from Backend.Data.DataManager.db_calls import start_validation_pipeline, insert_param_and_start_vals, insert_forecast_vals, \
    get_all_table_data
from Backend.Data.DataManager.matrix_data import get_weekly_intervals_grid, get_predictors_for_ml_layer
from Backend.Data.DataManager.remote_file_manager import download_db_file
from Backend.Modeling.Differential_Equation_Modeling.prediction_intervals import get_prediction_intervals
from Backend.Modeling.Differential_Equation_Modeling.seiurv_model import seiurv_pipeline
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Modeling.Util.pipeline_util import train_test_split, get_list_of_random_dates, \
    get_list_of_random_districts, date_difference_strings
from Backend.Modeling.forecasting_wrapper_functions import forecast_all_models, convert_all_forecasts_to_incidences, convert_seven_day_averages
from Backend.Visualization.plotting import plot_train_fitted_and_validation, plot_train_fitted_and_predictions, plot_all_forecasts
import copy


# from Backend.Modeling.Regression Model.ARIMA import sarima_pipeline


def diff_eq_pipeline_DEPRECATED(train_end_date: date, duration: int, districts: list, validation_duration: int,
                                visualize=False, verbose=False, validate=True, store_results_to_db=True,
                                with_db_update=False) -> None:
    # iterate over districts(list) of interest
    # results_dict = []

    # retrieve the latest db file from the server
    if with_db_update:
        download_db_file()

    # store pipeline data in the DB
    pipeline_id = start_validation_pipeline(train_end_date, validation_duration, visualize, validate, verbose)

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
        pipeline_result = seiurv_pipeline(y_train=y_train, start_vals_fixed=start_vals,
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
            insert_forecast_vals(pipeline_id, district, pipeline_result['y_pred_without_train_period'],
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


def diff_eq_pipeline_wrapper_DEPRECATED(**kwargs):
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
            res = diff_eq_pipeline_DEPRECATED(train_end_date=random_train_end_date,
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


def model_validation_pipeline_v2_wrapper():
    """
    This function is wrapped around the model_validation_pipeline and therefore has the same name + wrapper.
    Here
    """

    ########################## Optional:  ##########################
    # Small helper function for determining date shifted by a given number of weeks which might be useful when
    # setting up the dates in the pipeline configuration part below:
    target_date = '2021-11-03'
    num_days_shift = 14
    shifted_date_obj = datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=num_days_shift)
    shifted_date_str = shifted_date_obj.strftime('%Y-%m-%d')
    print(f'Date 8 weeks after target_date: {shifted_date_str}')
    #################################################################


    #################### Pipeline Configuration: ####################

    # Multiple intervals over which the pipeline is supposed to be run can be setup here.
    # For each interval the model validation pipeline is called. This is usually only done once, unless one
    # wants to run two unconnected time intervals. (Example: Run model for Apr 2020 - Oct 2020 + Jun 2021 - Jan 2022)

    pipeline_intervals = [
        ('2021-01-01', '2022-01-28'),
    ]

    # Number of days to be forecasted:
    forecasting_horizon = 14

    # Number of days prior to forecasting period that the models are supposed to be trained on:
    train_length_diffeqmodel = 14
    train_length_sarima = 42
    training_period_max = max(train_length_diffeqmodel, train_length_sarima)

    # Get List of Districts:
    opendata = get_all_table_data(table_name='district_list')
    districts = opendata['district'].tolist()
    districts.sort()

    ################################ Optional: ################################
    ### Option 1: District Sample:
    ## Uncomment the lines below if only a sample of districts should be used.
    # Only sample:
    # np.random.seed(420)
    # num_districts_sample = 80
    # districts = random.sample(districts, num_districts_sample)
    # districts.sort()
    #
    ### Option 2: Set districts manually:
    # districts = ['Münster', 'Bielefeld']
    ##############################################################################

    # The ensemble model is computed as a weighted average of the other three models. The weights can be set here:
    ensemble_model_share = {
        'seirv_last_beta': 0.2,
        'seirv_ml_beta': 0.4,
        'sarima': 0.4
    }

    ### ML Layer:
    # For the machine learning layer the XGBoost Model provided by the beta estimation machine learning jupyter notebook
    # (Backend/Modeling/Differential_Equation_Modeling/Machine_Learning_Layer/beta_estimation) has to be imported
    # so that the model can be used for producing forecasts for our ML_Beta DiffEq model. As the model needs the
    # predictors to be standardized the standardizer model also has to be imported:

    # Both Models will only be created by running the above described jupyter notebook.
    ml_model_path = '../Assets/MachineLearningLayer/Models/xgb_model_lukas.pkl'
    standardizer_model_path = '../Assets/MachineLearningLayer/Models/standardizer_model.pkl'

    ################################################################

    # For each Pipeline Interval the Model Validation pipeline is called. This is usually only done once, unless one
    # wants to run two unconnected time intervals. (Example: Run model for Apr 2020 - Oct 2020 + Jun 2021 - Jan 2022)

    # The results are saved in a dictionary which is instantiated below:
    results_dict = {}
    for num, pipeline_interval in enumerate(pipeline_intervals):
        print(f'################## Starting Model Validation Pipeline for pipeline interval {num+1}/{len(pipeline_intervals)} '
              f'for time interval: {pipeline_interval[0]} until {pipeline_interval[1]} ################## ')

        # As described above, the results are saved in the results_dict dictionary:
        results_dict[num] = (
            model_validation_pipeline_v2(pipeline_start_date=pipeline_interval[0], pipeline_end_date=pipeline_interval[1],
                                         forecasting_horizon=forecasting_horizon,
                                         train_length_diffeqmodel=train_length_diffeqmodel,
                                         train_length_sarima=train_length_sarima, training_period_max=training_period_max,
                                         ml_model_path=ml_model_path, standardizer_model_path=standardizer_model_path,
                                         ensemble_model_share=ensemble_model_share,
                                         districts=districts)
        )


    ######################## Prepare results for exporting them as a dataframe: ########################
    # Below the results are unpacked to produce two dataframes which can be used for further analyses.
    # However, as this requires a deeply nested dictionary to be unpacked to code below is somewhat complicated.
    # Unless you want to dive into the depths of nested python dictionaries we suggest skipping this part and
    # continuing at the bottom ;-D

    #### Unpacking :
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

    # The Forecasts Dataframe (df_lvl1) contains detailed information regarding the forecasts of the different
    # models and the correct data (validation data). Using the idx column it this table is connected
    # to the Metrics Dataframe (df_lvl2) that evaluates the different approaches over each forecasting horizon
    # and district combination.

    # Run-Information (including metrics)
    df_lvl1.to_csv(
        path_or_buf=f'../Assets/Data/Evaluation/model_validation_data_metrics_{datetime.now().strftime("%d_%m_%H:%M")}.csv')

    # Forecast for each date/district combination for each forecasting interval set at the top of the wrapper function.
    df_lvl2.to_csv(
        path_or_buf=f'../Assets/Data/Evaluation/model_validation_data_forecasts_{datetime.now().strftime("%d_%m_%H:%M")}.csv')





def model_validation_pipeline_v2(pipeline_start_date, pipeline_end_date, forecasting_horizon, train_length_diffeqmodel,
                                 train_length_sarima, training_period_max,
                                 ml_model_path, standardizer_model_path, ensemble_model_share, districts,
                                 run_diff_eq_last_beta=True,
                                 run_diff_eq_ml_beta=True,
                                 run_sarima=True,
                                 run_ensemble=True,
                                 debug=True):

    # 1) Preparation steps:

    ## Create time_grid:
    # Intervals_grid is a list of dictionaries. The provided time interval is split into smaller intervals which
    # are eventually used for training and then validating our models. To explain this better one example:
    # Let's assume that the pipeline is supposed to be run from Calendar Week (CW) 1 - 40 in 2020.
    # Given a two week forecasting horizon, as well as a 4 week training horizon (at least for the Arima model) this
    # means that the intervals in which one pipeline run is executed are the following:
    # 1.    Training:   CW 1 - 4    Forecasting / Validation:   CW 5 - 6
    # 2.    Training:   CW 2 - 5    Forecasting / Validation:   CW 6 - 7
    #         ...
    # 35.   Training:   CW 35 - 38  Forecasting / Validation:   CW 39 - 40
    #
    # => Each interval grid is shifted one week ahead.
    intervals_grid = get_weekly_intervals_grid(pipeline_start_date, pipeline_end_date, training_period_max,
                                               forecasting_horizon)

    # Compute training duration (in days):
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

    ## Import Prediction Intervals:
    # These intervals are used for computing the prediction intervals for forecasting. Using them we can provide
    # prediction intervals for both differential equation models. The models are computed in the evaluation_pipeline.R
    # script. Here more information can be found regarding how exactly the values are computed.
    pred_intervals_df = get_prediction_intervals()

    ## Iterate over districts:
    # Instantiate dictionary for storing results:
    results_dict = {}
    # Save start time for printing the estimated time until when the function is finished.
    start_time = datetime.now()
    for i, district in enumerate(districts):
        print_progress_with_computation_time_estimate(completed=i + 1, total=len(districts), start_time=start_time)

        ## 2) Import Training Data
        # 2a) Import Historical Infections for the current district:
        infections_df = get_smoothen_cases(district=district, duration=duration_full + 1, end_date=pipeline_end_date)

        # Append one column for formatted dates:
        infections_df['date_str'] = infections_df[Column.DATE].apply(
            lambda row: datetime.strptime(row, '%Y%m%d').strftime('%Y-%m-%d'))

        # Iterate over the weekly intervals set-up above in step 1:
        weekly_results = []
        for week_num, current_interval in enumerate(intervals_grid):

            ## Compute indices so that the correct training and validation data is used:
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

            ## 2b) Get Starting Values for SEIRV Model compartments:
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
            # Get the predictions for all 4 models (at least if all flags are set to true):
            seirv_last_beta_only_results, seirv_ml_results, sarima_results, ensemble_results, all_combined = \
                forecast_all_models(y_train_diffeq, y_train_sarima, forecasting_horizon,
                                    ml_matrix_predictors, start_vals_seirv, fixed_model_params_seirv,
                                    standardizer_obj, ml_model, district, ensemble_model_share,
                                    pred_intervals_df,
                                    run_diff_eq_last_beta, run_diff_eq_ml_beta, run_sarima, run_ensemble,
                                    )

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
                                   plot_val=True,
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
    model_validation_pipeline_v2_wrapper()
