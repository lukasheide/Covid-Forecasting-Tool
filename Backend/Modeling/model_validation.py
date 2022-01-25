import datetime
import pandas as pd
from datetime import date, datetime, timedelta
from Backend.Data.DataManager.data_access_methods import get_smoothen_cases, get_starting_values, get_model_params
from Backend.Data.DataManager.data_util import Column, date_int_str, compute_end_date_of_validation_period
from Backend.Data.DataManager.db_calls import start_pipeline, insert_param_and_start_vals, insert_prediction_vals, \
    get_all_table_data
from Backend.Data.DataManager.matrix_data import get_weekly_intervals_grid, get_predictors_for_ml_layer
from Backend.Data.DataManager.remote_db_manager import download_db_file
from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Modeling.Util.pipeline_util import train_test_split, get_list_of_random_dates, \
    get_list_of_random_districts, date_difference_strings
from Backend.Visualization.modeling_results import plot_train_fitted_and_validation, plot_sarima_pred_plot, \
    plot_sarima_val_line_plot, plot_train_fitted_and_predictions, visualize_multiple_models
from Backend.Modeling.Regression_Model.ARIMA import run_sarima, sarima_model_predictions


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
        pipeline_result = seirv_pipeline(y_train=y_train, start_vals_fixed=start_vals, fixed_model_params=fixed_model_params,
                                         allow_randomness_fixed_beta=False, random_runs=100, district=district)
        y_pred_without_train_period = pipeline_result['y_pred_without_train_period']

        # Run Sarima model
        # sarima_result = sarima_pipeline(y_train, y_val)

        # returned:
        # I) y_pred for both training and validation period,
        # II) Model Params (greek letters)
        # III) Model Starting values (number of people for each compartment at time t)

        ## 3) Evaluate the results

        # 3a) Visualize results (mainly for debugging)
        if visualize:
            if validate:
                plot_train_fitted_and_validation(y_train=y_train, y_val=y_val,
                                                 y_pred=pipeline_result['y_pred_including_train_period'],
                                                 y_pred_upper=pipeline_result['y_pred_without_train_period_upper_bound'],
                                                 y_pred_lower=pipeline_result['y_pred_without_train_period_lower_bound'])

            else:
                plot_train_fitted_and_predictions(y_train_fitted=pipeline_result['y_pred_including_train_period'][0:duration],
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
            insert_param_and_start_vals(pipeline_id, district, start_vals, pipeline_result['model_params_forecast_period'])
            insert_prediction_vals(pipeline_id, district, pipeline_result['y_pred_without_train_period'], train_end_date)

        #  5) Append results to dictionary:
        if validate:
            results_dict[district] = {
                'scores':scores
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

    random_train_end_dates = get_list_of_random_dates(num=20, lower_bound='2020-03-15', upper_bound='2021-12-15', seed=SEED)
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
        for round_lvl2,random_train_end_date in enumerate(random_train_end_dates):
            res = diff_eq_pipeline(train_end_date=random_train_end_date,
                                   duration=train_period_length+forecasting_horizon,
                                   districts=random_districts,
                                   validation_duration=forecasting_horizon,
                                   visualize=False,
                                   verbose=False,
                                   validate=True,
                                   store_results_to_db=False)

            # Print progess:
            print(f'Round {1+round_lvl2+round_lvl1*len(random_train_end_dates)} / '
                  f'{len(duration_param_grid) * len(random_train_end_dates)} completed!')

            results_lvl2.append({
                'date': random_train_end_date,
                'pipeline_results':res
            })

        results_level1.append({
            'train_period_length':train_period_length,
            'dict':results_lvl2
        })

    # build dataframe from results:
    results_transformed = []

    for lvl1 in results_level1:
        for lvl2 in lvl1['dict']:
            for city, result in lvl2['pipeline_results'].items():
                    results_transformed.append({
                        'train_length':lvl1['train_period_length'],
                        'training_end_date':lvl2['date'],
                        'city':city,
                        'rmse':result['scores']['rmse'],
                        'mape':result['scores']['mape'],
                        'mae':result['scores']['mae'],
                    })

    results_df = pd.DataFrame(results_transformed)

    grouped_df = results_df.groupby('train_length')
    temp = grouped_df[['rmse', 'mae', 'mape']].median()

    pass

#SARIMA Model
def sarima_pipeline(train_end_date: date, duration: int, districts: list, validation_duration: int,
                     visualize=False, verbose=False, validate=True, evaluate=False, with_db_update=False) -> None:
    if with_db_update:
        download_db_file()

    # iterate over districts(list) of interest
    # results_dict = []
    # store pipeline data in the DB
    #pipeline_id = start_pipeline(train_end_date, validation_duration, visualize, verbose)
    # pipeline_id = start_pipeline(train_end_date, validation_duration, visualize, verbose)
    predictions_list = []

    if evaluate:
        rmse_list = []

    for i, district in enumerate(districts):

        if validate == False:
            format = "%Y-%m-%d"
            train_end_date = datetime.datetime.strptime(train_end_date, format)
            train_end_date = train_end_date - datetime.timedelta(days=validation_duration)
            train_end_date = str(train_end_date)
            train_end_date = train_end_date[:10]

        # 1) Import Data
        # 1a) get_smoothed_infection_counts() -> directly from Database
        val_end_date = compute_end_date_of_validation_period(train_end_date, validation_duration)
        smoothen_cases = get_smoothen_cases(district, val_end_date, duration)

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
                plot_sarima_val_line_plot(y_train, y_val, predictions_val, pred_start_date=train_end_date, district=district)
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



def model_validation_pipeline_v2():

    #################### Pipeline Configuration: ####################
    pipeline_start_date = '2021-11-01'
    pipeline_end_date = '2022-01-15'
    forecasting_horizon = 14

    train_length_diffeqmodel = 14
    train_length_sarima = 28
    training_period_max = max(train_length_diffeqmodel, train_length_sarima)

    debug = True

    opendata = get_all_table_data(table_name='district_list')
    districts = opendata['district'].tolist()
    districts.sort()

    districts = ['Bielefeld', 'Münster']

    ################################################################


    # Create time_grid:
    intervals_grid = get_weekly_intervals_grid(pipeline_start_date, pipeline_end_date, training_period_max, forecasting_horizon)

    # Compute training duration:
    duration_full = date_difference_strings(d1=pipeline_start_date, d2=pipeline_end_date)


    # Iterate over districts:
    results_dict = {}
    for i, district in enumerate(districts):
        print(f'Computing district {district}: {i+1} / {len(districts)}')

        ## 2a) Import Historical Infections
        # Import Training Data:

        infections_df = get_smoothen_cases(district=district, duration=duration_full+1, end_date=pipeline_end_date)
        # append one column for formatted dates:
        infections_df['date_str'] = infections_df[Column.DATE].apply(lambda row: datetime.strptime(row, '%Y%m%d').strftime('%Y-%m-%d'))

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
            y_train_sarima = infections_df.loc[idx_train_start_sarima:idx_train_end, Column.SEVEN_DAY_SMOOTHEN].reset_index(drop=True)
            y_train_diffeq = infections_df.loc[idx_train_start_diffeq:idx_train_end, Column.SEVEN_DAY_SMOOTHEN].reset_index(drop=True)
            y_val = infections_df.loc[idx_val_start:idx_val_end, Column.SEVEN_DAY_SMOOTHEN].reset_index(drop=True)

            ## 2b) Get Starting Values for SEIRV Model:
            train_start_date_diffeq_obj = current_interval['start_day_train_obj'] + timedelta(days=14)
            train_start_date_diff_eq_str = train_start_date_diffeq_obj.strftime('%Y-%m-%d')

            start_vals_seirv = get_starting_values(district, train_start_date_diff_eq_str)
            fixed_model_params_seirv = get_model_params(district, train_start_date_diff_eq_str)

            ## 2c) Import Data for Machine Learning Matrix:
            ml_training_data = get_predictors_for_ml_layer(district, train_start_date_diff_eq_str)


            ### 3) Models

            ## 3.1) SEIRV + Last Beta
            seirv_beta_results = seirv_pipeline(y_train=y_train_diffeq,
                                                start_vals_fixed=start_vals_seirv,
                                                fixed_model_params=fixed_model_params_seirv,
                                                forecast_horizon=forecasting_horizon,
                                                allow_randomness_fixed_beta=False, district=district)

            ## 3.2) SEIRV + Machine Learning Layer


            ## 3.3) SARIMA


            ## 3.4) Ensemble Model


            # Combine results:
            y_pred = {
                'Diff_Eq_Last_Beta': seirv_beta_results['y_pred_without_train_period'],
            }

            ## 4) Visualization:
            if debug:
                visualize_multiple_models(y_train=y_train_diffeq,
                                          y_pred_full_diffeq=seirv_beta_results['y_pred_including_train_period'],
                                          y_forecast_diffeq=y_pred['Diff_Eq_Last_Beta'],
                                          y_forecast_sarima=None)

            ## 5) Compute metrics:
            metrics = {
                'Diff_Eq_Last_Beta': compute_evaluation_metrics(y_pred=y_pred['Diff_Eq_Last_Beta'], y_val=y_val),
            }


            ## 6) Append everything to result list:
            weekly_results.append({
                'week_num': week_num,
                'year_start_forecast': current_interval['start_day_val_obj'].isocalendar()[0],
                'calendar_week_start_forecast': current_interval['start_day_val_obj'].isocalendar()[1],
                'start_day_train_sarima': current_interval['start_day_train_str'],
                'start_day_train_diffeq': train_start_date_diff_eq_str,
                'end_day_train': current_interval['end_day_train_str'],
                'start_day_val': current_interval['start_day_val_str'],
                'end_day_val': current_interval['end_day_val_str'],
                'y_train_sarima': y_train_sarima,
                'y_train_diffeq': y_train_diffeq,
                'y_val': y_val,
                'y_pred': y_pred,
                'metrics': metrics
            })

        # Append everything to results dict:
        results_dict[district] = weekly_results

    pass


if __name__ == '__main__':
    # smoothen_cases = get_smoothen_cases('Münster', '2021-03-31', 40)
    # y_train, y_val = train_test_split(smoothen_cases)
    # train_start_date = date_int_str(y_train[Column.DATE][0])
    # start_vals = get_starting_values('Münster', train_start_date)
    # print(start_vals)

    model_validation_pipeline_v2()