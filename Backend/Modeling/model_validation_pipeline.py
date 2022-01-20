import pandas as pd
from datetime import date

from Backend.Data.DataManager.data_access_methods import get_smoothen_cases, get_starting_values, get_model_params
from Backend.Data.data_util import Column, date_int_str, compute_end_date_of_validation_period
from Backend.Data.DataManager.db_calls import start_pipeline, insert_param_and_start_vals, insert_prediction_vals
from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline
from Backend.Evaluation.metrics import compute_evaluation_metrics
from Backend.Modeling.Util.pipeline_util import train_test_split, get_list_of_random_dates, get_list_of_random_districts
from Backend.Visualization.modeling_results import plot_train_fitted_and_validation, plot_sarima_pred_plot, \
    plot_sarima_val_line_plot
from Backend.Modeling.Regression_Model.ARIMA import run_sarima, sarima_model_predictions

# from Backend.Modeling.Regression Model.ARIMA import sarima_pipeline


def diff_eq_pipeline(train_end_date: date, duration: int, districts: list, validation_duration: int,
                     visualize=False, verbose=False, validate=True, store_results_to_db=True) -> None:
    # iterate over districts(list) of interest
    # results_dict = []
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
            y_train, y_val = train_test_split(data=smoothen_cases[Column.SEVEN_DAY_SMOOTHEN.value],
                                              validation_duration=validation_duration)
            train_start_date = date_int_str(smoothen_cases[Column.DATE.value][0])

        else:
            y_train = get_smoothen_cases(district, train_end_date, duration)
            train_start_date = date_int_str(y_train[Column.DATE.value][0])
            y_train = y_train[Column.SEVEN_DAY_SMOOTHEN.value]

        # 1c) get_starting_values() -> N=population, R0=recovered to-the-date, V0=vaccinated to-the-date
        start_vals = get_starting_values(district, train_start_date)
        fixed_model_params = get_model_params(district, train_start_date)

        ## 2) Run model_pipeline
        pipeline_result = seirv_pipeline(y_train=y_train, start_vals_fixed=start_vals, fixed_model_params=fixed_model_params,
                                         allow_randomness_fixed_beta=False, random_runs=100)
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
            plot_train_fitted_and_validation(y_train=y_train, y_val=y_val,
                                             y_pred=pipeline_result['y_pred_including_train_period'],
                                             y_pred_upper=pipeline_result['y_pred_without_train_period_upper_bound'],
                                             y_pred_lower=pipeline_result['y_pred_without_train_period_lower_bound'])

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
        results_dict[district] = {
            'scores':scores
        }

    return results_dict



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
                     visualize=False, verbose=False, validate=True, evaluate=False) -> None:
    # iterate over districts(list) of interest
    # results_dict = []
    # store pipeline data in the DB
    #pipeline_id = start_pipeline(train_end_date, validation_duration, visualize, verbose)
    if evaluate:
        rmse_list = []

    for i, district in enumerate(districts):
        # 1) Import Data
        # 1a) get_smoothed_infection_counts() -> directly from Database
        val_end_date = compute_end_date_of_validation_period(train_end_date, validation_duration)
        smoothen_cases = get_smoothen_cases(district, val_end_date, duration)

        # 1b) split_into_train_and_validation()
        y_train, y_val = train_test_split(data=smoothen_cases[Column.SEVEN_DAY_SMOOTHEN.value],
                                              validation_duration=validation_duration)

        ## 2) Run model_pipeline
        sarima_model = run_sarima(y_train=y_train, y_val=y_val)
        season = sarima_model["season"]
        predictions_val = sarima_model["model"].predict(validation_duration)

        # 2b) Run model without validation data
        if validate==False:
            y_train_pred = get_smoothen_cases(district, train_end_date, duration-validation_duration)
            y_train_pred = y_train_pred[Column.SEVEN_DAY_SMOOTHEN.value]
            sarima_model_without_val = sarima_model_predictions(y_train=y_train_pred, m=season, length=validation_duration)
            predictions = sarima_model_without_val.predict(duration)


        # returned:
        # I) sarima_model: season, model
        # II) sarima_model_without_val: model

        ## 3) Evaluate the results

        # 3a) Visualize results (mainly for debugging)
        if visualize:
            if validate:
                plot_sarima_val_line_plot(y_train, y_val, predictions_val)
            else:
                plot_sarima_pred_plot(y_train_pred, predictions)

        # 3b) Compute metrics (RMSE, MAPE, ...)
        if validate:
            scores = compute_evaluation_metrics(y_pred=predictions_val, y_val=y_val)
            # collecting pipeline results to a list to be used in step four
            # results_dict.append({
            #     'district': district,
            #     'pipeline_results': pipeline_result,
            #     'scores': scores,
            # })

        if evaluate:
            rmse_list.append(scores["rmse"])

        # 4) Store results in database:
        #insert_param_and_start_vals(pipeline_id, district, start_vals, pipeline_result['model_params'])
        #insert_prediction_vals(pipeline_id, district, pipeline_result['y_pred_without_train_period'], train_end_date)

    if evaluate:
        return rmse_list

    pass

    ## 4a) Meta parameters
    # 1) which model?
    # 2) what period?
    # 3) with what parameters?

    ## 4b) Predictions

    # -> basically all parameters that are set


if __name__ == '__main__':
    smoothen_cases = get_smoothen_cases('Münster', '2021-03-31', 40)
    # y_train, y_val = train_test_split(smoothen_cases)
    # train_start_date = date_int_str(y_train[Column.DATE.value][0])
    # start_vals = get_starting_values('Münster', train_start_date)
    # print(start_vals)
