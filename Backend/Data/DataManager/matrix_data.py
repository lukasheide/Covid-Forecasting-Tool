import os
import time
from math import floor

from meteostat import Point, Daily
from Backend.Data.DataManager.data_access_methods import get_starting_values, get_model_params
from Backend.Data.DataManager.data_util import Column, date_int_str, print_progress, \
    print_progress_with_computation_time_estimate, format_name
from Backend.Data.DataManager.db_calls import get_all_table_data, get_district_data, get_table_data_by_duration, \
    update_db, get_policy_data, get_variant_data, get_mobility_data, get_weather_data, drop_table_by_name

from Backend.Modeling.Differential_Equation_Modeling.seiurv_model import seiurv_pipeline, fit_seirv_model, \
    fit_seiurv_model_only_beta
from Backend.Visualization.plotting import plot_train_and_fitted_infections_line_plot, \
    plot_beta_matrix_estimation

from isoweek import Week
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def create_weekly_matrix():
    start_time_pipeline = datetime.now()

    opendata = get_all_table_data(table_name='district_list')
    district_list = opendata['district'].tolist()
    district_list.sort()

    # For Debugging:
    # start_with_district = 'Höxter'
    # start_idx = district_list.index(start_with_district)
    # district_list = district_list[start_idx:]

    mob_data = get_all_table_data(table_name='destatis_mobility_data')
    start_date = '2020-03-10'
    end_date = datetime.today().strftime('%Y-%m-%d')

    # start_date = '2021-11-01'
    # end_date = '2021-12-01'


    # GET INTERVENTION DATA
    weekly_policy_dict = get_weekly_policy_data(start_date)
    # GET VARIANT DATA
    weekly_variant_dict = get_weekly_variant_data(start_date)

    # is taken from the latest date of mobility data since is has the slowest updating frequency
    # end_date = [*mob_data.columns[-1:]][0]

    # only one city for debugging:
    # district_list = ['Essen', 'Bielefeld', 'Münster', 'Dortmund', 'Bochum', 'Warendorf']



    for j, district in enumerate(district_list):
        # print(district)
        start_time = datetime.now()
        district_matrix_list = []
        shortest_dict = {}

        # GET MOBILITY DATA
        weekly_mobility_dict = get_weekly_mobility_data(district, mob_data, start_date)
        # GET WEATHER DATA
        weekly_temp_dict, weekly_wind_dict = get_weekly_weather_data(district, start_date)
        # GET LAST WEEK BETA
        weekly_beta_dict, weekly_beta_t_minus_1, weekly_infections_dict, start_forecasting_dict = get_weekly_beta_v2(
            district, start_date, end_date)

        # if len(weekly_policy_dict) > len(weekly_variant_dict):
        #     shortest_dict = weekly_variant_dict
        # else:
        #     shortest_dict = weekly_policy_dict
        # if len(weekly_mobility_dict) < len(shortest_dict):
        #     shortest_dict = weekly_mobility_dict
        # if len(weekly_temp_dict) < len(shortest_dict):
        #     shortest_dict = weekly_temp_dict
        # if len(weekly_beta_dict) < len(shortest_dict):
        #     shortest_dict = weekly_temp_dict

        for week, value in weekly_beta_dict.items():

            # cut of weeks not included:
            min_week = min(weekly_beta_dict.keys())
            max_week = max(weekly_beta_dict.keys())
            if min_week <= week <= max_week:
                district_matrix_list.append((week,
                                             weekly_policy_dict.get(week, 0),
                                             weekly_variant_dict.get(week, 0),
                                             weekly_mobility_dict.get(week, 0),
                                             weekly_temp_dict.get(week, 0),
                                             weekly_wind_dict.get(week, 0),
                                             weekly_infections_dict.get(week, 0),
                                             weekly_beta_dict.get(week, 0),
                                             weekly_beta_t_minus_1.get(week, 0),
                                             start_forecasting_dict.get(week, 0)))

        df = pd.DataFrame(district_matrix_list)
        df.columns = ['week',
                      'policy_index',
                      'variant',
                      'mobility',
                      'temperature',
                      'wind',
                      'infections',
                      'beta',
                      'beta_t_minus_1',
                      'start_date_forecasting']
        # this table creation is only intermediate
        # and will be deleted at the end of create_complete_matrix_data() execution
        update_db('matrix_' + district, df)
        end_time = datetime.now()
        extra_str = '--> ' + district + ' | calculation time: ' + str(end_time - start_time)
        print_progress_with_computation_time_estimate(completed=j + 1, total=len(district_list), extra=extra_str, start_time=start_time_pipeline)


def get_weekly_mobility_data(district, mob_data, start_date):
    weekly_mobility_dict = {}

    dist_mobility = mob_data.loc[mob_data['Kreisname'] == district]
    dist_mobility = dist_mobility.iloc[:, dist_mobility.columns.get_loc(start_date):]
    dist_mobility = dist_mobility.fillna(0)  # move this to db data store methods
    current_week = datetime.strptime(start_date, '%Y-%m-%d').isocalendar()[1]
    week_tot = 0
    no_weeks = 1
    no_of_days = 0
    for date_col in dist_mobility.columns:
        current_day = datetime.strptime(date_col, '%Y-%m-%d')
        week = current_day.isocalendar()[1]

        if current_week == week:
            week_tot = week_tot + float(dist_mobility[date_col].iloc[0])
            no_of_days = no_of_days + 1
        else:
            weekly_mobility_dict[no_weeks] = week_tot / no_of_days

            current_week = week
            no_weeks = no_weeks + 1
            week_tot = float(dist_mobility[date_col].iloc[0])
            no_of_days = 1

    return weekly_mobility_dict


def get_weekly_weather_data(district, start_date):
    if district == 'Garmisch-Partenkirchen':
        district = 'Weilheim-Schongau'
    location = get_district_data(district, [Column.LATITUDE, Column.LONGITUDE])
    dist_lat = float(location[Column.LATITUDE].iloc[0])
    dist_lon = float(location[Column.LONGITUDE].iloc[0])

    district_loc = Point(dist_lat, dist_lon)
    data = Daily(district_loc,
                 datetime.strptime(start_date, '%Y-%m-%d'),
                 datetime.today(), '%Y-%m-%d')
    data = data.fetch()

    weekly_temp_dict = weather_data_extractor(data, 'tavg', start_date)
    weekly_wind_dict = weather_data_extractor(data, 'wspd', start_date)

    return weekly_temp_dict, weekly_wind_dict


def weather_data_extractor(data, attribute, start_date):
    weekly_dict = {}
    current_week = datetime.strptime(start_date, '%Y-%m-%d').isocalendar()[1]
    week_tot = 0
    no_of_days = 0
    no_weeks = 1

    for i, item in data[attribute].items():
        week = i.isocalendar()[1]

        if current_week == week:
            week_tot = week_tot + float(item)
            no_of_days = no_of_days + 1
        else:
            weekly_dict[no_weeks] = week_tot / no_of_days

            current_week = week
            no_weeks = no_weeks + 1
            week_tot = float(item)
            no_of_days = 1

    return weekly_dict


def get_weekly_policy_data(start_date):
    policy_data = get_all_table_data(table_name='xocgrt_policy_data')
    weekly_policy_dict = {}
    current_week = datetime.strptime(start_date, '%Y-%m-%d').isocalendar()[1]
    week_tot = 0
    no_of_days = 0
    no_weeks = 1
    start_flag = False

    for i, row in policy_data.iterrows():
        if row['date'] == start_date:
            start_flag = True

        if start_flag:
            week = datetime.strptime(row['date'], '%Y-%m-%d').isocalendar()[1]

            if current_week == week:
                week_tot = week_tot + float(row['policy_index'])
                no_of_days = no_of_days + 1
            else:
                weekly_policy_dict[no_weeks] = week_tot / no_of_days

                current_week = week
                no_weeks = no_weeks + 1
                week_tot = float(row['policy_index'])
                no_of_days = 1

    return weekly_policy_dict


def get_weekly_variant_data(start_date):
    variant_data = get_all_table_data(table_name='ecdc_varient_data')
    weekly_variant_dict = {}
    no_weeks = 1
    week_variant = ''
    weeks_percentage = 0
    today_date_obj = datetime.today()
    today_week = today_date_obj.isocalendar()[1]
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    start_week = start_date_obj.isocalendar()[1]
    active_week = int(variant_data['year_week'][0].split("-")[1])
    start_week_gaps = active_week - start_week
    end_week_gaps = today_week - int(variant_data['year_week'].iloc[-1].split("-")[1])

    while start_week_gaps > 0:
        weekly_variant_dict[no_weeks] = 'Other'
        no_weeks = no_weeks + 1
        start_week_gaps = start_week_gaps - 1

    for i, row in variant_data.iterrows():
        this_date = row['year_week'].split("-")
        this_week = int(this_date[1])

        if weeks_percentage < float(row['percent_variant']):
            week_variant = row['variant']
            weeks_percentage = float(row['percent_variant'])

        if active_week != this_week:
            weekly_variant_dict[no_weeks] = week_variant
            no_weeks = no_weeks + 1
            active_week = this_week
            weeks_percentage = 0

    while end_week_gaps > 0:
        weekly_variant_dict[no_weeks] = 'B.1.1.529'
        no_weeks = no_weeks + 1
        end_week_gaps = end_week_gaps - 1

    return weekly_variant_dict


def get_weekly_beta_DEPRECATED(district, start_date, debug=False):
    weekly_beta_values = {}
    weekly_infections = {}
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')

    duration = 14

    running_week = start_date_obj.isocalendar()[1]
    week_no = 1
    break_start = 0

    # Adapt starting date so that it starts on a Monday:
    start_date_monday = Week(start_date_obj.year, start_date_obj.isocalendar()[1]).monday()

    # Ensure that we start atleast n (duration) days behind 01.03.2020:
    if start_date_monday \
            < (datetime.strptime("2020-03-01", '%Y-%m-%d') + timedelta(days=duration + 7)).date():
        min_date = (datetime.strptime("2020-03-01", '%Y-%m-%d') + timedelta(days=duration + 7)).date()
        start_date_monday = min_date

    running_week = start_date_monday.isocalendar()[1]

    idx_end = duration - 1

    start_date_dataframe = datetime.combine(start_date_monday, datetime.min.time()) - timedelta(days=duration - 1)
    all_smoothen_cases = get_table_data_by_duration(table=district,
                                                    start_date=start_date_dataframe.strftime('%Y-%m-%d'),
                                                    attributes=[Column.DATE, Column.SEVEN_DAY_SMOOTHEN])

    # cut of last rows so that we have multiples of 7:
    last_idx = floor(len(all_smoothen_cases) / 7) * 7

    all_smoothen_cases = all_smoothen_cases.iloc[0:last_idx]

    for i, row in all_smoothen_cases.iterrows():
        current_week = datetime.strptime(row['date'], '%Y%m%d').isocalendar()[1]

        if current_week != running_week and idx_end <= len(all_smoothen_cases):
            idx_start = idx_end + 1 - duration
            y_train = all_smoothen_cases[idx_start:idx_end + 1]
            train_start_date = date_int_str(all_smoothen_cases[Column.DATE.value][break_start])
            y_train = y_train[Column.SEVEN_DAY_SMOOTHEN.value].reset_index(drop=True)
            start_vals = get_starting_values(district, train_start_date)
            fixed_model_params = get_model_params(district, train_start_date)
            pipeline_result = seiurv_pipeline(y_train=y_train, start_vals_fixed=start_vals,
                                              fixed_model_params=fixed_model_params,
                                              allow_randomness_fixed_beta=False, random_runs=100)

            # Plots for debugging:
            if debug:
                plot_train_and_fitted_infections_line_plot(y_train, pipeline_result['y_pred_including_train_period'])

            weekly_beta_values[week_no] = pipeline_result['model_params_forecast_period']['beta']
            weekly_infections[week_no] = y_train.mean()

            running_week = current_week
            week_no = week_no + 1
            idx_end = idx_end + 7

    return weekly_beta_values, weekly_infections


def get_weekly_beta_v2(district, start_date, end_date, debug=False):

    """
    The Purpose of this function is to compute for each district / time interval combination the fitted beta
    for the training period (last beta) and the fitted beta for the validation period (what would've been the
    perfect beta in this period). This data is then later used for training the machine learning model.
    """

    print(f"Starting computation of weekly betas at time: {datetime.now()}")

    weekly_beta_t_minus_1_values = {}
    weekly_beta_values = {}
    weekly_infections = {}
    weekly_forecasting_start_date = {}

    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')

    ### Params:
    train_duration = 14
    validation_duration = 14

    intervals_grid = get_weekly_intervals_grid(start_date, end_date, train_duration, validation_duration)

    all_smoothen_cases = get_table_data_by_duration(table=district,
                                                    start_date=intervals_grid[0]['start_day_train_str'],
                                                    attributes=[Column.DATE, Column.SEVEN_DAY_SMOOTHEN])
    # append one column for formatted dates:
    all_smoothen_cases['date_str'] = all_smoothen_cases[Column.DATE].apply(
        lambda row: datetime.strptime(row, '%Y%m%d').strftime('%Y-%m-%d'))

    results = []

    start_time = time.time()

    for week_num, current_interval in enumerate(intervals_grid):

        # For Debugging:
        # if week_num == 64:
        #    print('stop')

        # Get DataFrame indices:
        start_idx_train = \
            all_smoothen_cases.loc[all_smoothen_cases['date_str'] == current_interval['start_day_train_str']].index[0]
        end_idx_train = \
        all_smoothen_cases.loc[all_smoothen_cases['date_str'] == current_interval['end_day_train_str']].index[0]
        start_idx_val = \
        all_smoothen_cases.loc[all_smoothen_cases['date_str'] == current_interval['start_day_val_str']].index[0]
        end_idx_val = \
        all_smoothen_cases.loc[all_smoothen_cases['date_str'] == current_interval['end_day_val_str']].index[0]

        # Get infection counts for training and for validation:
        y_train = all_smoothen_cases[Column.SEVEN_DAY_SMOOTHEN].iloc[start_idx_train:end_idx_train + 1].reset_index(
            drop=True)
        y_val = all_smoothen_cases[Column.SEVEN_DAY_SMOOTHEN].iloc[start_idx_val:end_idx_val + 1].reset_index(drop=True)

        # Get starting values for training
        start_vals_train = get_starting_values(district, current_interval['start_day_train_str'])
        fixed_model_params_train = get_model_params(district, current_interval['start_day_train_str'])

        ## 1) Run Pipeline for training period(compute beta_t -1):
        training_pipeline_results = seiurv_pipeline(y_train=y_train,
                                                    start_vals_fixed=start_vals_train,
                                                    fixed_model_params=fixed_model_params_train,
                                                    allow_randomness_fixed_beta=False,
                                                    random_runs=100)

        ## 2) Run fitting again for validation period: -> "What would've been the perfect beta?"

        fixed_start_vals_from_training = {
            'S0': training_pipeline_results['model_start_vals_forecast_period']['S'],
            'E0': training_pipeline_results['model_start_vals_forecast_period']['E'],
            'I0': training_pipeline_results['model_start_vals_forecast_period']['I'],
            'U0': training_pipeline_results['model_start_vals_forecast_period']['U'],
            'R0': training_pipeline_results['model_start_vals_forecast_period']['R'],
            'V0': training_pipeline_results['model_start_vals_forecast_period']['V'],
        }

        validation_pipeline_result = fit_seiurv_model_only_beta(y_val,
                                                                start_vals_fixed=fixed_start_vals_from_training,
                                                                fixed_model_params=fixed_model_params_train,
                                                                district=district)

        if debug:
            plot_beta_matrix_estimation(y_train_true=y_train,
                                        y_val_true=y_val,
                                        y_train_pred_full=training_pipeline_results['y_pred_including_train_period'],
                                        y_val_pred=validation_pipeline_result['daily_infections'],
                                        district=district,
                                        start_date=current_interval['start_day_train_str'],
                                        end_date=current_interval['end_day_val_str'], )

        # index starts with 0 -> +1
        # starts with training period of 14 days, so the first beta estimate corresponds to end of week two and not week one-> +1:
        # 1 + 1 -> 2 index shift
        weekly_beta_t_minus_1_values[week_num + 2] = training_pipeline_results['model_params_forecast_period']['beta']
        weekly_beta_values[week_num + 2] = validation_pipeline_result['fitted_params']['beta']
        weekly_infections[week_num + 2] = y_val.mean()
        weekly_forecasting_start_date[week_num + 2] = current_interval['start_day_val_str']

    end_time = time.time()
    # print(f'Duration: {end_time-start_time}')

    return weekly_beta_values, weekly_beta_t_minus_1_values, weekly_infections, weekly_forecasting_start_date


def get_weekly_intervals_grid(start_day, last_day, duration_train, duration_val):
    full_duration = duration_train + duration_val

    # Compute Date Objects and Dates as strings for training_begin, training_end, validation_begin, validation_end:
    start_day_dt_obj = datetime.strptime(start_day, '%Y-%m-%d')
    last_day_dt_obj = datetime.strptime(last_day, '%Y-%m-%d')

    current_start_day_train = start_day_dt_obj
    current_end_day_train = current_start_day_train + timedelta(days=duration_train - 1)
    current_start_day_val = current_start_day_train + timedelta(days=duration_train)
    current_end_day_val = current_start_day_train + timedelta(days=full_duration - 1)

    intervals_grid = []
    while True:
        intervals_grid.append({
            # Date Objects
            'start_day_train_obj': current_start_day_train,
            'end_day_train_obj': current_end_day_train,
            'start_day_val_obj': current_start_day_val,
            'end_day_val_obj': current_end_day_val,

            # Strings:
            'start_day_train_str': current_start_day_train.strftime('%Y-%m-%d'),
            'end_day_train_str': current_end_day_train.strftime('%Y-%m-%d'),
            'start_day_val_str': current_start_day_val.strftime('%Y-%m-%d'),
            'end_day_val_str': current_end_day_val.strftime('%Y-%m-%d'),
        })

        # Increase all:
        # by 7 days
        current_start_day_train = current_start_day_train + timedelta(days=7)
        current_end_day_train = current_end_day_train + timedelta(days=7)
        current_start_day_val = current_start_day_val + timedelta(days=7)
        current_end_day_val = current_end_day_val + timedelta(days=7)

        # Stop once end is reached:
        if current_end_day_val > last_day_dt_obj:
            break

    return intervals_grid


def create_complete_matrix_data(debug=False):
    create_weekly_matrix()
    districts = get_all_table_data(table_name='district_list')
    districts_list = districts['district'].tolist()

    final_df = pd.DataFrame([])

    for i, district in enumerate(districts_list):
        print_progress(completed=i+1, total=len(districts_list))
        district_df = get_all_table_data(table_name='matrix_' + district)
        if district == 'Garmisch-Partenkirchen':
            replacement_data = get_all_table_data(table_name='matrix_Weilheim_Schongau')
            district_df['wind'] = replacement_data['wind']
            district_df['temperature'] = replacement_data['temperature']

        district_df['district'] = district
        final_df = pd.concat([final_df, district_df])

    final_df.fillna(0)
    os.makedirs(os.path.dirname('Assets/Data/all_matrix_data_v3.csv'), exist_ok=True)
    final_df.to_csv('Assets/Data/all_matrix_data_v3.csv')
    update_db('all_matrix_data', final_df)

    # remove all intermediate 'matrix_[district_name]' tables
    for i, district in enumerate(districts_list):
        drop_table_by_name(format_name('matrix_'+district))

    ## Debugging:
    if debug:
        # get outliers:
        df_outliers = final_df[final_df['beta'] < 0.01]

        # group by district:
        temp_df = df_outliers.groupby(['district'])['district'].count()


def get_predictors_for_ml_layer(district, start_date):
    # GET INTERVENTION DATA
    policy_index = get_policy_data(start_date)

    # GET VARIANT DATA
    variant = get_variant_data(start_date)

    # GET MOBILITY DATA
    mobility = get_mobility_data(district, start_date)

    # GET WEATHER DATA
    temperature, wind = get_weather_data(district, start_date)

    # Create dictionary:
    ml_predictors_dict = {
        # One-hot-encode variant data:
        'B.1.1.529': 1 if variant == 'B.1.1.529' else 0,
        'B.1.1.7': 1 if variant == 'B.1.1.7' else 0,
        'B.1.617.2': 1 if variant == 'B.1.617.2' else 0,
        'policy_index': policy_index,
        'mobility': mobility,
        'temperature': temperature,
        'wind': wind,
    }

    return ml_predictors_dict


def prepare_all_beta_predictors(y_train_last_two_weeks: np.array, previous_beta: float,
                                ml_predictors=dict) -> pd.Series:
    mean_infections_last_two_weeks = y_train_last_two_weeks.mean()

    ml_predictors['infections'] = mean_infections_last_two_weeks
    ml_predictors['beta_t_minus_1'] = previous_beta

    df = pd.DataFrame(columns=list(ml_predictors.keys()))
    df.loc[0] = list(ml_predictors.values())

    return df


if __name__ == '__main__':
    # to create the matrix data for the machine learning layer
    create_complete_matrix_data()
