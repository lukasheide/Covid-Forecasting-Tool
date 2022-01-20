from math import floor

import pandas as pd
import datetime
from meteostat import Point, Daily
from Backend.Data.DataManager.data_access_methods import get_starting_values, get_model_params
from Backend.Data.DataManager.data_util import Column, date_int_str
from Backend.Data.DataManager.db_calls import get_all_table_data, get_district_data, get_table_data_by_duration, update_db

from Backend.Modeling.Differential_Equation_Modeling.seirv_model import seirv_pipeline
from Backend.Visualization.modeling_results import plot_train_and_fitted_infections_line_plot

from isoweek import Week


def create_weekly_matrix():
    opendata = get_all_table_data(table_name='district_list')
    opendata_dist = opendata['district'].tolist()
    mob_data = get_all_table_data(table_name='destatis_mobility_data')
    start_date = '2020-03-01'
    # GET INTERVENTION DATA
    weekly_policy_dict = get_weekly_policy_data(start_date)
    # GET VARIANT DATA
    weekly_variant_dict = get_weekly_variant_data(start_date)

    # is taken from the latest date of mobility data since is has the slowest updating frequency
    # end_date = [*mob_data.columns[-1:]][0]

    for j, district in enumerate(opendata_dist):
        print(district)
        district_matrix_list = []
        shortest_dict = {}

        # GET MOBILITY DATA
        weekly_mobility_dict = get_weekly_mobility_data(district, mob_data, start_date)
        # GET WEATHER DATA
        weekly_temp_dict, weekly_wind_dict = get_weekly_weather_data(district, start_date)
        # GET LAST WEEK BETA
        weekly_beta_dict, weekly_infections_dict = get_weekly_beta(district, start_date)

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
            district_matrix_list.append((week,
                                         weekly_policy_dict.get(week, 0),
                                         weekly_variant_dict.get(week, 0),
                                         weekly_mobility_dict.get(week, 0),
                                         weekly_temp_dict.get(week, 0),
                                         weekly_wind_dict.get(week, 0),
                                         weekly_infections_dict.get(week, 0),
                                         weekly_beta_dict.get(week, 0)))

        df = pd.DataFrame(district_matrix_list)
        df.columns = ['week',
                      'policy_index',
                      'variant',
                      'mobility',
                      'temperature',
                      'wind',
                      'infections',
                      'beta']
        update_db('matrix_' + district, df)
        print('--> progress: ' + str((j+1)*100 / 401))


def get_weekly_mobility_data(district, mob_data, start_date):
    weekly_mobility_dict = {}

    dist_mobility = mob_data.loc[mob_data['Kreisname'] == district]
    dist_mobility = dist_mobility.iloc[:, dist_mobility.columns.get_loc(start_date):]
    dist_mobility = dist_mobility.fillna(0)  # move this to db data store methods
    current_week = datetime.datetime.strptime(start_date, '%Y-%m-%d').isocalendar()[1]
    week_tot = 0
    no_weeks = 1
    no_of_days = 0
    for date_col in dist_mobility.columns:
        current_day = datetime.datetime.strptime(date_col, '%Y-%m-%d')
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
    location = get_district_data(district, [Column.LATITUDE.value, Column.LONGITUDE.value])
    dist_lat = float(location[Column.LATITUDE.value].iloc[0])
    dist_lon = float(location[Column.LONGITUDE.value].iloc[0])

    district_loc = Point(dist_lat, dist_lon)
    data = Daily(district_loc,
                 datetime.datetime.strptime(start_date, '%Y-%m-%d'),
                 datetime.datetime.today(), '%Y-%m-%d')
    data = data.fetch()

    weekly_temp_dict = weather_data_extractor(data, 'tavg', start_date)
    weekly_wind_dict = weather_data_extractor(data, 'wspd', start_date)

    return weekly_temp_dict, weekly_wind_dict


def weather_data_extractor(data, attribute, start_date):
    weekly_dict = {}
    current_week = datetime.datetime.strptime(start_date, '%Y-%m-%d').isocalendar()[1]
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
    current_week = datetime.datetime.strptime(start_date, '%Y-%m-%d').isocalendar()[1]
    week_tot = 0
    no_of_days = 0
    no_weeks = 1
    start_flag = False

    for i, row in policy_data.iterrows():
        if row['date'] == start_date:
            start_flag = True

        if start_flag:
            week = datetime.datetime.strptime(row['date'], '%Y-%m-%d').isocalendar()[1]

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
    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    start_week = start_date_obj.isocalendar()[1]
    no_weeks = 1
    week_variant = ''
    weeks_percentage = 0
    active_week = int(variant_data['year_week'][0].split("-")[1])
    week_gaps = active_week - start_week

    while week_gaps > 0:
        weekly_variant_dict[no_weeks] = 'Other'
        no_weeks = no_weeks + 1
        week_gaps = week_gaps - 1

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

    return weekly_variant_dict


def get_weekly_beta(district, start_date, debug=False):
    weekly_beta_values = {}
    weekly_infections = {}
    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')

    duration = 14

    running_week = start_date_obj.isocalendar()[1]
    week_no = 1
    break_start = 0

    # Adapt starting date so that it starts on a Monday:
    start_date_monday = Week(start_date_obj.year, start_date_obj.isocalendar()[1]).monday()

    # Ensure that we start atleast n (duration) days behind 01.03.2020:
    if start_date_monday \
            < (datetime.datetime.strptime("2020-03-01", '%Y-%m-%d') + datetime.timedelta(days=duration + 7)).date():

        min_date = (datetime.datetime.strptime("2020-03-01", '%Y-%m-%d') + datetime.timedelta(days=duration + 7)).date()
        start_date_monday = min_date

    running_week = start_date_monday.isocalendar()[1]

    idx_end = duration-1

    start_date_dataframe = datetime.datetime.combine(start_date_monday, datetime.datetime.min.time()) - datetime.timedelta(days=duration-1)
    all_smoothen_cases = get_table_data_by_duration(table=district,
                                                    start_date=start_date_dataframe.strftime('%Y-%m-%d'),
                                                    attributes=[Column.DATE.value, Column.SEVEN_DAY_SMOOTHEN.value])


    # cut of last rows so that we have multiples of 7:
    last_idx = floor(len(all_smoothen_cases)/7)*7

    all_smoothen_cases = all_smoothen_cases.iloc[0:last_idx]


    for i, row in all_smoothen_cases.iterrows():
        current_week = datetime.datetime.strptime(row['date'], '%Y%m%d').isocalendar()[1]

        if current_week != running_week and idx_end <= len(all_smoothen_cases):
            idx_start = idx_end+1-duration
            y_train = all_smoothen_cases[idx_start:idx_end+1]
            train_start_date = date_int_str(all_smoothen_cases[Column.DATE.value][break_start])
            y_train = y_train[Column.SEVEN_DAY_SMOOTHEN.value].reset_index(drop=True)
            start_vals = get_starting_values(district, train_start_date)
            fixed_model_params = get_model_params(district, train_start_date)
            pipeline_result = seirv_pipeline(y_train=y_train, start_vals_fixed=start_vals,
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


def create_complete_matrix_data():
    districts = get_all_table_data(table_name='district_list')
    districts_list = districts['district'].tolist()

    final_df = pd.DataFrame([])

    for district in districts_list:
        district_df = get_all_table_data(table_name='matrix_' + district)
        district_df['district'] = district
        final_df = pd.concat([final_df, district_df])

    final_df.fillna(0)
    final_df.to_csv('../Assets/Data/all_matrix_data.csv')
    # update_db('all_matrix_data', final_df)

if __name__ == '__main__':
    # mob_data = get_all_table_data(table_name='destatis_mobility_data')
    # mob_data_dist = mob_data['Kreisname'].to_list()
    # opendata = get_all_table_data(table_name='district_details')
    # opendata_dist = opendata['district'].tolist()
    #
    # for mob_dist in mob_data_dist:
    #
    #     found = False
    #     for district in opendata_dist:
    #         if mob_dist == district:
    #             found = True
    #
    #     if not found:
    #         print("case '"+mob_dist+"' :")
    # Kaiserslautern
    # create_weekly_matrix()
    # get_weekly_variant_data('2020-03-01')
    # weekly_mobility_dict = get_weekly_mobility_data('Stadt Neustadt a.d. W.', get_all_table_data(table_name='destatis_mobility_data'),  '2020-03-01')
    create_weekly_matrix()
    # create_complete_matrix_data()
    # get_weekly_beta('Münster','2021-02-01')
