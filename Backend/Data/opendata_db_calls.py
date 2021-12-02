import datetime
import time

import pandas as pd
import requests
import re

from Backend.Data.db_functions import update_db, get_table_data

population_map = {}


def get_data_by_date_and_attr(table, date1, date2, attributes):
    """

    :param table: 'table_name' as mentioned in district_list(../../Assets/Data/district_list.csv)
    generated from update_district_list().
    :param date1: from date in YYYYMMDD format. 20200301 is the starting date
    :param date2: to date in YYYYMMDD format.
    :param attributes: column name(s) needed in a string; eg: "daily_cases" or ["daily_cases", "daily_deaths"]
    :return: query result as a 'pd dataframe'
    """
    if not table:
        print("Please provide district name as table name!")

    else:
        # will update/ create+update the table about to queried
        update_district_data(table)
        return get_table_data(table, date1, date2, attributes)


def update_population_map():
    pop_list = get_table_data("district_population", 0, 0, ["district", "population"], False)

    global population_map
    for index, row in pop_list.iterrows():

        if row['district'] in population_map:
            existing_pop = population_map.get(row['district'])
            del population_map[row['district']]

            # set the bigger population always as Kreis
            if existing_pop > int(row['population']):
                population_map[row['district'] + ', Kreis'] = existing_pop
                population_map[row['district'] + ', Stadt'] = int(row['population'])

            else:
                population_map[row['district'] + ', Kreis'] = int(row['population'])
                population_map[row['district'] + ', Stadt'] = existing_pop

        else:
            population_map[row['district']] = int(row['population'])

    # special name changes
    population_map['München, Landeshauptstadt'] = population_map.pop('München, Stadt')
    population_map['Leipzig, Kreis'], population_map['Leipzig, Stadt'] = population_map['Leipzig, Stadt'], population_map['Leipzig, Kreis']


def update_all_district_data():
    update_district_list()
    district_list = get_table_data("district_list", 0, 0, "district", False)
    update_population_map()

    for district in district_list['district']:
        update_district_data(district)
        time.sleep(1)


def update_district_data(district):
    headers = {
        'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
                         '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}

    response_cases = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000'
                                  '&resource_id=8966dc58-c7f6-47a5-8af6-603fe72a5d4a&q=' + district + ':kr_inf_md'
                                  , headers=headers)

    response_deaths = requests.get(
        'https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000&resource'
        '_id=af5ad86a-5c10-48e0-a232-1e3464ae4270&q=' + district + ':kr_tod_md'
        , headers=headers)

    response_recoveries = requests.get(
        'https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000&resou'
        'rce_id=d469b463-daee-40c6-b2ad-f58b00142608&q=' + district + ':kr_gen_md'
        , headers=headers)

    response_vaccination = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?'
                                        'limit=1000&resource_id=df59e579-875d-497a-9eda-369722150d89&q=' + district
                                        , headers=headers)

    response_incidents = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000'
                                      '&resource_id=8966dc58-c7f6-47a5-8af6-603fe72a5d4a&q=' + district + ':kr_inz_rate'
                                      , headers=headers)

    cases = response_cases.json()
    deaths = response_deaths.json()
    recoveries = response_recoveries.json()
    vaccination = response_vaccination.json()
    incidents = response_incidents.json()

    daily_cases_list = {}
    cum_cases_list = {}
    daily_cases_after14d = {}
    cum_daily_cases_after14d = {}

    daily_deaths_list = {}
    cum_deaths_list = {}
    cum_deaths_after14d = {}

    daily_recoveries_list = {}
    cum_recoveries_list = {}

    daily_vacc_list = {}
    cum_vacc_list = {}

    daily_incidents_rate = {}

    column_check_okay = False

    cum_cases_a14d = 0
    for key, value in cases['result']['records'][0].items():

        if value == 'kr_inf_md':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            daily_cases_list[key] = value

            date_obj = datetime.datetime.strptime(str(key).replace("d", ""), '%Y%m%d')
            date_obj = date_obj + datetime.timedelta(days=14)
            date = 'd' + date_obj.strftime('%Y%m%d')

            daily_cases_after14d[date] = value
            cum_cases_a14d = cum_cases_a14d + int(value)
            cum_daily_cases_after14d[date] = cum_cases_a14d

    column_check_okay = False
    for key, value in cases['result']['records'][1].items():

        if value == 'kr_inf_md_kum':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            cum_cases_list[key] = value

    column_check_okay = False
    cum_death_a14d = 0
    for key, value in deaths['result']['records'][0].items():

        if value == 'kr_tod_md':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            daily_deaths_list[key] = value

            date_obj = datetime.datetime.strptime(str(key).replace("d", ""), '%Y%m%d')
            date_obj = date_obj + datetime.timedelta(days=14)
            date = 'd' + date_obj.strftime('%Y%m%d')

            # daily_cases_after14d[date] = value
            cum_death_a14d = cum_death_a14d + int(value)
            cum_deaths_after14d[date] = cum_death_a14d

    column_check_okay = False
    for key, value in deaths['result']['records'][1].items():

        if value == 'kr_tod_md_kum':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            cum_deaths_list[key] = value

    column_check_okay = False
    for key, value in recoveries['result']['records'][0].items():

        if value == 'kr_gen_md':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            daily_recoveries_list[key] = value
    column_check_okay = False
    for key, value in recoveries['result']['records'][1].items():

        if value == 'kr_gen_md_kum':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            cum_recoveries_list[key] = value

    cum_vac = 0
    for rec in vaccination['result']['records']:
        date_obj = datetime.datetime.strptime(rec['datum'], '%Y-%m-%d')
        date_obj = date_obj + datetime.timedelta(days=14)
        date = 'd' + date_obj.strftime('%Y%m%d')
        daily_vacc_list[date] = rec['kr_zweitimpf']
        # cum_vac = cum_vac + rec['kr_zweitimpf']
        # cum_vacc_list[date] = cum_vac

        # print(date, rec['kr_zweitimpf'], cum_vac)

    for key, value in incidents['result']['records'][0].items():

        if value == 'kr_inz_rate':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            if float(value) < 0:
                daily_incidents_rate[key] = 0
            else:
                daily_incidents_rate[key] = value

    shortest = {}

    if len(daily_cases_list) > len(daily_deaths_list):
        shortest = daily_deaths_list
    else:
        shortest = daily_cases_list
    if len(daily_recoveries_list) < len(shortest):
        shortest = daily_recoveries_list

    final_data = []

    cum_vac = 0
    curr_infectious = 0
    for date, value in shortest.items():
        cum_vac = cum_vac + daily_vacc_list.get(date, 0)
        adjusted_active_cases = int(cum_cases_list.get(date, 0)) - int(cum_deaths_list.get(date, 0))
        if int(cum_daily_cases_after14d.get(date, 0)) > 0:
            adjusted_active_cases = adjusted_active_cases - (
                    int(cum_daily_cases_after14d.get(date, 0)) - int(cum_deaths_list.get(date, 0)))

        seven_day_avg = 0
        for day in range(0, 7):
            current_day = datetime.datetime.strptime(str(date).replace("d", ""), '%Y%m%d')
            current_day = current_day - datetime.timedelta(days=day)
            date_key = 'd' + current_day.strftime('%Y%m%d')
            seven_day_avg = seven_day_avg + int(daily_cases_list.get(date_key, 0))

        seven_day_avg = round(seven_day_avg / 7)
        vacc_percentage = round(int(cum_vacc_list.get(date, cum_vac)) * 100 / int(population_map.get(district)), 2)

        current_day1 = datetime.datetime.strptime(str(date).replace("d", ""), '%Y%m%d')
        date_bfr_3days = current_day1 - datetime.timedelta(days=3)
        date_bfr_3days_key = 'd' + date_bfr_3days.strftime('%Y%m%d')
        curr_infectious = (curr_infectious
                           + int(daily_cases_list.get(date, 0))
                           - int(daily_cases_list.get(date_bfr_3days_key, 0)))

        final_data.append((date,
                           daily_cases_list.get(date, 0),
                           curr_infectious,
                           seven_day_avg,
                           cum_cases_list.get(date, 0),
                           daily_deaths_list.get(date, 0),
                           cum_deaths_list.get(date, 0),
                           daily_recoveries_list.get(date, 0),
                           cum_recoveries_list.get(date, 0),
                           (int(cum_cases_list.get(date, 0))
                            - int(cum_deaths_list.get(date, 0))
                            - int(cum_recoveries_list.get(date, 0))),
                           adjusted_active_cases,
                           daily_incidents_rate.get(date, 0),
                           daily_vacc_list.get(date, 0),
                           cum_vacc_list.get(date, cum_vac),
                           vacc_percentage))

    df = pd.DataFrame(final_data)
    df.columns = ['date',
                  'daily_infec',
                  'curr_infectious',
                  'seven_day_infec',
                  'cum_infec',
                  'daily_deaths',
                  'cum_deaths',
                  'daily_rec',
                  'cum_rec',
                  'active_cases',
                  'adjusted_active_cases',
                  'daily_incidents_rate',
                  'daily_vacc',
                  'cum_vacc',
                  'vacc_percentage']
    df['date'] = df['date'].apply(lambda x: x.replace('d', ''))

    update_db(district, df)


def update_district_list():
    headers = {
        'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
                         '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}
    response = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000&resource_id='
                            'af5ad86a-5c10-48e0-a232-1e3464ae4270'
                            , headers=headers)
    cases = response.json()
    district_list = []
    for rec in cases['result']['records']:
        district_list.append((rec['bundesland'], rec['kreis']))
    district_list = list(set(district_list))
    df = pd.DataFrame(district_list)
    df.columns = ['state', 'district']
    df.to_csv('../../Assets/Data/district_list.csv')
    update_db('district_list', df)


def update_district_population():
    headers = {
        'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
                         '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}
    response = requests.get(
        'https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000&resource_id='
        '6b208bc8-9b13-45c6-8614-d3ceef180e99'
        , headers=headers)
    populations = response.json()
    district_list = []
    for rec in populations['result']['records']:
        district_list.append((rec['bundesland'], rec['kreis'], rec['kr_ew_19']))
    district_list = list(set(district_list))
    df = pd.DataFrame(district_list)
    df.columns = ['state', 'district', 'population']
    # df.to_csv('../../Assets/Data/district_list.csv')
    update_db('district_population', df)


if __name__ == '__main__':
    # README: before running update_all_district_data()/update_district_data("district_name") for the FIRST time
    #         RUN update_district_list() AND update_district_population() first.
    #
    #         ALWAYS execute update_population_map() in the line BEFORE you run
    #         update_district_data("district_name")

    # update_district_list()
    # update_district_population()
    # update_population_map()
    update_all_district_data()
    # update_district_data("Münster")
    # result_df = get_data_by_date_and_attr('Rhein-Neckar-Kreis', 20210101, 20211031, ["daily_infec", "daily_deaths"])
    # print(result_df)
