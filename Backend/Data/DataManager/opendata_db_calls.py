import datetime
import time
import urllib

import pandas as pd
import requests
import re

from Backend.Data.DataManager.db_functions import update_db, get_table_data
from Backend.Data.DataManager.remote_db_manager import upload_db_file
from Backend.Modeling.Vaccination_Efficiency.get_vaccination_effectiveness_fast import get_vaccination_effectiveness
from Backend.Modeling.Differential_Equation_Modeling.starting_values import get_starting_values

from concurrent.futures import ThreadPoolExecutor

population_map = {}


def get_data_by_date_and_attr(table, date1, date2, attributes):
    """

    :param table: 'table_name' as mentioned in district_list(Assets/Data/district_list.csv)
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
    pop_list = get_table_data("district_details", 0, 0, ["district", "population"], False)

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
    # population_map['München, Stadt'] = population_map.pop('München, Landeshauptstadt')
    population_map['Leipzig, Kreis'], population_map['Leipzig, Stadt'] = population_map['Leipzig, Stadt'], population_map['Leipzig, Kreis']


def update_all_district_data():
    update_district_list()
    district_list = get_table_data("district_list", 0, 0, "district", False)
    district_list.sort_values("district", inplace=True)
    update_population_map()

    for i, district in enumerate(district_list['district']):

        update_district_data(district)
        # time.sleep(0.1)
        print('progress: ' + str((i+1)/400))


def parallel_corona_datenplatform_api_requests(district):

    headers = {
        'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
                         '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}

    list_of_urls = [
        # Cases
        'https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000'
        '&resource_id=8966dc58-c7f6-47a5-8af6-603fe72a5d4a&q=' + district + ':kr_inf_md',

        # Deaths
        'https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000&resource'
        '_id=af5ad86a-5c10-48e0-a232-1e3464ae4270&q=' + district + ':kr_tod_md',

        # Recoveries
        'https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000&resou'
        'rce_id=d469b463-daee-40c6-b2ad-f58b00142608&q=' + district + ':kr_gen_md',

        # Vaccination:
        'https://www.corona-datenplattform.de/api/3/action/datastore_search?'
        'limit=1000&resource_id=df59e579-875d-497a-9eda-369722150d89&q=' + district,

        # Incidence:
        'https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000'
        '&resource_id=8966dc58-c7f6-47a5-8af6-603fe72a5d4a&q=' + district + ':kr_inz_rate'
    ]

    def get_url(url):
        return requests.get(url, headers=headers)

    with ThreadPoolExecutor(max_workers=10) as pool:
        response_list = list(pool.map(get_url, list_of_urls))

    responses_tuple = (r for r in response_list)

    return responses_tuple





def update_district_data(district):
    response_cases, response_deaths, response_recoveries, response_vaccination, response_incidents = \
        parallel_corona_datenplatform_api_requests(district=district)

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
    daily_booster_list = {}
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
        date_obj_14 = date_obj + datetime.timedelta(days=14)
        date_14 = 'd' + date_obj_14.strftime('%Y%m%d')
        date = 'd' + date_obj.strftime('%Y%m%d')
        daily_vacc_list[date_14] = rec['kr_zweitimpf']
        daily_booster_list[date] = rec['kr_drittimpf']
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

        # seven_day_avg = 0
        # for day in range(0, 7):
        #     current_day = datetime.datetime.strptime(str(date).replace("d", ""), '%Y%m%d')
        #     current_day = current_day - datetime.timedelta(days=day)
        #     date_key = 'd' + current_day.strftime('%Y%m%d')
        #     seven_day_avg = seven_day_avg + int(daily_cases_list.get(date_key, 0))
        #
        # seven_day_avg = round(seven_day_avg / 7)
        # this value has to be maximum 90 %
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
                           # seven_day_avg,
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
                           daily_booster_list.get(date, 0),
                           cum_vacc_list.get(date, cum_vac),
                           vacc_percentage))

    df = pd.DataFrame(final_data)
    df.columns = ['date',
                  'daily_infec',
                  'curr_infectious',
                  # 'seven_day_infec',
                  'cum_infec',
                  'daily_deaths',
                  'cum_deaths',
                  'daily_rec',
                  'cum_rec',
                  'active_cases',
                  'adjusted_active_cases',
                  'daily_incidents_rate',
                  'daily_vacc',
                  'daily_booster',
                  'cum_vacc',
                  'vacc_percentage']
    df['date'] = df['date'].apply(lambda x: x.replace('d', ''))

    # Compute 7 day cases:
    df['seven_day_infec'] = df['daily_infec'].rolling(7).mean()
    df['seven_day_infec'].fillna(value=0, inplace=True)

    # Compute vaccination effectiveness:
    df = get_vaccination_effectiveness(df)
    df = get_starting_values(df, district)

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
    df.to_csv('../Assets/Data/district_list.csv')
    update_db('district_list', df)


def get_list_of_districts():
    headers = {
        'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
                         '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}
    response = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000&resource_id='
                            'af5ad86a-5c10-48e0-a232-1e3464ae4270'
                            , headers=headers)
    cases = response.json()
    district_list = []
    for rec in cases['result']['records']:
        district_list.append(rec['kreis'])
    district_list = list(set(district_list))

    return district_list


def update_district_details():
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

        address = rec['kreis']
        if rec['kreis'] == 'Kreisfreie Stadt Frankfurt am Main':
            address = 'Stadt Frankfurt am Main'
        if rec['kreis'] == 'Kreisfreie Stadt Kassel':
            address = 'Stadt Kassel'
        if rec['kreis'] == 'Würzburg, Kreis':
            address = 'Bayern Würzburg'
        url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) + '?format=json'
        response = requests.get(url).json()
        print(rec['bundesland'], rec['kreis'])
        print((rec['kr_ew_19'], response[0]["lat"], response[0]["lon"]))
        district_list.append((rec['bundesland'], rec['kreis'], rec['kr_ew_19'], response[0]["lat"], response[0]["lon"]))
        time.sleep(0.1)

    district_list = list(set(district_list))
    df = pd.DataFrame(district_list)
    df.columns = ['state', 'district', 'population', 'latitude', 'longitude']
    # df.to_csv('Assets/Data/district_list.csv')
    update_db('district_details', df)


if __name__ == '__main__':
    # README: before running update_all_district_data()/update_district_data("district_name") for the FIRST time
    #         RUN update_district_list() AND update_district_population() first.
    #
    #         ALWAYS execute update_population_map() in the line BEFORE you run
    #         update_district_data("district_name")

    update_district_list()
    # update_district_details()
    update_population_map()
    # update_all_district_data()
    update_district_data("Berlin")
    # result_df = get_data_by_date_and_attr('Rhein-Neckar-Kreis', 20210101, 20211031, ["daily_infec", "daily_deaths"])
    # print(result_df)
