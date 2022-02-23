import datetime
import os
import re
import time
import urllib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from Backend.Data.DataManager.data_util import print_progress_with_computation_time_estimate, Column
from Backend.Data.DataManager.db_calls import update_db
from Backend.Data.DataManager.db_calls import get_table_data
from Backend.Modeling.Differential_Equation_Modeling.starting_values import get_starting_values
from Backend.Modeling.Vaccination_Efficiency.get_vaccination_effectiveness_fast import get_vaccination_effectiveness

population_map = {}


def update_all_district_data():
    """
        extract all the required covid-19 related data is extracted for all the districts
    """
    update_district_list()
    # update_district_details()
    district_list = get_table_data("district_list", [Column.DISTRICT])
    district_list.sort_values(Column.DISTRICT, inplace=True, ascending=True)
    update_population_map()
    no_of_districts = len(district_list['district'])
    print('retrieving data from Corona Daten Platform:')

    start_time = datetime.datetime.now()

    for i, district in enumerate(district_list['district']):
        update_district_data(district)
        print_progress_with_computation_time_estimate(completed=i+1, total=no_of_districts, start_time=start_time, extra=district)


def parallel_corona_datenplatform_api_requests(district):
    """
        paralleling all the requests send at each district data extract iteration, to reduce waiting time
    """
    headers = {
        'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
                         '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}

    list_of_urls = [
        # Cases
        'https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000'
        '&resource_id=3cc32ee9-a5de-42f2-afca-f9c07c956d66&q=' + district + ':kr_inf_md',

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
        '&resource_id=3cc32ee9-a5de-42f2-afca-f9c07c956d66&q=' + district + ':kr_inz_rate'
    ]

    def get_url(url):
        return requests.get(url, headers=headers)

    with ThreadPoolExecutor(max_workers=10) as pool:
        response_list = list(pool.map(get_url, list_of_urls))

    responses_tuple = (r for r in response_list)

    return responses_tuple


def update_district_data(district):
    """
        extract all required covid-19 related data is extracted for a given district
    """
    # send the five main requests to corresponding corona datenplatform endpoints
    response_cases, response_deaths, response_recoveries, response_vaccination, response_incidents = \
        parallel_corona_datenplatform_api_requests(district=district)

    # retrieve the json data from the response
    cases = response_cases.json()
    deaths = response_deaths.json()
    recoveries = response_recoveries.json()
    vaccination = response_vaccination.json()
    incidents = response_incidents.json()

    # initialize the dictionaries for the data extraction process from the jsons
    daily_cases_list = {}
    cum_cases_list = {}
    daily_cases_after14d = {}
    cum_daily_cases_after14d = {}

    daily_deaths_list = {}
    cum_deaths_list = {}

    daily_recoveries_list = {}
    cum_recoveries_list = {}

    daily_vacc_list = {}
    daily_booster_list = {}
    cum_vacc_list = {}

    daily_incidents_rate = {}

    # due to the special structure of the json data.
    # a flag need to be maintained to identify the starting point of the data records
    column_check_okay = False

    # is maintained to accumulate recovered cases by having a 14days delayed gap
    cum_cases_a14d = 0
    # collect daily infections
    for key, value in cases['result']['records'][0].items():

        if value == 'kr_inf_md':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            daily_cases_list[key] = value if value is not None else daily_cases_list[list(daily_cases_list)[-1]]

            date_obj = datetime.datetime.strptime(str(key).replace("d", ""), '%Y%m%d')
            date_obj = date_obj + datetime.timedelta(days=14)
            date = 'd' + date_obj.strftime('%Y%m%d')

            daily_cases_after14d[date] = value
            cum_cases_a14d = cum_cases_a14d + int(value)
            cum_daily_cases_after14d[date] = cum_cases_a14d

    column_check_okay = False
    # collect cumulative infections
    for key, value in cases['result']['records'][1].items():

        if value == 'kr_inf_md_kum':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            cum_cases_list[key] = value if value is not None else cum_cases_list[list(cum_cases_list)[-1]]

    column_check_okay = False
    for key, value in deaths['result']['records'][0].items():

        if value == 'kr_tod_md':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            daily_deaths_list[key] = value if value is not None else daily_deaths_list[list(daily_deaths_list)[-1]]

    column_check_okay = False
    # collect cumulative deaths
    for key, value in deaths['result']['records'][1].items():

        if value == 'kr_tod_md_kum':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            cum_deaths_list[key] = value if value is not None else cum_deaths_list[list(cum_deaths_list)[-1]]

    column_check_okay = False
    # collect daily recoveries
    for key, value in recoveries['result']['records'][0].items():

        if value == 'kr_gen_md':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            daily_recoveries_list[key] = value if value is not None else daily_recoveries_list[list(daily_recoveries_list)[-1]]
    column_check_okay = False
    # collect cumulative recoveries
    for key, value in recoveries['result']['records'][1].items():

        if value == 'kr_gen_md_kum':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            cum_recoveries_list[key] = value if value is not None else cum_recoveries_list[list(cum_recoveries_list)[-1]]

    # collect daily fully vaccinations and boosters
    for rec in vaccination['result']['records']:
        date_obj = datetime.datetime.strptime(rec['datum'], '%Y-%m-%d')
        # adjustment for the 14days waiting period to be considered fully vaccinated
        date_obj_14 = date_obj + datetime.timedelta(days=14)
        date_14 = 'd' + date_obj_14.strftime('%Y%m%d')
        date = 'd' + date_obj.strftime('%Y%m%d')
        daily_vacc_list[date_14] = rec['kr_zweitimpf']
        daily_booster_list[date] = rec['kr_drittimpf']

    # collect daily incident rate
    for key, value in incidents['result']['records'][0].items():

        if value == 'kr_inz_rate':
            column_check_okay = True

        if column_check_okay and re.match("^(d[0-9]{8})", key):
            if float(value) < 0:
                daily_incidents_rate[key] = 0
            else:
                daily_incidents_rate[key] = value

    # to find the shortest duration of the collected data dates to assign as the common data extracted duration
    shortest = {}

    if len(daily_cases_list) > len(daily_deaths_list):
        shortest = daily_deaths_list
    else:
        shortest = daily_cases_list
    if len(daily_recoveries_list) < len(shortest):
        shortest = daily_recoveries_list

    # data that will be stored under the district name
    final_data = []

    cum_vac = 0
    curr_infectious = 0
    for date, value in shortest.items():
        cum_vac = cum_vac + daily_vacc_list.get(date, 0)
        # adjusted_active_cases equation simplified
        adjusted_active_cases = int(cum_cases_list.get(date, 0)) - int(cum_daily_cases_after14d.get(date, 0))

        # this value has to be maximum 90 %
        vacc_percentage = round(int(cum_vacc_list.get(date, cum_vac)) * 100 / int(population_map.get(district)), 2)
        if vacc_percentage > 90:
            vacc_percentage = 90.

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
    df = get_vaccination_effectiveness(df, district)
    df = get_starting_values(df, district)

    # store the retrieved data
    update_db(district, df)


def update_district_list():
    """
        retrieves the list of districts used in corona datenplatform and update the district_list table
    """
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
    os.makedirs(os.path.dirname("Assets/Data/district_list.csv"), exist_ok=True)
    df.to_csv('Assets/Data/district_list.csv')
    update_db('district_list', df)


def update_population_map():
    """
        initialize the population map with all the districts prior to updating corona data of the districts.
        the map is used for calculate the vaccination percentage of each district.
    """
    pop_list = get_table_data("district_details", [Column.DISTRICT, Column.POPULATION])

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

    # special name changes to align with corona datenplatform district names
    # population_map['München, Stadt'] = population_map.pop('München, Landeshauptstadt')
    population_map['Leipzig, Kreis'], population_map['Leipzig, Stadt'] = population_map['Leipzig, Stadt'], population_map['Leipzig, Kreis']


def update_district_details():
    """
       update the 'district_details' table which contains 'population','state' of the district and
       'latitude' / 'longitude' of each district main city
   """
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
    update_db('district_details', df)


if __name__ == '__main__':
    """
        to update the covid data for all the districts, please run below method: 
    """
    # update_all_district_data()

    """
        if you just need to update a single district, please run below methods in the given order: 
    """
    # update_district_list()
    # update_district_details()
    # update_district_data("[district_name]")
