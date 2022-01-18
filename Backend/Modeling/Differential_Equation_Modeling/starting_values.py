# goal: calculate starting values for SRV for any given day for differential compartment models
import re

from Backend.Modeling.Vaccination_Efficiency.get_vaccination_efficiency import *
import datetime
import time

# def get_vaccination_data(district, state):
#     df_vaccination_info = get_vaccination_number(district)
#     df_manufacturer = get_manufacturers(state)
#     df_merged = merge_vaccinations_manufacturers(df_vaccination_info, df_manufacturer)
#     df_merged = get_manufacturer_share(df_merged)
#     df_merged = get_vaccination_status(df_merged)
#     df_merged = get_vaccination_efficiency(df_merged)
#     df_merged = get_vaccinaction_efficiency_waning(df_merged)
#     df_merged = get_vaccinaction_efficiency_waning_booster(df_merged)
#
#     df_vaccination = df_merged[['date', 'people_vacc_last_180_days_booster', 'total_vacc_efficiency_today_corrected_waning_booster']]
#     df_vaccination = df_vaccination.rename(columns={'people_vacc_last_180_days_booster': 'number_people_vacc', 'total_vacc_efficiency_today_corrected_waning_booster': 'vacc_efficiency'})
#
#     #clean NaN values from dataframe
#     df_vaccination = df_vaccination.fillna(0)
#
#     print(df_vaccination)
#     return df_merged, df_vaccination

# def get_recovered_data(district):
#     headers = {
#         'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
#                          '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}
#     response_recovered = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?'
#                                          'limit=1000&resource_id=d469b463-daee-40c6-b2ad-f58b00142608&q=' + district + ':kr_gen_md'
#                                          , headers=headers)
#     recovered = response_recovered.json()
#
#     daily_recovered = {}
#
#
#     column_check_okay = False
#     for key, value in recovered['result']['records'][0].items():
#         if value == 'kr_gen_md':
#             column_check_okay = True
#         if column_check_okay:
#             daily_recovered[key] = value
#
#     data_recovered_items = daily_recovered.items()
#     data_recovered_list = list(data_recovered_items)
#     df_recovered = pd.DataFrame(data_recovered_list)
#     df_recovered.columns = ['date', 'recovered']
#
#     #clean data, bring date into right format
#     df_recovered['date'] = df_recovered['date'].apply(lambda x: x.replace('d', ''))
#     df_recovered = df_recovered.drop(df_recovered[df_recovered.date == 'variable'].index)
#     df_recovered = df_recovered.drop(df_recovered[df_recovered.date == 'rank'].index)
#     df_recovered['date'] = df_recovered['date'].apply( lambda x: datetime.datetime.strptime(str(x).replace("d", ""), '%Y%m%d'))
#     print(df_recovered)
#
#     # adding cumulated column
#     recovered_help_cum = df_recovered['recovered']
#     recovered_help_cum = recovered_help_cum.cumsum()
#     df_recovered['total_recovered'] = recovered_help_cum
#
#     # adding cumulated for last 180 days column
#     df_recovered['total_recovered_180'] = 0
#     for i in range(1, len(df_recovered)):
#         if i <= 180:
#             df_recovered.loc[i, 'total_recovered_180'] = df_recovered.loc[i, 'total_recovered']
#         elif i > 180:
#             df_recovered.loc[i, 'total_recovered_180'] = df_recovered.loc[i, 'total_recovered'] - df_recovered.loc[i-180, 'total_recovered']
#
#     # calculate recovered including undetected cases (undetected rate of 1.35)
#     df_recovered['R'] = 0
#     df_recovered['R'] = df_recovered.apply(lambda x: x['total_recovered_180'] * 1.35, axis=1)
#     print(df_recovered)
#     return df_recovered

def get_number_inhabitants(district):
    # get number of inhabitants
    headers = {
        'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
                         '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}
    inhabitants = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?'
                               'limit=1000&resource_id=6b208bc8-9b13-45c6-8614-d3ceef180e99&q=' + district,
                               headers=headers)
    inhabitants_list = inhabitants.json()
    temp_1 = inhabitants_list['result']
    temp_2 = temp_1['records']
    temp_3 = temp_2[0]
    number_inhabitants = temp_3['kr_ew_19']
    return number_inhabitants

# def get_vaccination_breakthrough(df_vaccination, df_recovered, district):
    # #get number of inhabitants
    # headers = {
    #     'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
    #                      '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}
    # inhabitants = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?'
    #                                   'limit=1000&resource_id=6b208bc8-9b13-45c6-8614-d3ceef180e99&q=' + district,
    #                            headers=headers)
    # inhabitants_list = inhabitants.json()
    # temp_1 = inhabitants_list['result']
    # temp_2 = temp_1['records']
    # temp_3 = temp_2[0]
    # number_inhabitants = temp_3['kr_ew_19']

    #get vaccination rate: number vaccinated/ number_inhabitants
    # df_vaccination_rate = df_vaccination[1]
    # df_vaccination_rate['vaccination_rate'] = 0
    # df_vaccination_rate['vaccination_rate'] = df_vaccination_rate.apply(lambda x: x['number_people_vacc'] / number_inhabitants, axis=1)

    #get number of infected/active cases

    # headers = {
    #     'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
    #                      '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}
    #
    # response_cases = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000'
    #                               '&resource_id=8966dc58-c7f6-47a5-8af6-603fe72a5d4a&q=' + district + ':kr_inf_md'
    #                               , headers=headers)
    #
    # response_deaths = requests.get(
    #     'https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000&resource'
    #     '_id=af5ad86a-5c10-48e0-a232-1e3464ae4270&q=' + district + ':kr_tod_md'
    #     , headers=headers)
    #
    # response_recoveries = requests.get(
    #     'https://www.corona-datenplattform.de/api/3/action/datastore_search?limit=1000&resou'
    #     'rce_id=d469b463-daee-40c6-b2ad-f58b00142608&q=' + district + ':kr_gen_md'
    #     , headers=headers)
    #
    # cases = response_cases.json()
    # deaths = response_deaths.json()
    # recoveries = response_recoveries.json()
    #
    # daily_cases_list = {}
    # cum_cases_list = {}
    # daily_cases_after14d = {}
    # cum_daily_cases_after14d = {}
    # daily_deaths_list = {}
    # cum_deaths_list = {}
    # cum_deaths_after14d = {}
    # daily_recoveries_list = {}
    # cum_recoveries_list = {}
    #
    # column_check_okay = False
    # cum_cases_a14d = 0
    #
    # for key, value in cases['result']['records'][0].items():
    #
    #     if value == 'kr_inf_md':
    #         column_check_okay = True
    #
    #     if column_check_okay and re.match("^(d[0-9]{8})", key):
    #         daily_cases_list[key] = value
    #
    #         date_obj = datetime.datetime.strptime(str(key).replace("d", ""), '%Y%m%d')
    #         date_obj = date_obj + datetime.timedelta(days=14)
    #         date = 'd' + date_obj.strftime('%Y%m%d')
    #
    #         daily_cases_after14d[date] = value
    #         cum_cases_a14d = cum_cases_a14d + int(value)
    #         cum_daily_cases_after14d[date] = cum_cases_a14d
    #
    # column_check_okay = False
    # for key, value in cases['result']['records'][1].items():
    #
    #     if value == 'kr_inf_md_kum':
    #         column_check_okay = True
    #
    #     if column_check_okay and re.match("^(d[0-9]{8})", key):
    #         cum_cases_list[key] = value
    #
    # column_check_okay = False
    # cum_death_a14d = 0
    # for key, value in deaths['result']['records'][0].items():
    #
    #     if value == 'kr_tod_md':
    #         column_check_okay = True
    #
    #     if column_check_okay and re.match("^(d[0-9]{8})", key):
    #         daily_deaths_list[key] = value
    #
    #         date_obj = datetime.datetime.strptime(str(key).replace("d", ""), '%Y%m%d')
    #         date_obj = date_obj + datetime.timedelta(days=14)
    #         date = 'd' + date_obj.strftime('%Y%m%d')
    #
    #         # daily_cases_after14d[date] = value
    #         cum_death_a14d = cum_death_a14d + int(value)
    #         cum_deaths_after14d[date] = cum_death_a14d
    #
    # column_check_okay = False
    # for key, value in deaths['result']['records'][1].items():
    #
    #     if value == 'kr_tod_md_kum':
    #         column_check_okay = True
    #
    #     if column_check_okay and re.match("^(d[0-9]{8})", key):
    #         cum_deaths_list[key] = value
    #
    # column_check_okay = False
    # for key, value in recoveries['result']['records'][0].items():
    #
    #     if value == 'kr_gen_md':
    #         column_check_okay = True
    #
    #     if column_check_okay and re.match("^(d[0-9]{8})", key):
    #         daily_recoveries_list[key] = value
    # column_check_okay = False
    # for key, value in recoveries['result']['records'][1].items():
    #
    #     if value == 'kr_gen_md_kum':
    #         column_check_okay = True
    #
    #     if column_check_okay and re.match("^(d[0-9]{8})", key):
    #         cum_recoveries_list[key] = value
    #
    # shortest = {}
    # if len(daily_cases_list) > len(daily_deaths_list):
    #     shortest = daily_deaths_list
    # else:
    #     shortest = daily_cases_list
    # if len(daily_recoveries_list) < len(shortest):
    #     shortest = daily_recoveries_list
    #
    # final_data = []
    #
    # for date, value in shortest.items():
    #     adjusted_active_cases = int(cum_cases_list.get(date, 0)) - int(cum_deaths_list.get(date, 0))
    #     if int(cum_daily_cases_after14d.get(date, 0)) > 0:
    #         adjusted_active_cases = adjusted_active_cases - (
    #                 int(cum_daily_cases_after14d.get(date, 0)) - int(cum_deaths_list.get(date, 0)))
    #     seven_day_avg = 0
    #     for day in range(0, 7):
    #         current_day = datetime.datetime.strptime(str(date).replace("d", ""), '%Y%m%d')
    #         current_day = current_day - datetime.timedelta(days=day)
    #         date_key = 'd' + current_day.strftime('%Y%m%d')
    #         seven_day_avg = seven_day_avg + int(daily_cases_list.get(date_key, 0))
    #
    #     current_day1 = datetime.datetime.strptime(str(date).replace("d", ""), '%Y%m%d')
    #     date_bfr_3days = current_day1 - datetime.timedelta(days=3)
    #
    #     final_data.append((date,
    #                        adjusted_active_cases
    #                        ))
    # df_cases = pd.DataFrame(final_data)
    # df_cases.columns = ['date',
    #               'active_cases']
    # df_cases['date'] = df_cases['date'].apply(lambda x: x.replace('d', ''))
    # df_cases['date'] = df_cases['date'].apply(
    #     lambda x: datetime.datetime.strptime(str(x).replace("d", ""), '%Y%m%d'))
    #
    #
    #
    # print(number_inhabitants)
    # print(df_vaccination_rate)
    # print(df_cases)
    #
    # #calculate amount of vaccinated in recovered compartment
    # #dataframe including: vaccination rate and active cases
    # df_rec_and_vacc = pd.merge(df_vaccination_rate, df_cases, how='right', on='date')
    # df_rec_and_vacc = pd.merge(df_rec_and_vacc, df_recovered, how='left', on='date')
    # #drop unneccesary columns
    # df_rec_and_vacc = df_rec_and_vacc.drop(['vacc_efficiency', 'total_recovered', 'recovered', 'total_recovered_180'], axis=1)
    # #clean NaN with above value
    # for i in range(1, len(df_rec_and_vacc)):
    #     if np.isnan(df_rec_and_vacc.loc[i, 'vaccination_rate']):
    #         df_rec_and_vacc.loc[i, 'vaccination_rate'] = df_rec_and_vacc.loc[i - 1, 'vaccination_rate']
    # #clean NaN with 0
    # df_rec_and_vacc = df_rec_and_vacc.fillna(0)
    # #calculation
    # df_rec_and_vacc['share_vacc_in_rec'] = 0
    # df_rec_and_vacc['share_vacc_in_rec'] = df_rec_and_vacc.apply(
    #     lambda x : x['vaccination_rate']/(x['vaccination_rate'] + (1 - x['vaccination_rate'])*15.9296), axis=1)
    #
    # #calculate number of people that are vaccinated and recovered
    # df_rec_and_vacc['vacc_and_rec'] = 0
    # df_rec_and_vacc['vacc_and_rec'] = df_rec_and_vacc.apply(lambda x: x['share_vacc_in_rec'] * x['R'], axis=1)
    #
    # print(df_rec_and_vacc)
    # return(df_rec_and_vacc)

# also include the recovered compartment here, change to one input value
def get_starting_values(df, district): #(df_vaccination, df_rec_and_vacc):
    # R compartment
    # adding cumulated recovered for last 180 days column
    df['total_recovered_180'] = 0
    df['total_recovered_180'] = df['daily_rec'].rolling(180).sum()
    df['total_recovered_180'] = df.apply(lambda x: (x['cum_rec']) if (pd.isna(x['total_recovered_180'])) else x['total_recovered_180'], axis=1 )

    # calculate recovered including undetected cases (undetected rate of 1.35)
    df['R'] = 0
    df['R'] = df['total_recovered_180'] * 1.35

    # V compartment
    df_vaccination = df[['date', 'total_vacc', 'vacc_eff']]
    df_vaccination = df_vaccination.rename(columns={'total_vacc': 'number_people_vacc',
                                                    'vacc_eff': 'vacc_efficiency'})
    inhabitants = get_number_inhabitants(district)

    df_vaccination['vaccination_rate'] = 0
    df_vaccination['vaccination_rate'] = df_vaccination.apply(lambda x: x['number_people_vacc'] / inhabitants, axis=1)

    # calculation share of vacc in rec
    df_vaccination['share_vacc_in_rec'] = 0
    df_vaccination['share_vacc_in_rec'] = df_vaccination.apply(lambda x: x['vaccination_rate'] / (x['vaccination_rate'] + (1 - x['vaccination_rate']) * 15.9296), axis=1)

    # calculate number of people that are vaccinated and recovered
    df_vaccination['vacc_and_rec'] = 0
    df_vaccination['vacc_and_rec'] = df_vaccination['share_vacc_in_rec'] * df['R']

    # df_starting_values = df_rec_and_vacc
    df['V'] = 0
    for i in range(0, len(df)):
        if i <=14:
            df.loc[i, 'V'] = df.loc[i, 'total_vacc']
        elif i >14:
            df.loc[i, 'V'] = df.loc[i, 'total_vacc'] + (df_vaccination.loc[i-15, 'vacc_and_rec'] - df_vaccination.loc[i-14, 'vacc_and_rec'])

    #df_starting_values = df_starting_values.drop(['number_people_vacc', 'vaccination_rate', 'active_cases', 'share_vacc_in_rec', 'vacc_and_rec'], axis=1)


    return df

    #print(df_starting_values)

if __name__ == '__main__':
    df_vaccination = get_vaccination_data("Münster", "Nordrhein-Westfalen") #df_merged
    df_recovered = get_recovered_data('Münster')
    df_rec_and_vacc = get_vaccination_breakthrough(df_vaccination, df_recovered, 'Münster') #df_merged
    get_starting_values(df_vaccination, df_rec_and_vacc)
