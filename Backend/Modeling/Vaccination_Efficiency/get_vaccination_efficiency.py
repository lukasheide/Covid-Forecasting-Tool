import numpy as np
import requests
import datetime
import pandas as pd
# goal is to get an estimated vaccination efficiency for each district on all days since beginning 2020

#vaccination_efficiencies
eff_bt_secc_initial = 94
eff_mn_secc_initial = 92
eff_az_secc_initial = 67
eff_jj_secc_initial = 74

eff_bt_secc_delta = 89.8
eff_mn_secc_delta = 94.5
eff_az_secc_delta = 66.7
eff_jj_secc_delta = 65



# get a list with number vaccination/day on district level
# return: datum, kr_zweitimpf, kr_drittimpf, kreis, bundesland
def get_vaccination_number(district):
    headers = {
        'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
                         '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}

    response_vaccination = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?'
                                        'limit=1000&resource_id=df59e579-875d-497a-9eda-369722150d89&q=' + district
                                        , headers=headers)
    vaccination = response_vaccination.json()
    district_level = {}
    state_level = {}
    daily_sec_vacc = {}
    daily_third_vacc = {}
    final_data = []
    for rec in vaccination['result']['records']:
          # date_obj = datetime.datetime.strptime(rec['datum'], '%Y-%m-%d')
          # date_obj = date_obj + datetime.timedelta(days=14)
          # date = 'd' + date_obj.strftime('%Y%m%d')
          date = rec['datum']
          district_level[date] = rec['kreis']
          state_level[date] = rec['bundesland']
          daily_sec_vacc[date] = rec['kr_zweitimpf']
          daily_third_vacc[date] = rec['kr_drittimpf']

          final_data.append((date,
                             district_level.get(date, 0),
                             state_level.get(date, 0),
                             daily_sec_vacc.get(date, 0),
                             daily_third_vacc.get(date, 0)))

    df_vaccination = pd.DataFrame(final_data)
    df_vaccination.columns = ['date',
                  'district',
                  'state',
                  'daily_sec_vacc',
                  'daily_third_vacc']
    df_vaccination['date'] = pd.to_datetime(df_vaccination.date)
    df_vaccination = df_vaccination.sort_values(by='date')
    print(df_vaccination)
    return df_vaccination

# get a list with number of vaccination/manufacturer on state level
# return: datum, bundesland, BL_ZWEITIMPF (Corona-Zweitimpfungen), BL_ZWEITIMPF_HS_BT (Biontech),
# BL_ZWEITIMPF_HS_MN (Moderna), BL_ZWEITIMPF_HS_AZ (Astra), BL_ZWEITIMPF_HS_JJ (Johnson), BL_DRITTIMPF (Booster),
# BL_DRITTIMPF_HS_BT (Booster Biontech), BL_DRITTIMPF_HS_MN (Booster Moderna), BL_DRITTIMPF_HS_JJ (Booster Johnson)

def get_manufacturers(state):
    headers = {
        'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MzcwNzU4MzksImp0aSI6ImV3Tk96Z0M4ZEJXdmYtc'
                         '2wybDJfdS10NFY1Q0hySjlNamlsRElnVVdfODQifQ.q_YvSVMAed7MMZUi8om0UWla5YkPlCmckqGs_RHclfs'}
    response_manufacturer = requests.get('https://www.corona-datenplattform.de/api/3/action/datastore_search?'
                                         'limit=1000&resource_id=944a8719-a7b9-43f5-ac8f-7c0856978df3&q=' + state
                                         , headers=headers)
    manufacturers = response_manufacturer.json()
    state_level = {}
    daily_sec_vacc = {}
    daily_sec_vacc_bt = {}
    daily_sec_vacc_mn = {}
    daily_sec_vacc_az = {}
    daily_sec_vacc_jj = {}
    daily_third_vacc = {}
    daily_third_vacc_bt = {}
    daily_third_vacc_mn = {}
    daily_third_vacc_jj = {}
    final_data = []

    for rec in manufacturers['result']['records']:
          date = rec['datum']
          state_level[date] = rec['bundesland']
          daily_sec_vacc[date] = rec['bl_zweitimpf_vt']
          daily_sec_vacc_bt[date] = rec['bl_zweitimpf_hs_bt']
          daily_sec_vacc_mn[date] = rec['bl_zweitimpf_hs_mn']
          daily_sec_vacc_az[date] = rec['bl_zweitimpf_hs_az']
          daily_sec_vacc_jj[date] = rec['bl_zweitimpf_hs_jj']
          daily_third_vacc[date] = rec['bl_drittimpf_vt']
          daily_third_vacc_bt[date] = rec['bl_drittimpf_hs_bt']
          daily_third_vacc_mn[date] = rec['bl_drittimpf_hs_mn']
          daily_third_vacc_jj[date] = rec['bl_drittimpf_hs_jj']

          final_data.append((date,
                             state_level.get(date, 0),
                             daily_sec_vacc.get(date, 0),
                             daily_sec_vacc_bt.get(date, 0),
                             daily_sec_vacc_mn.get(date, 0),
                             daily_sec_vacc_az.get(date, 0),
                             daily_sec_vacc_jj.get(date, 0),
                             daily_third_vacc.get(date, 0),
                             daily_third_vacc_bt.get(date, 0),
                             daily_third_vacc_mn.get(date, 0),
                             daily_third_vacc_jj.get(date, 0)))

    df_manufacturer = pd.DataFrame(final_data)
    df_manufacturer.columns = ['date',
                  'state',
                  'daily_sec_vacc',
                  'daily_sec_vacc_bt',
                  'daily_sec_vacc_mn',
                  'daily_sec_vacc_az',
                  'daily_sec_vacc_jj',
                  'daily_third_vacc',
                  'daily_third_vacc_bt',
                  'daily_third_vacc_mn',
                  'daily_third_vacc_jj']
    df_manufacturer['date'] = pd.to_datetime(df_manufacturer.date)
    df_manufacturer = df_manufacturer.sort_values(by='date')
    print(df_manufacturer)
    return df_manufacturer

# merge of the created dataframes
# if no values are available, fill in 0 or take number from previous day, depending on logic
def merge_vaccinations_manufacturers(df_vaccination, df_manufacturer):
    df_merged = pd.merge(left=df_vaccination, right=df_manufacturer, how='left', left_on=['date', 'state'],
                         right_on=['date', 'state'])
    # fill with zero or previous values
    # the manufacturer data is discontinued after 2021-11-30
    df_merged = df_merged.fillna(method='ffill')
    #df_temp_part1 = df_merged.loc[df_merged['date'] < '2021-11-30',].fillna(0)
    #df_temp_part2 = df_merged.loc[df_merged['date'] >= '2021-11-30',].fillna(method='ffill')
    #df_merged = pd.concat([df_temp_part1, df_temp_part2], axis=0)

    print(df_merged)
    print(df_merged.keys())
    print(df_merged.dtypes)


    return df_merged

def get_manufacturer_share(df_merged):
    #calculte percentage of use of vaccination manufacturer for each day
    df_merged['total_vaccinations_all_manufacturers'] = ""
    df_merged['vacc_share_bt'] = 0
    df_merged['vacc_share_mn'] = 0
    df_merged['vacc_share_az'] = 0
    df_merged['vacc_share_jj'] = 0
    # sum up total numbers of all manufacturers
    for i in range(1, len(df_merged)):
        df_merged['total_vaccinations_all_manufacturers'] = df_merged['daily_sec_vacc_bt'] \
        + df_merged['daily_sec_vacc_mn'] + df_merged['daily_sec_vacc_az'] + df_merged['daily_sec_vacc_jj'] +\
        df_merged['daily_third_vacc_bt'] \
        + df_merged['daily_third_vacc_mn'] + df_merged['daily_third_vacc_jj']
        # divide by number of manufacturer and save in dataframe
        df_merged['vacc_share_bt'] = (df_merged['daily_sec_vacc_bt'] + df_merged['daily_third_vacc_bt']) / df_merged['total_vaccinations_all_manufacturers']
        df_merged['vacc_share_mn'] = (df_merged['daily_sec_vacc_mn'] + df_merged['daily_third_vacc_mn']) / df_merged[
            'total_vaccinations_all_manufacturers']
        df_merged['vacc_share_az'] = df_merged['daily_sec_vacc_az'] / df_merged['total_vaccinations_all_manufacturers']
        df_merged['vacc_share_jj'] = (df_merged['daily_sec_vacc_jj'] + df_merged['daily_third_vacc_jj']) / df_merged[
            'total_vaccinations_all_manufacturers']

        # if np.isnan(df_merged.loc[i,'vacc_share_bt']):
        #     df_merged.loc[i, 'vacc_share_bt'] = 0
        # if np.isnan(df_merged.loc[i,'vacc_share_mn']):
        #     df_merged.loc[i,'vacc_share_mn'] = 0
        # if np.isnan(df_merged.loc[i,'vacc_share_az']):
        #     df_merged.loc[i,'vacc_share_az'] = 0
        # if np.isnan(df_merged.loc[i,'vacc_share_jj']):
        #     df_merged.loc[i,'vacc_share_mn'] = 0


    print(df_merged)
    return df_merged

# distribute people into vaccination status groups (extend groups when extending complexity)
def get_vaccination_status(df_merged):
    #df_merged['people_sec_vacc_bt_1_week']= 0
    df_merged['people_sec_vacc_bt_2_weeks']= 0
    #df_merged['people_sec_vacc_mn_1_week']= ''
    df_merged['people_sec_vacc_mn_2_weeks']= 0
    #df_merged['people_sec_vacc_az_1_week']= ''
    df_merged['people_sec_vacc_az_2_weeks']= 0
    #df_merged['people_sec_vacc_jj_1_week']= ''
    df_merged['people_sec_vacc_jj_2_weeks']= 0

    # df_merged['people_sec_vacc_bt_1_week'] = np.NaN
    # df_merged['people_sec_vacc_mn_1_week'] = np.NaN
    # df_merged['people_sec_vacc_az_1_week'] = np.NaN
    # df_merged['people_sec_vacc_jj_1_week'] = np.NaN
    # for i in range(7, len(df_merged) + 1):
    #     df_merged.loc[i, 'people_sec_vacc_bt_1_week'] = df_merged.loc[i - 7, 'daily_sec_vacc_x'] * df_merged.loc[
    #         i - 7, 'vacc_share_bt']
    #     df_merged.loc[i, 'people_sec_vacc_mn_1_week'] = df_merged.loc[i - 7, 'daily_sec_vacc_x'] * df_merged.loc[
    #         i - 7, 'vacc_share_mn']
    #     df_merged.loc[i, 'people_sec_vacc_az_1_week'] = df_merged.loc[i - 7, 'daily_sec_vacc_x'] * df_merged.loc[
    #         i - 7, 'vacc_share_az']
    #     df_merged.loc[i, 'people_sec_vacc_jj_1_week'] = df_merged.loc[i - 7, 'daily_sec_vacc_x'] * df_merged.loc[
    #         i - 7, 'vacc_share_jj']

    df_merged['people_sec_vacc_bt_2_weeks'] = 0
    df_merged['people_sec_vacc_mn_2_weeks'] = 0
    df_merged['people_sec_vacc_az_2_weeks'] = 0
    df_merged['people_sec_vacc_jj_2_weeks'] = 0

    for i in range(14, len(df_merged)):
        df_merged.loc[i, 'people_sec_vacc_bt_2_weeks'] = df_merged.loc[i - 14, 'daily_sec_vacc_x'] * df_merged.loc[
            i - 14, 'vacc_share_bt']
        df_merged.loc[i, 'people_sec_vacc_mn_2_weeks'] = df_merged.loc[i - 14, 'daily_sec_vacc_x'] * df_merged.loc[
                i - 14, 'vacc_share_mn']
        df_merged.loc[i, 'people_sec_vacc_az_2_weeks'] = df_merged.loc[i - 14, 'daily_sec_vacc_x'] * df_merged.loc[
                i - 14, 'vacc_share_az']
        df_merged.loc[i, 'people_sec_vacc_jj_2_weeks'] = df_merged.loc[i - 14, 'daily_sec_vacc_x'] * df_merged.loc[
                i - 14, 'vacc_share_jj']
    print(df_merged)
    return df_merged

        #df_merged['daily_sec_vacc_x'] = df_merged.apply(lambda row: row['daily_sec_vacc_x'] * 2, axis=1)

#how to include efficiency data -> 1. step, hardcode
#1. assumption without variant
def get_vaccination_efficiency(df_merged):
    # calculate vaccination efficiency on any given day
    df_merged['total_people_sec_vacc'] = 0 # all people who received their second vacc at least 2 weeks ago so far
    df_merged['total_people_sec_vacc_corrected'] = 0 # all people who received their second vacc at least 2 weeks ago so far, excluding vacc > 180
    df_merged['vacc_efficiency_daily_new_vacc'] = 0  # vaccination efficiency of the people that got their shot two weeks ago today
    df_merged['help_value_efficiency_1'] = 0 # (vaccinated two weeks ago) * (vaccination efficiency of people vacc/day two weeks ago)
    df_merged['help_value_efficiency_1_sum'] = 0 # summed up version of help_value_efficiency_1
    df_merged['help_value_efficiency_1_sum_corrected'] = 0  # summed up version of help_value_efficiency_1, excluding vacc > 180
    df_merged['total_vacc_efficiency_today'] = 0  # overall vaccination efficiency of people vaccinated at least for two weeks
    df_merged['total_vacc_efficiency_today_corrected'] = 0 # overall vaccination efficiency of people vaccinated at least for two weeks, excluding vacc > 180 days

    # sum up the people who received their second vacc so far
    for x in range(1, len(df_merged)):
        df_merged.loc[x, 'total_people_sec_vacc'] = df_merged.loc[x-1,'total_people_sec_vacc'] + df_merged.loc[x, 'daily_sec_vacc_x']

    for z in range(1, len(df_merged)):
        if z < 180:
            df_merged.loc[z, 'total_people_sec_vacc_corrected'] = df_merged.loc[z - 1, 'total_people_sec_vacc_corrected'] + df_merged.loc[z, 'daily_sec_vacc_x']
        elif z >= 180:
            df_merged.loc[z, 'total_people_sec_vacc_corrected'] = df_merged.loc[z - 1, 'total_people_sec_vacc_corrected'] +  df_merged.loc[z, 'daily_sec_vacc_x'] - df_merged.loc[z - 180, 'daily_sec_vacc_x']

    print(df_merged)

    for i in range(14, len(df_merged)):
        df_merged.loc[i, 'vacc_efficiency_daily_new_vacc'] = (df_merged.loc[i, 'people_sec_vacc_bt_2_weeks'] * eff_bt_secc_initial + df_merged.loc[i, 'people_sec_vacc_mn_2_weeks'] * eff_mn_secc_initial + df_merged.loc[i, 'people_sec_vacc_az_2_weeks'] * eff_az_secc_initial + df_merged.loc[i, 'people_sec_vacc_jj_2_weeks'] * eff_jj_secc_initial) / df_merged.loc[i-14, 'daily_sec_vacc_x']
        #clean NaN values
        if np.isnan(df_merged.loc[i, 'vacc_efficiency_daily_new_vacc']):
            df_merged.loc[i, 'vacc_efficiency_daily_new_vacc'] = df_merged.loc[i-1, 'vacc_efficiency_daily_new_vacc']

    for y in range(14, len(df_merged)):
        df_merged.loc[y, 'help_value_efficiency_1'] = df_merged.loc[y-14, 'daily_sec_vacc_x'] * df_merged.loc[y
        , 'vacc_efficiency_daily_new_vacc']
        # clean NaN values from temp
        if np.isnan(df_merged.loc[y, 'help_value_efficiency_1']):
            df_merged.loc[y, 'help_value_efficiency_1'] = df_merged.loc[y-1, 'help_value_efficiency_1']

    for a in range(1, len(df_merged)):
        df_merged.loc[a, 'help_value_efficiency_1_sum'] = df_merged.loc[a - 1, 'help_value_efficiency_1_sum'] + df_merged.loc[a, 'help_value_efficiency_1']

    for c in range(1, len(df_merged)):
        if c < 180:
            df_merged.loc[c, 'help_value_efficiency_1_sum_corrected'] = df_merged.loc[c - 1, 'help_value_efficiency_1_sum_corrected'] + df_merged.loc[c, 'help_value_efficiency_1']
        if c >= 180:
            df_merged.loc[c, 'help_value_efficiency_1_sum_corrected'] = df_merged.loc[c - 1, 'help_value_efficiency_1_sum_corrected'] + df_merged.loc[c, 'help_value_efficiency_1'] - df_merged.loc[c - 180, 'help_value_efficiency_1']

    for b in range(14, len(df_merged)):
        df_merged.loc[b, 'total_vacc_efficiency_today'] = df_merged.loc[b, 'help_value_efficiency_1_sum'] / df_merged.loc[b-14, 'total_people_sec_vacc']

    for d in range(14, len(df_merged)):
        df_merged.loc[d, 'total_vacc_efficiency_today_corrected'] = df_merged.loc[d, 'help_value_efficiency_1_sum_corrected'] / df_merged.loc[d-14, 'total_people_sec_vacc_corrected']

    print(df_merged)
    return df_merged

def get_vaccinaction_efficiency_waning(df_merged):
    df_merged['people_sec_vacc_bt_within_1_month'] = 0
    df_merged['people_sec_vacc_bt_within_2_month'] = 0
    df_merged['people_sec_vacc_bt_within_3_month'] = 0
    df_merged['people_sec_vacc_bt_within_4_month'] = 0
    df_merged['people_sec_vacc_bt_within_5_month'] = 0
    df_merged['people_sec_vacc_bt_within_6_month'] = 0

    df_merged['people_sec_vacc_mn_within_1_month'] = 0
    df_merged['people_sec_vacc_mn_within_2_month'] = 0
    df_merged['people_sec_vacc_mn_within_3_month'] = 0
    df_merged['people_sec_vacc_mn_within_4_month'] = 0
    df_merged['people_sec_vacc_mn_within_5_month'] = 0
    df_merged['people_sec_vacc_mn_within_6_month'] = 0

    df_merged['people_sec_vacc_az_within_1_month'] = 0
    df_merged['people_sec_vacc_az_within_2_month'] = 0
    df_merged['people_sec_vacc_az_within_3_month'] = 0
    df_merged['people_sec_vacc_az_within_4_month'] = 0
    df_merged['people_sec_vacc_az_within_5_month'] = 0
    df_merged['people_sec_vacc_az_within_6_month'] = 0

    df_merged['people_sec_vacc_jj_within_1_month'] = 0
    df_merged['people_sec_vacc_jj_within_2_month'] = 0
    df_merged['people_sec_vacc_jj_within_3_month'] = 0
    df_merged['people_sec_vacc_jj_within_4_month'] = 0
    df_merged['people_sec_vacc_jj_within_5_month'] = 0
    df_merged['people_sec_vacc_jj_within_6_month'] = 0

    df_merged['people_vacc_last_180_days'] = 0
    df_merged['vacc_types_and_efficiency'] = 0
    df_merged['total_vacc_efficiency_today_corrected_waning'] = 0

    #clean NaN values from dataframe and substitue with 0 to enable following calculations
    df_merged = df_merged.fillna(0)

    for i in range(1, len(df_merged)):
        #vaccinated within last month (-2 weeks) -> 16 days
        if i < 16: # then all the values so fare can be added -> check with different example if it works!
            for a in range (0, i):
                df_merged.loc[i, 'people_sec_vacc_bt_within_1_month'] = df_merged.loc[i-1, 'people_sec_vacc_bt_within_1_month'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_1_month'] = df_merged.loc[i-1, 'people_sec_vacc_mn_within_1_month'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_1_month'] = df_merged.loc[i-1, 'people_sec_vacc_az_within_1_month'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_1_month'] = df_merged.loc[i-1, 'people_sec_vacc_jj_within_1_month'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']

        elif i >= 16:
            for x in range(0, 17): #16 days are looped through -> start at 0
                df_merged.loc[i, 'people_sec_vacc_bt_within_1_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_1_month'] + df_merged.loc[i - (16 - x), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_1_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_1_month'] + df_merged.loc[i - (16 - x), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_1_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_1_month'] + df_merged.loc[i - (16 - x), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_1_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_1_month'] + df_merged.loc[i - (16 - x), 'people_sec_vacc_jj_2_weeks']
        #vaccinated within two months ago
        if i < 46:
            for a in range (0, i-16):
                df_merged.loc[i, 'people_sec_vacc_bt_within_2_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_2_month'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_2_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_2_month'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_2_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_2_month'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_2_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_2_month'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
            #print(i)
        elif i >= 46:
            for b in range(0,31): #30 days are looped through
                df_merged.loc[i, 'people_sec_vacc_bt_within_2_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_2_month'] + df_merged.loc[i - (46 - b), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_2_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_2_month'] + df_merged.loc[i - (46 - b), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_2_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_2_month'] + df_merged.loc[i - (46 - b), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_2_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_2_month'] + df_merged.loc[i - (46 - b), 'people_sec_vacc_jj_2_weeks']
        #vaccinated within three months ago
        if i < 76:
            for a in range(0, i - 46):
                df_merged.loc[i, 'people_sec_vacc_bt_within_3_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_3_month'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_3_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_3_month'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_3_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_3_month'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_3_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_3_month'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
        elif i >= 76:
            for b in range(0,31): #30 days are looped through
                df_merged.loc[i, 'people_sec_vacc_bt_within_3_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_3_month'] + df_merged.loc[i - (76 - b), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_3_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_3_month'] + df_merged.loc[i - (76 - b), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_3_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_3_month'] + df_merged.loc[i - (76 - b), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_3_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_3_month'] + df_merged.loc[i - (76 - b), 'people_sec_vacc_jj_2_weeks']
        #vaccinated within four months ago
        if i < 106:
            for a in range(0, i - 76):
                df_merged.loc[i, 'people_sec_vacc_bt_within_4_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_4_month'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_4_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_4_month'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_4_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_4_month'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_4_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_4_month'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
        elif i >= 106:
            for b in range(0,31): #30 days are looped through
                df_merged.loc[i, 'people_sec_vacc_bt_within_4_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_4_month'] + df_merged.loc[i - (106 - b), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_4_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_4_month'] + df_merged.loc[i - (106 - b), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_4_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_4_month'] + df_merged.loc[i - (106 - b), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_4_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_4_month'] + df_merged.loc[i - (106 - b), 'people_sec_vacc_jj_2_weeks']
        #vaccinated within five months ago
        if i < 136:
            for a in range(0, i - 106):
                df_merged.loc[i, 'people_sec_vacc_bt_within_5_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_5_month'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_5_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_5_month'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_5_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_5_month'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_5_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_5_month'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
        elif i >= 136:
            for b in range(0,31): #30 days are looped through
                df_merged.loc[i, 'people_sec_vacc_bt_within_5_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_5_month'] + df_merged.loc[i - (136 - b), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_5_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_5_month'] + df_merged.loc[i - (136 - b), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_5_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_5_month'] + df_merged.loc[i - (136 - b), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_5_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_5_month'] + df_merged.loc[i - (136 - b), 'people_sec_vacc_jj_2_weeks']
        #vaccinated within six months ago
        if i < 166:
            for a in range(0, i - 136):
                df_merged.loc[i, 'people_sec_vacc_bt_within_6_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_6_month'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_6_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_6_month'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_6_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_6_month'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_6_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_6_month'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
        elif i >= 166:
            for b in range(0,31): #30 days are looped through
                df_merged.loc[i, 'people_sec_vacc_bt_within_6_month'] = df_merged.loc[i, 'people_sec_vacc_bt_within_6_month'] + df_merged.loc[i - (166 - b), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_mn_within_6_month'] = df_merged.loc[i, 'people_sec_vacc_mn_within_6_month'] + df_merged.loc[i - (166 - b), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_az_within_6_month'] = df_merged.loc[i, 'people_sec_vacc_az_within_6_month'] + df_merged.loc[i - (166 - b), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_6_month'] = df_merged.loc[i, 'people_sec_vacc_jj_within_6_month'] + df_merged.loc[i - (166 - b), 'people_sec_vacc_jj_2_weeks']

        #print(df_merged)

    for i in range(1, len(df_merged)):
        # calculate overall efficiency, including waning
        df_merged.loc[i, 'vacc_types_and_efficiency'] = \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_1_month'] * eff_bt_secc_initial) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_1_month'] * eff_mn_secc_initial) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_1_month'] * eff_az_secc_initial) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_1_month'] * eff_jj_secc_initial) + \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_2_month'] * (eff_bt_secc_initial * 0.92)) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_2_month'] * (eff_mn_secc_initial * 0.92)) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_2_month'] * (eff_az_secc_initial * 0.92)) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_2_month'] * (eff_jj_secc_initial * 0.92)) + \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_3_month'] * (eff_bt_secc_initial * 0.84)) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_3_month'] * (eff_mn_secc_initial * 0.84)) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_3_month'] * (eff_az_secc_initial * 0.84)) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_3_month'] * (eff_jj_secc_initial * 0.84)) + \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_4_month'] * (eff_bt_secc_initial * 0.76)) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_4_month'] * (eff_mn_secc_initial * 0.76)) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_4_month'] * (eff_az_secc_initial * 0.76)) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_4_month'] * (eff_jj_secc_initial * 0.76)) + \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_5_month'] * (eff_bt_secc_initial * 0.68)) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_5_month'] * (eff_mn_secc_initial * 0.68)) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_5_month'] * (eff_az_secc_initial * 0.68)) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_5_month'] * (eff_jj_secc_initial * 0.68)) + \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_6_month'] * (eff_bt_secc_initial * 0.60)) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_6_month'] * (eff_mn_secc_initial * 0.60)) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_6_month'] * (eff_az_secc_initial * 0.60)) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_6_month'] * (eff_jj_secc_initial * 0.60))

    #print(df_merged)
        # number_of_people_vacc = \
        #     df_merged.loc[i, 'people_sec_vacc_bt_within_1_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_mn_within_1_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_az_within_1_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_jj_within_1_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_bt_within_2_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_mn_within_2_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_az_within_2_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_jj_within_2_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_bt_within_3_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_mn_within_3_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_az_within_3_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_jj_within_3_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_bt_within_4_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_mn_within_4_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_az_within_4_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_jj_within_4_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_bt_within_5_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_mn_within_5_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_az_within_5_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_jj_within_5_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_bt_within_6_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_mn_within_6_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_az_within_6_month']  + \
        #     df_merged.loc[i, 'people_sec_vacc_jj_within_6_month']

    df_merged['people_vacc_last_180_days'] = df_merged.apply(lambda x: x['people_sec_vacc_bt_within_1_month'] + \
        x['people_sec_vacc_mn_within_1_month'] + \
        x['people_sec_vacc_az_within_1_month'] + \
        x['people_sec_vacc_jj_within_1_month'] + \
        x['people_sec_vacc_bt_within_2_month'] + \
        x['people_sec_vacc_mn_within_2_month'] + \
        x['people_sec_vacc_az_within_2_month'] + \
        x['people_sec_vacc_jj_within_2_month'] + \
        x['people_sec_vacc_bt_within_3_month'] + \
        x['people_sec_vacc_mn_within_3_month'] + \
        x['people_sec_vacc_az_within_3_month'] + \
        x['people_sec_vacc_jj_within_3_month'] + \
        x['people_sec_vacc_bt_within_4_month'] + \
        x['people_sec_vacc_mn_within_4_month'] + \
        x['people_sec_vacc_az_within_4_month'] + \
        x['people_sec_vacc_jj_within_4_month'] + \
        x['people_sec_vacc_bt_within_5_month'] + \
        x['people_sec_vacc_mn_within_5_month'] + \
        x['people_sec_vacc_az_within_5_month'] + \
        x['people_sec_vacc_jj_within_5_month'] + \
        x['people_sec_vacc_bt_within_6_month'] + \
        x['people_sec_vacc_mn_within_6_month'] + \
        x['people_sec_vacc_az_within_6_month'] + \
        x['people_sec_vacc_jj_within_6_month'], axis = 1)
    # df_merged['daily_sec_vacc_x'] = df_merged.apply(lambda row: row['daily_sec_vacc_x'] * 2, axis=1)
        #print(df_merged)
    for i in range(0, len(df_merged)):
        df_merged.loc[i, 'total_vacc_efficiency_today_corrected_waning'] = \
            df_merged.loc[i, 'vacc_types_and_efficiency'] / df_merged.loc[i, 'people_vacc_last_180_days']

    print(df_merged)
    return df_merged

def get_vaccinaction_efficiency_waning_booster(df_merged):
    df_merged['people_third_vacc_bt_today'] = 0 # number of people that got their third vacc with biontech today
    df_merged['people_third_vacc_mn_today'] = 0
    df_merged['people_third_vacc_jj_today'] = 0

    df_merged['people_sec_vacc_bt_within_1_month_booster'] = 0
    df_merged['people_third_vacc_bt_within_1_month_booster'] = 0
    df_merged['people_sec_vacc_mn_within_1_month_booster'] = 0
    df_merged['people_third_vacc_mn_within_1_month_booster'] = 0
    df_merged['people_sec_vacc_az_within_1_month_booster'] = 0
    df_merged['people_sec_vacc_jj_within_1_month_booster'] = 0
    df_merged['people_third_vacc_jj_within_1_month_booster'] = 0

    df_merged['people_sec_vacc_bt_within_2_month_booster'] = 0
    df_merged['people_third_vacc_bt_within_2_month_booster'] = 0
    df_merged['people_sec_vacc_mn_within_2_month_booster'] = 0
    df_merged['people_third_vacc_mn_within_2_month_booster'] = 0
    df_merged['people_sec_vacc_az_within_2_month_booster'] = 0
    df_merged['people_sec_vacc_jj_within_2_month_booster'] = 0
    df_merged['people_third_vacc_jj_within_2_month_booster'] = 0

    df_merged['people_sec_vacc_bt_within_3_month_booster'] = 0
    df_merged['people_third_vacc_bt_within_3_month_booster'] = 0
    df_merged['people_sec_vacc_mn_within_3_month_booster'] = 0
    df_merged['people_third_vacc_mn_within_3_month_booster'] = 0
    df_merged['people_sec_vacc_az_within_3_month_booster'] = 0
    df_merged['people_sec_vacc_jj_within_3_month_booster'] = 0
    df_merged['people_third_vacc_jj_within_3_month_booster'] = 0

    df_merged['people_sec_vacc_bt_within_4_month_booster'] = 0
    df_merged['people_third_vacc_bt_within_4_month_booster'] = 0
    df_merged['people_sec_vacc_mn_within_4_month_booster'] = 0
    df_merged['people_third_vacc_mn_within_4_month_booster'] = 0
    df_merged['people_sec_vacc_az_within_4_month_booster'] = 0
    df_merged['people_sec_vacc_jj_within_4_month_booster'] = 0
    df_merged['people_third_vacc_jj_within_4_month_booster'] = 0

    df_merged['people_sec_vacc_bt_within_5_month_booster'] = 0
    df_merged['people_third_vacc_bt_within_5_month_booster'] = 0
    df_merged['people_sec_vacc_mn_within_5_month_booster'] = 0
    df_merged['people_third_vacc_mn_within_5_month_booster'] = 0
    df_merged['people_sec_vacc_az_within_5_month_booster'] = 0
    df_merged['people_sec_vacc_jj_within_5_month_booster'] = 0
    df_merged['people_third_vacc_jj_within_5_month_booster'] = 0

    df_merged['people_sec_vacc_bt_within_6_month_booster'] = 0
    df_merged['people_third_vacc_bt_within_6_month_booster'] = 0
    df_merged['people_sec_vacc_mn_within_6_month_booster'] = 0
    df_merged['people_third_vacc_mn_within_6_month_booster'] = 0
    df_merged['people_sec_vacc_az_within_6_month_booster'] = 0
    df_merged['people_sec_vacc_jj_within_6_month_booster'] = 0
    df_merged['people_third_vacc_jj_within_6_month_booster'] = 0

    df_merged['share_people_secc_vacc_bt_today'] = 0
    df_merged['share_people_secc_vacc_mn_today'] = 0
    df_merged['share_people_secc_vacc_az_today'] = 0
    df_merged['share_people_secc_vacc_jj_today'] = 0

    df_merged['help_value_substraction_booster_bt'] = 0
    df_merged['help_value_substraction_booster_mn'] = 0
    df_merged['help_value_substraction_booster_az'] = 0
    df_merged['help_value_substraction_booster_jj'] = 0

    df_merged['vacc_types_and_efficiency_booster'] = 0
    df_merged['people_vacc_last_180_days'] = 0
    df_merged['total_vacc_efficiency_today_corrected_waning_booster'] = 0

    df_merged['people_third_vacc_bt_today'] = df_merged.apply(lambda x: x['daily_third_vacc_x'] * x['vacc_share_bt'], axis=1)
    df_merged['people_third_vacc_mn_today'] = df_merged.apply(lambda x: x['daily_third_vacc_x'] * x['vacc_share_mn'], axis=1)
    df_merged['people_third_vacc_jj_today'] = df_merged.apply(lambda x: x['daily_third_vacc_x'] * x['vacc_share_jj'], axis=1)

    for i in range(1, len(df_merged)):
        # vaccinated within 1 months ago
        if i < 16:
            for a in range (0, i): # substract people that now count to the boostered -> betrifft nur Menschen aus 6 mon -> falsch möglicherweise auf mehr compartments
                df_merged.loc[i, 'people_sec_vacc_bt_within_1_month_booster'] = df_merged.loc[i - 1, 'people_sec_vacc_bt_within_1_month_booster'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_1_month_booster'] = df_merged.loc[i - 1, 'people_third_vacc_bt_within_1_month_booster'] + df_merged.loc[a, 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_1_month_booster'] = df_merged.loc[i - 1, 'people_sec_vacc_mn_within_1_month_booster'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_1_month_booster'] = df_merged.loc[i - 1, 'people_third_vacc_mn_within_1_month_booster'] + df_merged.loc[a, 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_1_month_booster'] = df_merged.loc[i - 1, 'people_sec_vacc_az_within_1_month_booster'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_1_month_booster'] = df_merged.loc[i - 1, 'people_sec_vacc_jj_within_1_month_booster'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_1_month_booster'] = df_merged.loc[i - 1, 'people_third_vacc_jj_within_1_month_booster'] + df_merged.loc[a, 'people_third_vacc_jj_today']
        elif i >= 16:
            for x in range(0, 17):
                df_merged.loc[i, 'people_sec_vacc_bt_within_1_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_1_month_booster'] + df_merged.loc[i - (16 - x), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_1_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_1_month_booster'] + df_merged.loc[i - (16 - x), 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_1_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_1_month_booster'] + df_merged.loc[i - (16 - x), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_1_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_1_month_booster'] + df_merged.loc[i - (16 - x), 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_1_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_1_month_booster'] + df_merged.loc[i - (16 - x), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_1_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_1_month_booster'] + df_merged.loc[i - (16 - x), 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_1_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_1_month_booster'] + df_merged.loc[i - (16 - x), 'people_third_vacc_jj_today']

        # vaccinated within two months ago
        if i < 46:
            for a in range(0, i - 16):
                df_merged.loc[i, 'people_sec_vacc_bt_within_2_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_2_month_booster'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_2_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_2_month_booster'] + df_merged.loc[a, 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_2_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_2_month_booster'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_2_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_2_month_booster'] + df_merged.loc[a, 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_2_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_2_month_booster'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_2_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_2_month_booster'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_2_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_2_month_booster'] + df_merged.loc[a, 'people_third_vacc_jj_today']

        elif i >= 46:
            for b in range(0, 31):  # 30 days are looped through
                df_merged.loc[i, 'people_sec_vacc_bt_within_2_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_2_month_booster'] + df_merged.loc[i - (46 - b), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_2_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_2_month_booster'] + df_merged.loc[i - (46 - b), 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_2_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_2_month_booster'] + df_merged.loc[i - (46 - b), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_2_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_2_month_booster'] + df_merged.loc[i - (46 - b), 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_2_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_2_month_booster'] + df_merged.loc[i - (46 - b), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_2_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_2_month_booster'] + df_merged.loc[i - (46 - b), 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_2_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_2_month_booster'] + df_merged.loc[i - (46 - b), 'people_third_vacc_jj_today']

        # vaccinated within three months ago
        if i < 76:
            for a in range(0, i - 46):
                df_merged.loc[i, 'people_sec_vacc_bt_within_3_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_3_month_booster'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_3_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_3_month_booster'] + df_merged.loc[a, 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_3_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_3_month_booster'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_3_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_3_month_booster'] + df_merged.loc[a, 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_3_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_3_month_booster'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_3_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_3_month_booster'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_3_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_3_month_booster'] + df_merged.loc[a, 'people_third_vacc_jj_today']
        elif i >= 76:
            for b in range(0, 31):  # 30 days are looped through
                df_merged.loc[i, 'people_sec_vacc_bt_within_3_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_3_month_booster'] + df_merged.loc[i - (76 - b), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_3_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_3_month_booster'] + df_merged.loc[i - (76 - b), 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_3_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_3_month_booster'] + df_merged.loc[i - (76 - b), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_3_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_3_month_booster'] + df_merged.loc[i - (76 - b), 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_3_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_3_month_booster'] + df_merged.loc[i - (76 - b), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_3_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_3_month_booster'] + df_merged.loc[i - (76 - b), 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_3_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_3_month_booster'] + df_merged.loc[i - (76 - b), 'people_third_vacc_jj_today']

        # vaccinated within four months ago
        if i < 106:
            for a in range(0, i - 76):
                df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_4_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_4_month_booster'] + df_merged.loc[a, 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_4_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_4_month_booster'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_4_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_4_month_booster'] + df_merged.loc[a, 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_4_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_4_month_booster'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_4_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_4_month_booster'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_4_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_4_month_booster'] + df_merged.loc[a, 'people_third_vacc_jj_today']
        elif i >= 106:
            for b in range(0, 31):  # 30 days are looped through
                df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] + df_merged.loc[i - (106 - b), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_4_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_4_month_booster'] + df_merged.loc[i - (106 - b), 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_4_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_4_month_booster'] + df_merged.loc[i - (106 - b), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_4_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_4_month_booster'] + df_merged.loc[i - (106 - b), 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_4_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_4_month_booster'] + df_merged.loc[i - (106 - b), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_4_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_4_month_booster'] + df_merged.loc[i - (106 - b), 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_4_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_4_month_booster'] + df_merged.loc[i - (106 - b), 'people_third_vacc_jj_today']

        # vaccinated within five months ago
        if i < 136:
            for a in range(0, i - 106):
                df_merged.loc[i, 'people_sec_vacc_bt_within_5_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_5_month_booster'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_5_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_5_month_booster'] + df_merged.loc[a, 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_5_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_5_month_booster'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_5_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_5_month_booster'] + df_merged.loc[a, 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_5_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_5_month_booster'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_5_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_5_month_booster'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_5_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_5_month_booster'] + df_merged.loc[a, 'people_third_vacc_jj_today']
        elif i >= 136:
            for b in range(0, 31):  # 30 days are looped through
                df_merged.loc[i, 'people_sec_vacc_bt_within_5_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_5_month_booster'] + df_merged.loc[i - (136 - b), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_5_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_5_month_booster'] + df_merged.loc[i - (136 - b), 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_5_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_5_month_booster'] + df_merged.loc[i - (136 - b), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_5_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_5_month_booster'] + df_merged.loc[i - (136 - b), 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_5_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_5_month_booster'] + df_merged.loc[i - (136 - b), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_5_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_5_month_booster'] + df_merged.loc[i - (136 - b), 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_5_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_5_month_booster'] + df_merged.loc[i - (136 - b), 'people_third_vacc_jj_today']

        # vaccinated within six months ago
        if i < 166:
            for a in range(0, i - 136):
                df_merged.loc[i, 'people_sec_vacc_bt_within_6_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_6_month_booster'] + df_merged.loc[a, 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_6_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_6_month_booster'] + df_merged.loc[a, 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_6_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_6_month_booster'] + df_merged.loc[a, 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_6_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_6_month_booster'] + df_merged.loc[a, 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_6_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_6_month_booster'] + df_merged.loc[a, 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_6_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_6_month_booster'] + df_merged.loc[a, 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_6_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_6_month_booster'] + df_merged.loc[a, 'people_third_vacc_jj_today']

        elif i >= 166:
            for b in range(0, 31):  # 30 days are looped through
                df_merged.loc[i, 'people_sec_vacc_bt_within_6_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_6_month_booster'] + df_merged.loc[i - ( 166 - b), 'people_sec_vacc_bt_2_weeks']
                df_merged.loc[i, 'people_third_vacc_bt_within_6_month_booster'] = df_merged.loc[i, 'people_third_vacc_bt_within_6_month_booster'] + df_merged.loc[i - (166 - b), 'people_third_vacc_bt_today']
                df_merged.loc[i, 'people_sec_vacc_mn_within_6_month_booster'] = df_merged.loc[i, 'people_sec_vacc_mn_within_6_month_booster'] + df_merged.loc[i - ( 166 - b), 'people_sec_vacc_mn_2_weeks']
                df_merged.loc[i, 'people_third_vacc_mn_within_6_month_booster'] = df_merged.loc[i, 'people_third_vacc_mn_within_6_month_booster'] + df_merged.loc[i - (166 - b), 'people_third_vacc_mn_today']
                df_merged.loc[i, 'people_sec_vacc_az_within_6_month_booster'] = df_merged.loc[i, 'people_sec_vacc_az_within_6_month_booster'] + df_merged.loc[i - ( 166 - b), 'people_sec_vacc_az_2_weeks']
                df_merged.loc[i, 'people_sec_vacc_jj_within_6_month_booster'] = df_merged.loc[i, 'people_sec_vacc_jj_within_6_month_booster'] + df_merged.loc[i - ( 166 - b), 'people_sec_vacc_jj_2_weeks']
                df_merged.loc[i, 'people_third_vacc_jj_within_6_month_booster'] = df_merged.loc[i, 'people_third_vacc_jj_within_6_month_booster'] + df_merged.loc[i - (166 - b), 'people_third_vacc_jj_today']

        # df_merged.loc[i, 'share_people_secc_vacc_bt_today'] = (df_merged.loc[i, 'people_sec_vacc_bt_within_1_month_booster'] + \
        # df_merged.loc[i, 'people_sec_vacc_bt_within_2_month_booster'] + \
        # df_merged.loc[i, 'people_sec_vacc_bt_within_3_month_booster'] + \
        # df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] + \
        # df_merged.loc[i, 'people_sec_vacc_bt_within_5_month_booster'] + \
        # df_merged.loc[i, 'people_sec_vacc_bt_within_6_month_booster']) / df_merged.loc[i, 'total_people_sec_vacc_corrected']
        # df_merged.loc[i, 'share_people_secc_vacc_mn_today'] = 0
        # df_merged.loc[i, 'share_people_secc_vacc_az_today'] = 0
        # df_merged.loc[i, 'share_people_secc_vacc_jj_today'] = 0

    df_merged['share_people_secc_vacc_bt_today'] = df_merged.apply(lambda x: (x['total_people_sec_vacc_corrected']) if x['total_people_sec_vacc_corrected'] == 0  else ((x['people_sec_vacc_bt_within_1_month_booster'] + \
        x['people_sec_vacc_bt_within_2_month_booster'] +
        x['people_sec_vacc_bt_within_3_month_booster'] + \
        x['people_sec_vacc_bt_within_4_month_booster'] + \
        x['people_sec_vacc_bt_within_5_month_booster'] + \
        x['people_sec_vacc_bt_within_6_month_booster']) / x['total_people_sec_vacc_corrected']), axis = 1)
    df_merged['share_people_secc_vacc_mn_today'] = df_merged.apply(
        lambda x: (x['total_people_sec_vacc_corrected']) if x['total_people_sec_vacc_corrected'] == 0  else (x['people_sec_vacc_mn_within_1_month_booster'] + \
                   x['people_sec_vacc_mn_within_2_month_booster'] + \
                   x['people_sec_vacc_mn_within_3_month_booster'] + \
                   x['people_sec_vacc_mn_within_4_month_booster'] + \
                   x['people_sec_vacc_mn_within_5_month_booster'] + \
                   x['people_sec_vacc_mn_within_6_month_booster']) / x['total_people_sec_vacc_corrected'], axis=1)
    df_merged['share_people_secc_vacc_az_today'] = df_merged.apply(
        lambda x: (x['total_people_sec_vacc_corrected']) if x['total_people_sec_vacc_corrected'] == 0  else (x['people_sec_vacc_az_within_1_month_booster'] + \
                   x['people_sec_vacc_az_within_2_month_booster'] +
                   x['people_sec_vacc_az_within_3_month_booster'] + \
                   x['people_sec_vacc_az_within_4_month_booster'] + \
                   x['people_sec_vacc_az_within_5_month_booster'] + \
                   x['people_sec_vacc_az_within_6_month_booster']) / x['total_people_sec_vacc_corrected'], axis=1)
    df_merged['share_people_secc_vacc_jj_today'] = df_merged.apply(
        lambda x: (x['total_people_sec_vacc_corrected']) if x['total_people_sec_vacc_corrected'] == 0  else (x['people_sec_vacc_jj_within_1_month_booster'] + \
                   x['people_sec_vacc_jj_within_2_month_booster'] + \
                   x['people_sec_vacc_jj_within_3_month_booster'] + \
                   x['people_sec_vacc_jj_within_4_month_booster'] + \
                   x['people_sec_vacc_jj_within_5_month_booster'] + \
                   x['people_sec_vacc_jj_within_6_month_booster']) / x['total_people_sec_vacc_corrected'], axis=1)

    df_merged['help_value_substraction_booster_bt'] = df_merged.apply(
        lambda x: x['share_people_secc_vacc_bt_today'] * x['daily_third_vacc_x'], axis=1
    )
    df_merged['help_value_substraction_booster_mn'] = df_merged.apply(
        lambda x: x['share_people_secc_vacc_mn_today'] * x['daily_third_vacc_x'], axis=1
    )
    df_merged['help_value_substraction_booster_az'] = df_merged.apply(
        lambda x: x['share_people_secc_vacc_az_today'] * x['daily_third_vacc_x'], axis=1
    )
    df_merged['help_value_substraction_booster_jj'] = df_merged.apply(
        lambda x: x['share_people_secc_vacc_jj_today'] * x['daily_third_vacc_x'], axis=1
    )

    for i in range(0, len(df_merged)):
        # biontech, temp_6_months, includes the rest of the substraction that could not be taken in 6 months compartment
        temp_6_months = df_merged.loc[i, 'people_sec_vacc_bt_within_6_month_booster'] - df_merged.loc[i, 'help_value_substraction_booster_bt']
        if temp_6_months < 0:
            # wert der abgezogen werden kann abziehen, has to be plus because value is negative
            df_merged.loc[i, 'people_sec_vacc_bt_within_6_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_6_month_booster']- (df_merged.loc[i, 'help_value_substraction_booster_bt'] + temp_6_months)
            #temp_5_months, includes the rest of the substraction that could not be taken in 5 months compartment
            temp_5_months = df_merged.loc[i, 'people_sec_vacc_bt_within_5_month_booster'] - temp_6_months
            if temp_5_months < 0:
                df_merged.loc[i, 'people_sec_vacc_bt_within_5_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_5_month_booster'] - (df_merged.loc[i, 'help_value_substraction_booster_bt'] + temp_5_months)
                temp_4_months = df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] - temp_5_months
                if temp_4_months < 0:
                    df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] = df_merged.loc[ i, 'people_sec_vacc_bt_within_4_month_booster'] - (df_merged.loc[i, 'help_value_substraction_booster_bt'] + temp_4_months)
                    temp_3_months = df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] - temp_4_months
                    if temp_3_months < 0:
                        df_merged.loc[i, 'people_sec_vacc_bt_within_3_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_3_month_booster'] - (df_merged.loc[i, 'help_value_substraction_booster_bt'] + temp_3_months)
                        temp_2_months = df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] - temp_3_months
                        if temp_2_months < 0:
                            df_merged.loc[i, 'people_sec_vacc_bt_within_2_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_2_month_booster'] - (df_merged.loc[i, 'help_value_substraction_booster_bt'] + temp_2_months)
                            temp_1_months = df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] - temp_2_months
                            if temp_1_months < 0:
                                df_merged.loc[i, 'people_sec_vacc_bt_within_1_month_booster'] = df_merged.loc[i, 'people_sec_vacc_bt_within_1_month_booster'] - (df_merged.loc[i, 'help_value_substraction_booster_bt'] + temp_1_months)
                            elif temp_1_months >= 0:
                                df_merged.loc[i, 'people_sec_vacc_bt_within_1_month_booster'] = temp_1_months
                        elif temp_2_months >= 0:
                            df_merged.loc[i, 'people_sec_vacc_bt_within_2_month_booster'] = temp_2_months
                    elif temp_3_months >= 0:
                        df_merged.loc[i, 'people_sec_vacc_bt_within_3_month_booster'] = temp_3_months
                elif temp_4_months >= 0:
                    df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] = temp_4_months
            elif temp_5_months >= 0:
                df_merged.loc[i, 'people_sec_vacc_bt_within_5_month_booster'] = temp_5_months
        elif temp_6_months >= 0:
            df_merged.loc[i, 'people_sec_vacc_bt_within_6_month_booster'] = temp_6_months

        # moderna, temp_6_months, includes the rest of the substraction that could not be taken in 6 months compartment
        temp_6_months = df_merged.loc[i, 'people_sec_vacc_mn_within_6_month_booster'] - df_merged.loc[
            i, 'help_value_substraction_booster_mn']
        if temp_6_months < 0:
            # wert der abgezogen werden kann abziehen, has to be plus because value is negative
            df_merged.loc[i, 'people_sec_vacc_mn_within_6_month_booster'] = df_merged.loc[
                                                                                i, 'people_sec_vacc_mn_within_6_month_booster'] - (
                                                                                        df_merged.loc[
                                                                                            i, 'help_value_substraction_booster_mn'] + temp_6_months)
            # temp_5_months, includes the rest of the substraction that could not be taken in 5 months compartment
            temp_5_months = df_merged.loc[i, 'people_sec_vacc_mn_within_5_month_booster'] - temp_6_months
            if temp_5_months < 0:
                df_merged.loc[i, 'people_sec_vacc_mn_within_5_month_booster'] = df_merged.loc[
                                                                                    i, 'people_sec_vacc_mn_within_5_month_booster'] - (
                                                                                            df_merged.loc[
                                                                                                i, 'help_value_substraction_booster_mn'] + temp_5_months)
                temp_4_months = df_merged.loc[i, 'people_sec_vacc_mn_within_4_month_booster'] - temp_5_months
                if temp_4_months < 0:
                    df_merged.loc[i, 'people_sec_vacc_mn_within_4_month_booster'] = df_merged.loc[
                                                                                        i, 'people_sec_vacc_mn_within_4_month_booster'] - (
                                                                                                df_merged.loc[
                                                                                                    i, 'help_value_substraction_booster_mn'] + temp_4_months)
                    temp_3_months = df_merged.loc[i, 'people_sec_vacc_mn_within_4_month_booster'] - temp_4_months
                    if temp_3_months < 0:
                        df_merged.loc[i, 'people_sec_vacc_mn_within_3_month_booster'] = df_merged.loc[
                                                                                            i, 'people_sec_vacc_mn_within_3_month_booster'] - (
                                                                                                    df_merged.loc[
                                                                                                        i, 'help_value_substraction_booster_mn'] + temp_3_months)
                        temp_2_months = df_merged.loc[
                                            i, 'people_sec_vacc_mn_within_4_month_booster'] - temp_3_months
                        if temp_2_months < 0:
                            df_merged.loc[i, 'people_sec_vacc_mn_within_2_month_booster'] = df_merged.loc[
                                                                                                i, 'people_sec_vacc_mn_within_2_month_booster'] - (
                                                                                                        df_merged.loc[
                                                                                                            i, 'help_value_substraction_booster_mn'] + temp_2_months)
                            temp_1_months = df_merged.loc[
                                                i, 'people_sec_vacc_mn_within_4_month_booster'] - temp_2_months
                            if temp_1_months < 0:
                                df_merged.loc[i, 'people_sec_vacc_mn_within_1_month_booster'] = df_merged.loc[
                                                                                                    i, 'people_sec_vacc_mn_within_1_month_booster'] - (
                                                                                                            df_merged.loc[
                                                                                                                i, 'help_value_substraction_booster_mn'] + temp_1_months)
                            elif temp_1_months >= 0:
                                df_merged.loc[i, 'people_sec_vacc_mn_within_1_month_booster'] = temp_1_months
                        elif temp_2_months >= 0:
                            df_merged.loc[i, 'people_sec_vacc_mn_within_2_month_booster'] = temp_2_months
                    elif temp_3_months >= 0:
                        df_merged.loc[i, 'people_sec_vacc_mn_within_3_month_booster'] = temp_3_months
                elif temp_4_months >= 0:
                    df_merged.loc[i, 'people_sec_vacc_mn_within_4_month_booster'] = temp_4_months
            elif temp_5_months >= 0:
                df_merged.loc[i, 'people_sec_vacc_mn_within_5_month_booster'] = temp_5_months
        elif temp_6_months >= 0:
            df_merged.loc[i, 'people_sec_vacc_mn_within_6_month_booster'] = temp_6_months

        # astra, temp_6_months, includes the rest of the substraction that could not be taken in 6 months compartment
        temp_6_months = df_merged.loc[i, 'people_sec_vacc_az_within_6_month_booster'] - df_merged.loc[
            i, 'help_value_substraction_booster_az']
        if temp_6_months < 0:
            # wert der abgezogen werden kann abziehen, has to be plus because value is negative
            df_merged.loc[i, 'people_sec_vacc_az_within_6_month_booster'] = df_merged.loc[
                                                                                i, 'people_sec_vacc_az_within_6_month_booster'] - (
                                                                                    df_merged.loc[
                                                                                        i, 'help_value_substraction_booster_az'] + temp_6_months)
            # temp_5_months, includes the rest of the substraction that could not be taken in 5 months compartment
            temp_5_months = df_merged.loc[i, 'people_sec_vacc_az_within_5_month_booster'] - temp_6_months
            if temp_5_months < 0:
                df_merged.loc[i, 'people_sec_vacc_az_within_5_month_booster'] = df_merged.loc[
                                                                                    i, 'people_sec_vacc_az_within_5_month_booster'] - (
                                                                                        df_merged.loc[
                                                                                            i, 'help_value_substraction_booster_az'] + temp_5_months)
                temp_4_months = df_merged.loc[i, 'people_sec_vacc_az_within_4_month_booster'] - temp_5_months
                if temp_4_months < 0:
                    df_merged.loc[i, 'people_sec_vacc_az_within_4_month_booster'] = df_merged.loc[
                                                                                        i, 'people_sec_vacc_az_within_4_month_booster'] - (
                                                                                            df_merged.loc[
                                                                                                i, 'help_value_substraction_booster_az'] + temp_4_months)
                    temp_3_months = df_merged.loc[
                                        i, 'people_sec_vacc_az_within_4_month_booster'] - temp_4_months
                    if temp_3_months < 0:
                        df_merged.loc[i, 'people_sec_vacc_az_within_3_month_booster'] = df_merged.loc[
                                                                                            i, 'people_sec_vacc_az_within_3_month_booster'] - (
                                                                                                df_merged.loc[
                                                                                                    i, 'help_value_substraction_booster_az'] + temp_3_months)
                        temp_2_months = df_merged.loc[
                                            i, 'people_sec_vacc_az_within_4_month_booster'] - temp_3_months
                        if temp_2_months < 0:
                            df_merged.loc[i, 'people_sec_vacc_az_within_2_month_booster'] = df_merged.loc[
                                                                                                i, 'people_sec_vacc_az_within_2_month_booster'] - (
                                                                                                    df_merged.loc[
                                                                                                        i, 'help_value_substraction_booster_az'] + temp_2_months)
                            temp_1_months = df_merged.loc[
                                                i, 'people_sec_vacc_az_within_4_month_booster'] - temp_2_months
                            if temp_1_months < 0:
                                df_merged.loc[i, 'people_sec_vacc_az_within_1_month_booster'] = df_merged.loc[
                                                                                                    i, 'people_sec_vacc_az_within_1_month_booster'] - (
                                                                                                        df_merged.loc[
                                                                                                            i, 'help_value_substraction_booster_az'] + temp_1_months)
                            elif temp_1_months >= 0:
                                df_merged.loc[i, 'people_sec_vacc_az_within_1_month_booster'] = temp_1_months
                        elif temp_2_months >= 0:
                            df_merged.loc[i, 'people_sec_vacc_az_within_2_month_booster'] = temp_2_months
                    elif temp_3_months >= 0:
                        df_merged.loc[i, 'people_sec_vacc_az_within_3_month_booster'] = temp_3_months
                elif temp_4_months >= 0:
                    df_merged.loc[i, 'people_sec_vacc_az_within_4_month_booster'] = temp_4_months
            elif temp_5_months >= 0:
                df_merged.loc[i, 'people_sec_vacc_az_within_5_month_booster'] = temp_5_months
        elif temp_6_months >= 0:
            df_merged.loc[i, 'people_sec_vacc_az_within_6_month_booster'] = temp_6_months

        # johnson&johnson, temp_6_months, includes the rest of the substraction that could not be taken in 6 months compartment
        temp_6_months = df_merged.loc[i, 'people_sec_vacc_jj_within_6_month_booster'] - df_merged.loc[
            i, 'help_value_substraction_booster_jj']
        if temp_6_months < 0:
            # wert der abgezogen werden kann abziehen, has to be plus because value is negative
            df_merged.loc[i, 'people_sec_vacc_jj_within_6_month_booster'] = df_merged.loc[
                                                                                i, 'people_sec_vacc_jj_within_6_month_booster'] - (
                                                                                    df_merged.loc[
                                                                                        i, 'help_value_substraction_booster_jj'] + temp_6_months)
            # temp_5_months, includes the rest of the substraction that could not be taken in 5 months compartment
            temp_5_months = df_merged.loc[i, 'people_sec_vacc_jj_within_5_month_booster'] - temp_6_months
            if temp_5_months < 0:
                df_merged.loc[i, 'people_sec_vacc_jj_within_5_month_booster'] = df_merged.loc[
                                                                                    i, 'people_sec_vacc_jj_within_5_month_booster'] - (
                                                                                        df_merged.loc[
                                                                                            i, 'help_value_substraction_booster_jj'] + temp_5_months)
                temp_4_months = df_merged.loc[i, 'people_sec_vacc_jj_within_4_month_booster'] - temp_5_months
                if temp_4_months < 0:
                    df_merged.loc[i, 'people_sec_vacc_jj_within_4_month_booster'] = df_merged.loc[
                                                                                        i, 'people_sec_vacc_jj_within_4_month_booster'] - (
                                                                                            df_merged.loc[
                                                                                                i, 'help_value_substraction_booster_jj'] + temp_4_months)
                    temp_3_months = df_merged.loc[
                                        i, 'people_sec_vacc_jj_within_4_month_booster'] - temp_4_months
                    if temp_3_months < 0:
                        df_merged.loc[i, 'people_sec_vacc_jj_within_3_month_booster'] = df_merged.loc[
                                                                                            i, 'people_sec_vacc_jj_within_3_month_booster'] - (
                                                                                                df_merged.loc[
                                                                                                    i, 'help_value_substraction_booster_jj'] + temp_3_months)
                        temp_2_months = df_merged.loc[
                                            i, 'people_sec_vacc_jj_within_4_month_booster'] - temp_3_months
                        if temp_2_months < 0:
                            df_merged.loc[i, 'people_sec_vacc_jj_within_2_month_booster'] = df_merged.loc[
                                                                                                i, 'people_sec_vacc_jj_within_2_month_booster'] - (
                                                                                                    df_merged.loc[
                                                                                                        i, 'help_value_substraction_booster_jj'] + temp_2_months)
                            temp_1_months = df_merged.loc[
                                                i, 'people_sec_vacc_jj_within_4_month_booster'] - temp_2_months
                            if temp_1_months < 0:
                                df_merged.loc[i, 'people_sec_vacc_jj_within_1_month_booster'] = df_merged.loc[
                                                                                                    i, 'people_sec_vacc_jj_within_1_month_booster'] - (
                                                                                                        df_merged.loc[
                                                                                                            i, 'help_value_substraction_booster_jj'] + temp_1_months)
                            elif temp_1_months >= 0:
                                df_merged.loc[i, 'people_sec_vacc_jj_within_1_month_booster'] = temp_1_months
                        elif temp_2_months >= 0:
                            df_merged.loc[i, 'people_sec_vacc_jj_within_2_month_booster'] = temp_2_months
                    elif temp_3_months >= 0:
                        df_merged.loc[i, 'people_sec_vacc_jj_within_3_month_booster'] = temp_3_months
                elif temp_4_months >= 0:
                    df_merged.loc[i, 'people_sec_vacc_jj_within_4_month_booster'] = temp_4_months
            elif temp_5_months >= 0:
                df_merged.loc[i, 'people_sec_vacc_jj_within_5_month_booster'] = temp_5_months
        elif temp_6_months >= 0:
            df_merged.loc[i, 'people_sec_vacc_jj_within_6_month_booster'] = temp_6_months

    for i in range(1, len(df_merged)):
        # calculate overall efficiency, including waning and voc
        if df_merged.loc[i, 'date'] < pd.Timestamp('2021-06-30'):
            efficiency_biontech = 94
            efficiency_moderna = 92
            efficiency_astra = 67
            efficiency_johnson = 74
        else:
            efficiency_biontech = 89.8
            efficiency_moderna = 94.5
            efficiency_astra = 66.7
            efficiency_johnson = 65


        df_merged.loc[i, 'vacc_types_and_efficiency_booster'] = \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_1_month_booster'] * efficiency_biontech) + \
            (df_merged.loc[i, 'people_third_vacc_bt_within_1_month_booster'] * efficiency_biontech) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_1_month_booster'] * efficiency_moderna) + \
            (df_merged.loc[i, 'people_third_vacc_mn_within_1_month_booster'] * efficiency_moderna) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_1_month_booster'] * efficiency_astra) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_1_month_booster'] * efficiency_johnson) + \
            (df_merged.loc[i, 'people_third_vacc_jj_within_1_month_booster'] * efficiency_johnson) + \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_2_month_booster'] * (efficiency_biontech * 0.92)) + \
            (df_merged.loc[i, 'people_third_vacc_bt_within_2_month_booster'] * (efficiency_biontech * 0.92)) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_2_month_booster'] * (efficiency_moderna * 0.92)) + \
            (df_merged.loc[i, 'people_third_vacc_mn_within_2_month_booster'] * (efficiency_moderna * 0.92)) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_2_month_booster'] * (efficiency_astra * 0.92)) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_2_month_booster'] * (efficiency_johnson * 0.92)) + \
            (df_merged.loc[i, 'people_third_vacc_jj_within_2_month_booster'] * (efficiency_johnson * 0.92)) + \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_3_month_booster'] * (efficiency_biontech * 0.84)) + \
            (df_merged.loc[i, 'people_third_vacc_bt_within_3_month_booster'] * (efficiency_biontech * 0.84)) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_3_month_booster'] * (efficiency_moderna * 0.84)) + \
            (df_merged.loc[i, 'people_third_vacc_mn_within_3_month_booster'] * (efficiency_moderna * 0.84)) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_3_month_booster'] * (efficiency_astra * 0.84)) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_3_month_booster'] * (efficiency_johnson * 0.84)) + \
            (df_merged.loc[i, 'people_third_vacc_jj_within_3_month_booster'] * (efficiency_johnson * 0.84)) + \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_4_month_booster'] * (efficiency_biontech * 0.76)) + \
            (df_merged.loc[i, 'people_third_vacc_bt_within_4_month_booster'] * (efficiency_biontech * 0.76)) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_4_month_booster'] * (efficiency_moderna * 0.76)) + \
            (df_merged.loc[i, 'people_third_vacc_mn_within_4_month_booster'] * (efficiency_moderna * 0.76)) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_4_month_booster'] * (efficiency_astra * 0.76)) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_4_month_booster'] * (efficiency_johnson * 0.76)) + \
            (df_merged.loc[i, 'people_third_vacc_jj_within_4_month_booster'] * (efficiency_johnson * 0.76)) + \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_5_month_booster'] * (efficiency_biontech * 0.68)) + \
            (df_merged.loc[i, 'people_third_vacc_bt_within_5_month_booster'] * (efficiency_biontech * 0.68)) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_5_month_booster'] * (efficiency_moderna * 0.68)) + \
            (df_merged.loc[i, 'people_third_vacc_mn_within_5_month_booster'] * (efficiency_moderna * 0.68)) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_5_month_booster'] * (efficiency_astra * 0.68)) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_5_month_booster'] * (efficiency_johnson * 0.68)) + \
            (df_merged.loc[i, 'people_third_vacc_jj_within_5_month_booster'] * (efficiency_johnson * 0.68)) + \
            (df_merged.loc[i, 'people_sec_vacc_bt_within_6_month_booster'] * (efficiency_biontech * 0.60)) + \
            (df_merged.loc[i, 'people_third_vacc_bt_within_6_month_booster'] * (efficiency_biontech * 0.60)) + \
            (df_merged.loc[i, 'people_sec_vacc_mn_within_6_month_booster'] * (efficiency_moderna * 0.60)) + \
            (df_merged.loc[i, 'people_third_vacc_mn_within_6_month_booster'] * (efficiency_moderna * 0.60)) + \
            (df_merged.loc[i, 'people_sec_vacc_az_within_6_month_booster'] * (efficiency_astra * 0.60)) + \
            (df_merged.loc[i, 'people_sec_vacc_jj_within_6_month_booster'] * (efficiency_johnson * 0.60)) + \
            (df_merged.loc[i, 'people_third_vacc_jj_within_6_month_booster'] * (efficiency_johnson * 0.60))

        df_merged['people_vacc_last_180_days_booster'] = df_merged.apply(lambda x: x['people_sec_vacc_bt_within_1_month_booster'] + \
                                                                                   x['people_third_vacc_bt_within_1_month_booster']  + \
                                                                                   x['people_sec_vacc_mn_within_1_month_booster'] + \
                                                                                   x['people_third_vacc_mn_within_1_month_booster'] + \
                                                                                   x['people_sec_vacc_az_within_1_month_booster'] + \
                                                                                   x['people_sec_vacc_jj_within_1_month_booster'] + \
                                                                                   x['people_third_vacc_jj_within_1_month_booster'] + \
                                                                                   x['people_sec_vacc_bt_within_2_month_booster'] + \
                                                                                   x['people_third_vacc_bt_within_2_month_booster'] + \
                                                                                   x['people_sec_vacc_mn_within_2_month_booster'] + \
                                                                                   x['people_third_vacc_mn_within_2_month_booster'] + \
                                                                                   x['people_sec_vacc_az_within_2_month_booster'] + \
                                                                                   x['people_sec_vacc_jj_within_2_month_booster'] + \
                                                                                   x['people_third_vacc_jj_within_2_month_booster'] + \
                                                                                   x['people_sec_vacc_bt_within_3_month_booster'] + \
                                                                                   x['people_third_vacc_bt_within_3_month_booster'] + \
                                                                                   x['people_sec_vacc_mn_within_3_month_booster'] + \
                                                                                   x['people_third_vacc_mn_within_3_month_booster'] + \
                                                                                   x['people_sec_vacc_az_within_3_month_booster'] + \
                                                                                   x['people_sec_vacc_jj_within_3_month_booster'] + \
                                                                                   x['people_third_vacc_jj_within_3_month_booster'] + \
                                                                                   x['people_sec_vacc_bt_within_4_month_booster'] + \
                                                                                   x['people_third_vacc_bt_within_4_month_booster'] + \
                                                                                   x['people_sec_vacc_mn_within_4_month_booster'] + \
                                                                                   x['people_third_vacc_mn_within_4_month_booster'] + \
                                                                                   x['people_sec_vacc_az_within_4_month_booster'] + \
                                                                                   x['people_sec_vacc_jj_within_4_month_booster'] + \
                                                                                   x['people_third_vacc_jj_within_4_month_booster'] + \
                                                                                   x['people_sec_vacc_bt_within_5_month_booster'] + \
                                                                                   x['people_third_vacc_bt_within_5_month_booster'] + \
                                                                                   x['people_sec_vacc_mn_within_5_month_booster'] + \
                                                                                   x['people_third_vacc_mn_within_5_month_booster'] + \
                                                                                   x['people_sec_vacc_az_within_5_month_booster'] + \
                                                                                   x['people_sec_vacc_jj_within_5_month_booster'] + \
                                                                                   x['people_third_vacc_jj_within_5_month_booster'] + \
                                                                                   x['people_sec_vacc_bt_within_6_month_booster'] + \
                                                                                   x['people_third_vacc_bt_within_6_month_booster'] + \
                                                                                   x['people_sec_vacc_mn_within_6_month_booster'] + \
                                                                                   x['people_third_vacc_mn_within_6_month_booster'] + \
                                                                                   x['people_sec_vacc_az_within_6_month_booster'] + \
                                                                                   x['people_sec_vacc_jj_within_6_month_booster'] + \
                                                                                   x['people_third_vacc_jj_within_6_month_booster'],

                                                                 axis=1)


    for i in range(0, len(df_merged)):
        df_merged.loc[i, 'total_vacc_efficiency_today_corrected_waning_booster'] = \
        df_merged.loc[i, 'vacc_types_and_efficiency_booster'] / df_merged.loc[i, 'people_vacc_last_180_days_booster']

    print(df_merged)
    return df_merged


if __name__ == '__main__':
    df_vaccination = get_vaccination_number("Münster")
    df_manufacturer = get_manufacturers("Nordrhein-Westfalen")
    df_merged = merge_vaccinations_manufacturers(df_vaccination, df_manufacturer)
    df_merged = get_manufacturer_share(df_merged)
    df_merged = get_vaccination_status(df_merged)
    df_merged = get_vaccination_efficiency(df_merged)
    df_merged = get_vaccinaction_efficiency_waning(df_merged)
    get_vaccinaction_efficiency_waning_booster(df_merged)
