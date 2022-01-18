import pandas as pd
import numpy as np
import time

from Backend.Modeling.Vaccination_Efficiency.get_vaccination_efficiency import get_vaccination_number


def get_vaccination_effectiveness():

    # 1) Import District Vaccination Data from API:
    df = get_vaccination_number('Münster')

    start = time.time()

    # 2) Create variable with lagged second vaccinations:
    df['SecVaccStatus'] = df['daily_sec_vacc'].shift(14, fill_value=0)

    # 3) Cumulate Second Vacinations and Booster Vaccinations:
    df['SecVaccStatus_cum'] = df['SecVaccStatus'].cumsum()
    df['Booster_cum'] = df['daily_third_vacc'].cumsum()


    # 4) Compute cohorts: (Group vaccinated individuals into groups depending on vaccination round and duration since vaccination)
    ## A) Second Vaccinations:
    # Description: Rolling rolls is a sliding window approach that rolls over the 30 previous entries:
    # This means that here the previous 30 days, including the current day are considered and their sum is computed.
    # minus 1 month = last month
    df['SecVaccStatus_minus_1_month_cohort'] = df['SecVaccStatus'].rolling(30).sum()

    # Now the the next cohort needs to be delayed by 7 days:
    df['SecVaccStatus_minus_2_month_cohort'] = df['SecVaccStatus'].shift(30, fill_value=0).rolling(30).sum()

    # The other cohorts:
    df['SecVaccStatus_minus_3_month_cohort'] = df['SecVaccStatus'].shift(30 * 2, fill_value=0).rolling(30).sum()
    df['SecVaccStatus_minus_4_month_cohort'] = df['SecVaccStatus'].shift(30 * 3, fill_value=0).rolling(30).sum()
    df['SecVaccStatus_minus_5_month_cohort'] = df['SecVaccStatus'].shift(30 * 4, fill_value=0).rolling(30).sum()
    df['SecVaccStatus_minus_6_month_cohort'] = df['SecVaccStatus'].shift(30 * 5, fill_value=0).rolling(30).sum()

    ## B) Third vaccinations:
    df['Booster_minus_1_month_cohort'] = df['daily_third_vacc'].shift(30 * 0, fill_value=0).rolling(30).sum()
    df['Booster_minus_2_month_cohort'] = df['daily_third_vacc'].shift(30 * 1, fill_value=0).rolling(30).sum()
    df['Booster_minus_3_month_cohort'] = df['daily_third_vacc'].shift(30 * 2, fill_value=0).rolling(30).sum()
    df['Booster_minus_4_month_cohort'] = df['daily_third_vacc'].shift(30 * 3, fill_value=0).rolling(30).sum()
    df['Booster_minus_5_month_cohort'] = df['daily_third_vacc'].shift(30 * 4, fill_value=0).rolling(30).sum()
    df['Booster_minus_6_month_cohort'] = df['daily_third_vacc'].shift(30 * 5, fill_value=0).rolling(30).sum()


    # 5) Now we needed to subtract the booster vaccinations to get the corrected status for people with seconds vacinations:
    # Idea: Subtract individuals that were boostered from group: (Oldest ones are vaccinated first)
    df['SecVaccStatus_minus_1_month_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_1_month_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(30*1, fill_value=0), 0))
    df['SecVaccStatus_minus_2_month_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_2_month_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(30*2, fill_value=0), 0))
    df['SecVaccStatus_minus_3_month_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_3_month_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(30*3, fill_value=0), 0))
    df['SecVaccStatus_minus_4_month_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_4_month_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(30*4, fill_value=0), 0))
    df['SecVaccStatus_minus_5_month_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_5_month_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(30*5, fill_value=0), 0))
    df['SecVaccStatus_minus_6_month_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_6_month_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(30*6, fill_value=0), 0))


    # for debugging purposes:
    temp = df[['date', 'daily_sec_vacc', 'SecVaccStatus', 'daily_third_vacc', 'Booster_cum', 'SecVaccStatus_minus_6_month_cohort', 'SecVaccStatus_minus_6_month_cohort_CORRECTED']]


    # 6) Compute vaccination efficiency:
    eff_vacc_initial, eff_vacc_delta = get_efficiencies_per_cohort()

    # Get number of people with second vaccination as their status at time t:
    df['total_second_vacc_status'] = np.sum(
        df[['SecVaccStatus_minus_1_month_cohort_CORRECTED',
            'SecVaccStatus_minus_2_month_cohort_CORRECTED',
            'SecVaccStatus_minus_3_month_cohort_CORRECTED',
            'SecVaccStatus_minus_4_month_cohort_CORRECTED',
            'SecVaccStatus_minus_5_month_cohort_CORRECTED',
            'SecVaccStatus_minus_6_month_cohort_CORRECTED',
        ]], axis=1
    )

    # Get number of people with third vaccination as their status at time t:
    df['total_booster_vacc'] = np.sum(
        df[['Booster_minus_1_month_cohort',
            'Booster_minus_2_month_cohort',
            'Booster_minus_3_month_cohort',
            'Booster_minus_4_month_cohort',
            'Booster_minus_5_month_cohort',
            'Booster_minus_6_month_cohort',
        ]], axis=1
    )

    # Get number of people with vaccinated as their status in time t:
    df['total_vacc'] = df['total_second_vacc_status'] + df['total_booster_vacc']

    #change into lambda function in order to check date to include influence of VOC

    df['vacc_eff'] = df.apply(lambda x: (((x['SecVaccStatus_minus_1_month_cohort_CORRECTED'] * eff_vacc_initial[0] +
        x['SecVaccStatus_minus_2_month_cohort_CORRECTED'] * eff_vacc_initial[1] +
        x['SecVaccStatus_minus_3_month_cohort_CORRECTED'] * eff_vacc_initial[2] +
        x['SecVaccStatus_minus_4_month_cohort_CORRECTED'] * eff_vacc_initial[3] +
        x['SecVaccStatus_minus_5_month_cohort_CORRECTED'] * eff_vacc_initial[4] +
        x['SecVaccStatus_minus_6_month_cohort_CORRECTED'] * eff_vacc_initial[5] +

        x['Booster_minus_1_month_cohort'] * eff_vacc_initial[0] +
        x['Booster_minus_2_month_cohort'] * eff_vacc_initial[1] +
        x['Booster_minus_3_month_cohort'] * eff_vacc_initial[2] +
        x['Booster_minus_4_month_cohort'] * eff_vacc_initial[3] +
        x['Booster_minus_5_month_cohort'] * eff_vacc_initial[4] +
        x['Booster_minus_6_month_cohort'] * eff_vacc_initial[5]
    ) / x['total_vacc']) if (x['date'] < pd.Timestamp('2021-06-30')) else
        ((x['SecVaccStatus_minus_1_month_cohort_CORRECTED'] * eff_vacc_initial[0] +
        x['SecVaccStatus_minus_2_month_cohort_CORRECTED'] * eff_vacc_delta[1] +
        x['SecVaccStatus_minus_3_month_cohort_CORRECTED'] * eff_vacc_delta[2] +
        x['SecVaccStatus_minus_4_month_cohort_CORRECTED'] * eff_vacc_delta[3] +
        x['SecVaccStatus_minus_5_month_cohort_CORRECTED'] * eff_vacc_delta[4] +
        x['SecVaccStatus_minus_6_month_cohort_CORRECTED'] * eff_vacc_delta[5] +

        x['Booster_minus_1_month_cohort'] * eff_vacc_delta[0] +
        x['Booster_minus_2_month_cohort'] * eff_vacc_delta[1] +
        x['Booster_minus_3_month_cohort'] * eff_vacc_delta[2] +
        x['Booster_minus_4_month_cohort'] * eff_vacc_delta[3] +
        x['Booster_minus_5_month_cohort'] * eff_vacc_delta[4] +
        x['Booster_minus_6_month_cohort'] * eff_vacc_delta[5]
    ) / x['total_vacc'])) if (x['total_vacc'] >0) else 0, axis=1)


    ## 7) Cosmetics: Fill NAs with 0 and vacc efficiency to 90% in the beginning:
    df['vacc_eff'].fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)

    ## 8)
    end = time.time()
    print("The time of execution of above program is :", end - start)
    pass



def get_efficiencies_per_cohort():

    ### Insert logic here ###
    # include VOC


    eff_vacc_initial = [0.91, 0.8372, 0.7702, 0.7086, 0.6519, 0.5998]
    eff_vacc_delta = [0.8748, 0.8048, 0.7404, 0.6812, 0.6267, 0.5766]

    return eff_vacc_initial, eff_vacc_delta


if __name__ == '__main__':
    get_vaccination_effectiveness()