import pandas as pd
import numpy as np

from Backend.Modeling.Vaccination_Efficiency.get_vaccination_efficiency import get_vaccination_number


def demo():
    # 1) Import CSV File:
    df = pd.read_csv("Impfeffektivität_Demo.csv", sep=';')

    df.rename(columns={
        'daily_sec_vacc_x':'daily_sec_vacc',
        'daily_third_vacc':'Booster',
    }, inplace=True)

    # 2) Create variable with lagged second vaccinations:
    df['SecVaccStatus'] = df['daily_sec_vacc'].shift(14, fill_value=0)

    # 3) Cumulate X and Y:
    df['SecVaccStatus_cum'] = df['SecVaccStatus'].cumsum()
    df['Booster_cum'] = df['Booster'].cumsum()

    # 4) Compute cohorts: (in this example 7-day cohorts instead of 1 month for demonstration purposes)
    ## A) Second Vaccinations:
    # Erklärung: Rolling rollt über die vorherigen 7 Einträge.
    # Also werden hier immer die vorherigen 7 Tage, einschließlich dem aktuellen Tag genommen und davon die Summe berechnet.
    # minus 1 week = last week
    df['SecVaccStatus_minus_1_weeks_cohort'] = df['SecVaccStatus'].rolling(7).sum()

    # Now the the next cohort needs to be delayed by 7 days:
    df['SecVaccStatus_minus_2_weeks_cohort'] = df['SecVaccStatus'].shift(7, fill_value=0).rolling(7).sum()

    # The other cohorts:
    df['SecVaccStatus_minus_3_weeks_cohort'] = df['SecVaccStatus'].shift(7*2, fill_value=0).rolling(7).sum()
    df['SecVaccStatus_minus_4_weeks_cohort'] = df['SecVaccStatus'].shift(7*3, fill_value=0).rolling(7).sum()
    df['SecVaccStatus_minus_5_weeks_cohort'] = df['SecVaccStatus'].shift(7*4, fill_value=0).rolling(7).sum()
    df['SecVaccStatus_minus_6_weeks_cohort'] = df['SecVaccStatus'].shift(7*5, fill_value=0).rolling(7).sum()

    ## B) Third vaccinations:
    df['Booster_minus_1_weeks_cohort'] = df['Booster'].shift(7 * 0, fill_value=0).rolling(7).sum()
    df['Booster_minus_2_weeks_cohort'] = df['Booster'].shift(7 * 1, fill_value=0).rolling(7).sum()
    df['Booster_minus_3_weeks_cohort'] = df['Booster'].shift(7 * 2, fill_value=0).rolling(7).sum()
    df['Booster_minus_4_weeks_cohort'] = df['Booster'].shift(7 * 3, fill_value=0).rolling(7).sum()
    df['Booster_minus_5_weeks_cohort'] = df['Booster'].shift(7 * 4, fill_value=0).rolling(7).sum()
    df['Booster_minus_6_weeks_cohort'] = df['Booster'].shift(7 * 5, fill_value=0).rolling(7).sum()


    # 5) Now we needed to subtract the booster vaccinations to get the corrected status for people with seconds vacinations:
    # Idea: Subtract individuals that were boostered from group:
    df['SecVaccStatus_minus_1_weeks_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_1_weeks_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(7*1, fill_value=0), 0))
    df['SecVaccStatus_minus_2_weeks_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_2_weeks_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(7*2, fill_value=0), 0))
    df['SecVaccStatus_minus_3_weeks_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_3_weeks_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(7*3, fill_value=0), 0))
    df['SecVaccStatus_minus_4_weeks_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_4_weeks_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(7*4, fill_value=0), 0))
    df['SecVaccStatus_minus_5_weeks_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_5_weeks_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(7*5, fill_value=0), 0))
    df['SecVaccStatus_minus_6_weeks_cohort_CORRECTED'] = np.maximum(0, df['SecVaccStatus_minus_6_weeks_cohort'] - np.maximum(df['Booster_cum']- df['SecVaccStatus_cum'].shift(7*6, fill_value=0), 0))


    # for debugging purposes:
    temp = df[['date', 'SecVaccStatus', 'Booster', 'Booster_cum', 'SecVaccStatus_minus_4_weeks_cohort', 'SecVaccStatus_minus_4_weeks_cohort_CORRECTED']]



    # 6) Compute vaccination efficiency:
    eff_sec_vacc_cohort_1 = 0.90
    eff_sec_vacc_cohort_2 = 0.85
    eff_sec_vacc_cohort_3 = 0.80
    eff_sec_vacc_cohort_4 = 0.75
    eff_sec_vacc_cohort_5 = 0.70
    eff_sec_vacc_cohort_6 = 0.65

    eff_booster_cohort_1 = 0.9
    eff_booster_cohort_2 = 0.8
    eff_booster_cohort_3 = 0.7
    eff_booster_cohort_4 = 0.6
    eff_booster_cohort_5 = 0.5
    eff_booster_cohort_6 = 0.4

    df['total_second_vacc_status'] = np.sum(
        df[['SecVaccStatus_minus_1_weeks_cohort_CORRECTED',
            'SecVaccStatus_minus_2_weeks_cohort_CORRECTED',
            'SecVaccStatus_minus_3_weeks_cohort_CORRECTED',
            'SecVaccStatus_minus_4_weeks_cohort_CORRECTED',
            'SecVaccStatus_minus_5_weeks_cohort_CORRECTED',
            'SecVaccStatus_minus_6_weeks_cohort_CORRECTED',
        ]], axis=1
    )

    df['total_booster_vacc'] = np.sum(
        df[['Booster_minus_1_weeks_cohort',
            'Booster_minus_2_weeks_cohort',
            'Booster_minus_3_weeks_cohort',
            'Booster_minus_4_weeks_cohort',
            'Booster_minus_5_weeks_cohort',
            'Booster_minus_6_weeks_cohort',
        ]], axis=1
    )

    df['total_vacc'] = df['total_second_vacc_status'] + df['total_booster_vacc']

    df['vacc_eff'] = (
        df['SecVaccStatus_minus_1_weeks_cohort_CORRECTED'] * eff_sec_vacc_cohort_1 +
        df['SecVaccStatus_minus_2_weeks_cohort_CORRECTED'] * eff_sec_vacc_cohort_2 +
        df['SecVaccStatus_minus_3_weeks_cohort_CORRECTED'] * eff_sec_vacc_cohort_3 +
        df['SecVaccStatus_minus_4_weeks_cohort_CORRECTED'] * eff_sec_vacc_cohort_4 +
        df['SecVaccStatus_minus_5_weeks_cohort_CORRECTED'] * eff_sec_vacc_cohort_5 +
        df['SecVaccStatus_minus_6_weeks_cohort_CORRECTED'] * eff_sec_vacc_cohort_6 +

        df['Booster_minus_1_weeks_cohort'] * eff_booster_cohort_1 +
        df['Booster_minus_2_weeks_cohort'] * eff_booster_cohort_2 +
        df['Booster_minus_3_weeks_cohort'] * eff_booster_cohort_3 +
        df['Booster_minus_4_weeks_cohort'] * eff_booster_cohort_4 +
        df['Booster_minus_5_weeks_cohort'] * eff_booster_cohort_5 +
        df['Booster_minus_6_weeks_cohort'] * eff_booster_cohort_6
    ) / df['total_vacc']

    pass



def demo_real_data():

    # 1) Import District Vaccination Data from API:
    df = get_vaccination_number('Münster')

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
    df['SecVaccStatus_minus_1_month_cohort'] = df['SecVaccStatus'].rolling(7).sum()

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
    eff_sec_vacc, eff_booster_vacc = get_efficiencies_per_cohort()

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

    df['vacc_eff'] = (
        df['SecVaccStatus_minus_1_month_cohort_CORRECTED'] * eff_sec_vacc[0] +
        df['SecVaccStatus_minus_2_month_cohort_CORRECTED'] * eff_sec_vacc[1] +
        df['SecVaccStatus_minus_3_month_cohort_CORRECTED'] * eff_sec_vacc[2] +
        df['SecVaccStatus_minus_4_month_cohort_CORRECTED'] * eff_sec_vacc[3] +
        df['SecVaccStatus_minus_5_month_cohort_CORRECTED'] * eff_sec_vacc[4] +
        df['SecVaccStatus_minus_6_month_cohort_CORRECTED'] * eff_sec_vacc[5] +

        df['Booster_minus_1_month_cohort'] * eff_booster_vacc[0] +
        df['Booster_minus_2_month_cohort'] * eff_booster_vacc[1] +
        df['Booster_minus_3_month_cohort'] * eff_booster_vacc[2] +
        df['Booster_minus_4_month_cohort'] * eff_booster_vacc[3] +
        df['Booster_minus_5_month_cohort'] * eff_booster_vacc[4] +
        df['Booster_minus_6_month_cohort'] * eff_booster_vacc[5]
    ) / df['total_vacc']

    ## 7) Cosmetics: Fill NAs with 0 and vacc efficiency to 90% in the beginning:
    df['vacc_eff'].fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)

    ## 8)

    pass



def get_efficiencies_per_cohort():

    ### Insert logic here ###

    eff_sec_vacc = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65]
    eff_booster_vacc = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]

    return eff_sec_vacc, eff_booster_vacc


if __name__ == '__main__':
    demo_real_data()