import datetime
import pandas as pd
import numpy as np

from Backend.Data.data_scraping.data_scraping_calls import update_ecdc_variant_data, update_destatis_mobility_data, \
    update_oxcgrt_policy_data
from Backend.Data.data_util import get_correct_district_name
from Backend.Data.db_calls import update_db, get_all_table_data


def update_ecdc_variant_table():
    update_ecdc_variant_data()
    print('latest update of variant data fetched successfully!')

    latest_file_loc = '../Assets/Data/Scraped/ecdc/' + datetime.datetime.today().strftime('%d%m%y') + '.csv'
    df = pd.read_csv(latest_file_loc)
    germany = df[df['country'] == 'Germany']
    germany = germany.drop(['country', 'country_code'], axis=1)
    germany = germany.sort_values(by=['year_week'])
    update_db('ecdc_varient_data', germany)


# IMPORTANT: only used to extract the initial data
def store_destatis_base_data():
    base_data_file = '../Assets/Data/Scraped/destatis/destatis_base.csv'
    base_data = pd.read_csv(base_data_file)

    for index, row in base_data.iterrows():
        base_data.at[index, 'Kreisname'] = get_correct_district_name(row['Kreisname'])

    update_db('destatis_mobility_data', base_data)


def update_destatis_mobility_table():
    update_destatis_mobility_data()
    print('latest update of mobility data fetched successfully!')

    all_data = get_all_table_data(table_name='destatis_mobility_data')
    all_data_last_date = [*all_data.columns[-1:]][0]

    latest_file = '../Assets/Data/Scraped/destatis/' + datetime.datetime.today().strftime('%d%m%y') + '.csv'
    latest_data = pd.read_csv(latest_file, sep=";")

    start_column = int(latest_data.columns.get_loc(all_data_last_date))
    start_column = start_column + 1
    # a_dataframe.drop(a_dataframe.columns[0], axis=1, inplace=True)
    latest_data.drop(latest_data.columns[0:start_column], axis=1, inplace=True)
    merged_new_data = pd.concat([all_data, latest_data], axis=1)
    update_db('destatis_mobility_data', merged_new_data)


def update_oxcgrt_policy_table():
    update_oxcgrt_policy_data()
    print('latest update of mobility data fetched successfully!')

    latest_file_loc = '../Assets/Data/Scraped/oxcgrt/' + datetime.datetime.today().strftime('%d%m%y') + '.csv'
    df = pd.read_csv(latest_file_loc)
    df = df.drop(df.columns[0], axis=1)
    germany = df[df['country_code'] == 'DEU']
    germany = germany.replace("", np.nan)
    germany = germany.fillna(0)
    germany = germany.drop(df.columns[0:2], axis=1)
    germany = germany.T
    germany = germany.rename(columns={germany.columns[0]: 'policy_index'})
    germany['date'] = germany.index
    germany['date'] = germany['date'].map(lambda date: datetime.datetime.strptime(date, '%d%b%Y').strftime('%Y-%m-%d'))
    germany = germany.reindex(columns=['date', 'policy_index'])

    update_db('xocgrt_policy_data', germany)


if __name__ == '__main__':
    ## store_destatis_base_data()
    update_ecdc_variant_table()
    update_destatis_mobility_table()
    update_oxcgrt_policy_table()
