import sqlite3
import datetime

import sqlalchemy
import pandas as pd

from Backend.Data.data_util import format_name, date_str_to_int, validate_dates_for_query, validate_date, Column


def get_engine():
    engine = sqlalchemy.create_engine('sqlite:///Assets/Data/opendaten.db')
    return engine


def get_db_connection():
    return sqlite3.connect('Assets/Data/opendaten.db')


def update_db(table_name, dataframe):
    table_name = format_name(table_name)
    # prepare_table(table_name) this will not need to be used
    engine = get_engine()
    dataframe.to_sql(table_name, engine, if_exists='replace', index=False)


def update_db_with_index(table_name, dataframe, index_label):
    table_name = format_name(table_name)
    engine = get_engine()
    dataframe.to_sql(table_name, engine, if_exists='replace', index=True, index_label=index_label)


def get_table_data_by_duration(table='Münster', start_date='2020-03-01',
                               end_date=datetime.datetime.today().strftime('%Y-%m-%d'),
                               duration=0, attributes=None):
    if attributes is None:
        attributes = 'all'
    table_name = format_name(table)
    if validate_dates_for_query(start_date, end_date):
        engine = get_engine()

        attributes_str = ''
        if type(attributes) is list:
            if len(attributes) == 1:
                attributes_str = attributes[0].value

            elif len(attributes) >= 1:
                attributes_str = ",".join(attributes)

        else:
            attributes_str = '*'
            print('no specific attribute(s) as parameters! queried for all attributes!!')

        # assign days_back as start_day
        if duration > 0:
            current_day = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            current_day = current_day - datetime.timedelta(days=duration)
            start_date = current_day.strftime('%Y-%m-%d')

        # get the int value of the given date strings
        start = date_str_to_int(start_date)
        end = date_str_to_int(end_date)

        query_sql = 'SELECT ' + attributes_str + \
                    ' FROM ' + table_name + \
                    ' WHERE date >= ' + str(start) + \
                    ' AND date <= ' + str(end)

        return pd.read_sql(query_sql, engine)

    else:
        print('please provide correct query parameters!')


def get_table_data_by_day(table='Münster', date=datetime.datetime.today().strftime('%Y%m%d'), attributes=None):
    if attributes is None:
        attributes = 'all'
    table_name = format_name(table)
    if validate_date(date):
        engine = get_engine()

        attributes_str = ''
        if type(attributes) is list:
            if len(attributes) == 1:
                attributes_str = attributes[0]

            elif len(attributes) >= 1:
                attributes_str = ",".join(attributes)

        else:
            attributes_str = '*'
            print('no specific attribute(s) as parameters! queried for all attributes!!')

        # get the int value of the given date strings
        date_int = date_str_to_int(date)
        query_sql = 'SELECT ' + attributes_str + \
                    ' FROM ' + table_name + \
                    ' WHERE date == ' + str(date_int)

        return pd.read_sql(query_sql, engine)

    else:
        print('please provide correct query parameters!')


def get_district_data(district, attributes=None):
    attributes_str = ''

    if type(attributes) is list:
        if len(attributes) == 1:
            attributes_str = attributes[0]

        elif len(attributes) >= 1:
            attributes_str = ",".join(attributes)

    else:
        attributes_str = '*'
        print('no specific attribute(s) as parameters! queried for all attributes!!')

    engine = get_engine()
    query_sql = 'SELECT ' + attributes_str + \
                ' FROM district_details' + \
                ' WHERE district == ' + '"' + district + '"'

    return pd.read_sql(query_sql, engine)


if __name__ == '__main__':
    get_table_data_by_duration('Bremen', '2020-10-22', '2020-11-22', attributes=[Column.ADJ_ACT_CASES.value,
                                                                                 Column.VACCINATION_PERCENTAGE.value,
                                                                                 Column.CURRENT_INFECTIOUS.value])
    # get_table_data_by_duration()
