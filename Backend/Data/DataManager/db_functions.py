import sqlite3

import pandas as pd
import sqlalchemy

from Backend.Data.DataManager.data_util import format_name


def get_engine():
    engine = sqlalchemy.create_engine('sqlite:///Assets/Data/opendaten.db')
    return engine


def prepare_table(table_name):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS ' + table_name + '('
                                                     'date INTEGER, '
                                                     'daily_infec INTEGER, '
                                                     'curr_infectious INTEGER, '
                                                     'seven_day_infec INTEGER, '
                                                     'cum_infec INTEGER, '
                                                     'daily_deaths INTEGER, '
                                                     'cum_deaths INTEGER, '
                                                     'daily_rec INTEGER, '
                                                     'cum_rec INTEGER,'
                                                     'active_cases INTEGER,'
                                                     'adjusted_active_cases INTEGER,'
                                                     'daily_incidents_rate INTEGER,'
                                                     'daily_vacc INTEGER, '
                                                     'cum_vacc INTEGER, '
                                                     'vacc_percentage INTEGER)')
    cursor.execute('DELETE FROM ' + table_name)
    connection.close()


def get_db_connection():
    return sqlite3.connect('Assets/Data/opendaten.db')


def update_db_DEPRECATED(table_name, dataframe):
    table_name = format_name(table_name)
    prepare_table(table_name)
    engine = get_engine()

    dataframe.to_sql(table_name, engine, if_exists='replace', index=False)
    # dataframe.to_sql(table_name, engine, if_exists='append', index=False)


def update_district_matrices_DEPRECATED(table_name, definition,  dataframe, index_label):
    """
        creates correlation matrix data table with given input specifications.
        this method is exclusively used for calculating correlation between districts based on covid data.
        UPDATE: this method is deprecated and has no impact to any of the pipelines
    """

    table_name = format_name(table_name)
    table_name = 'cor_matrix_' + definition + '_' + table_name
    # prepare_table(table_name)
    engine = get_engine()

    dataframe.to_sql(table_name, engine, if_exists='replace', index=True, index_label=index_label)


def get_relation_data_DEPRECATED():
    reloaded_data = get_table_data_DEPRECATED('cor_matrix_incidents_districts', 0, 0, False, True)
    return reloaded_data


def get_table_data_DEPRECATED(table, date1, date2, attributes, with_index):
    table_name = format_name(table)
    engine = get_engine()

    attributes_str = ''
    if type(attributes) is list or type(attributes) is tuple:
        attributes_str = ",".join(attributes)

    elif type(attributes) is str:
        attributes_str = attributes

    else:
        attributes_str = '*'

    query_sql = 'SELECT ' + attributes_str + ' FROM ' + table_name

    if date1 > 0 and date2 > 0:
        query_sql = query_sql + ' WHERE date >= ' + str(date1) + ' AND date<=' + str(date2)

    if with_index:
        return pd.read_sql(table, engine, index_col=['district_name'])

    else:
        return pd.read_sql(query_sql, engine)


def get_filtered_table_data(table, result_attr, cond_attr, con_value):
    table_name = format_name(table)
    engine = get_engine()

    query_str = 'SELECT ' + result_attr + ' FROM ' + table_name + ' WHERE ' + cond_attr + ' = ' + con_value

    return pd.read_sql(query_str, engine)


