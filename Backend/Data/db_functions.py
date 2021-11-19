import sqlite3

import pandas as pd
import sqlalchemy
from datetime import datetime


def get_engine():
    engine = sqlalchemy.create_engine('sqlite:///../../Assets/Data/opendaten.db')
    return engine


def prepare_table(table_name):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS ' + table_name + '('
                                                     'date INTEGER, '
                                                     'daily_infec INTEGER, '
                                                     'cum_infec INTEGER, '
                                                     'daily_deaths INTEGER, '
                                                     'cum_deaths INTEGER, '
                                                     'daily_rec INTEGER, '
                                                     'cum_rec INTEGER,'
                                                     'active_cases INTEGER,'
                                                     'adjusted_active_cases INTEGER,'
                                                     'daily_incidents_rate INTEGER,'
                                                     'daily_vacc INTEGER, '
                                                     'cum_vacc INTEGER)')
    cursor.execute('DELETE FROM ' + table_name)
    connection.close()


def format_name(table_name):
    string = table_name

    u = 'ü'.encode()
    U = 'Ü'.encode()
    a = 'ä'.encode()
    A = 'Ä'.encode()
    o = 'ö'.encode()
    O = 'Ö'.encode()
    ss = 'ß'.encode()

    string = string.encode()
    string = string.replace(u, b'ue')
    string = string.replace(U, b'Ue')
    string = string.replace(a, b'ae')
    string = string.replace(A, b'Ae')
    string = string.replace(o, b'oe')
    string = string.replace(O, b'Oe')
    string = string.replace(ss, b'ss')

    string = string.decode('utf-8')

    string = string.replace(", ", "_")
    string = string.replace("/", "_")
    string = string.replace(" ", "_")
    string = string.replace(".", "_")
    string = string.replace("-", "_")
    string = string.replace("(", "")
    string = string.replace(")", "")

    print(string)

    return string


def get_db_connection():
    return sqlite3.connect('../../Assets/Data/opendaten.db')


def update_db(table_name, dataframe):
    table_name = format_name(table_name)
    prepare_table(table_name)
    engine = get_engine()

    dataframe.to_sql(table_name, engine, if_exists='replace', index=False)


def evaluate_and_joining_dates(date1, date2):
    today = int(datetime.today().strftime('%Y%m%d'))
    # from_date = 0
    # to_date = 0
    #
    # if date1 < today and date2 < today:
    #     if date1 < date2:
    #         from_date = date1
    #         to_date = date2
    #
    #     else:
    #         from_date = date2
    #         to_date = date1
    #
    # elif today < date1 < date2:
    #     print("invalid date parameter!")


def execute_query(table, date1, date2, attributes):
    table_name = format_name(table)
    engine = get_engine()

    attributes_str = ''
    if type(attributes) is list or type(attributes) is tuple:
        attributes_str = ",".join(attributes)

    else:
        attributes_str = attributes

    query_sql = 'SELECT ' + attributes_str + ' FROM ' + table_name

    if date1 > 0 and date2 > 0:
        query_sql = query_sql + ' WHERE date >= ' + str(date1) + ' AND date<=' + str(date2)

    return pd.read_sql(query_sql, engine)
