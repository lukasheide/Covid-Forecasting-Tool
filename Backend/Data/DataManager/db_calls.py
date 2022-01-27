import sqlite3
from datetime import datetime, timedelta

import sqlalchemy
import pandas as pd
from meteostat import Point, Daily

from Backend.Data.DataManager.data_util import format_name, date_str_to_int, validate_dates_for_query, validate_date, \
    Column


def get_engine():
    engine = sqlalchemy.create_engine('sqlite:///../Assets/Data/opendaten.db')
    return engine


def get_db_connection():
    return sqlite3.connect('../Assets/Data/opendaten.db')


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
                               end_date=datetime.today().strftime('%Y-%m-%d'),
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
            current_day = datetime.strptime(end_date, '%Y-%m-%d')
            current_day = current_day - timedelta(days=duration)
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


def get_table_data_by_day(table='Münster', date=datetime.today().strftime('%Y%m%d'), attributes=None):
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


def get_all_table_data(table_name):
    table_name = format_name(table_name)
    engine = get_engine()
    return pd.read_sql(table_name, engine)


def get_policy_data(date=None):
    engine = get_engine()

    if date is not None:
        query_sql = 'SELECT policy_index ' \
                    'FROM xocgrt_policy_data ' \
                    'WHERE date = "%s"' % (date,)

        result = pd.read_sql(query_sql, engine)
        if result.empty:
            query_sql = 'SELECT policy_index ' \
                        'FROM xocgrt_policy_data ' \
                        'WHERE policy_index > 0 ' \
                        'ORDER BY date DESC ' \
                        'LIMIT 1'
            result = pd.read_sql(query_sql, engine)
            print('no policy data for the given date, latest available value is selected!')

        return result['policy_index'][0]

    else:
        query_sql = 'SELECT policy_index ' \
                    'FROM xocgrt_policy_data ' \
                    'WHERE policy_index > 0 ' \
                    'ORDER BY date DESC ' \
                    'LIMIT 1'
        result = pd.read_sql(query_sql, engine)
        print('no policy data for the given date, latest available data is selected!')
        return result['policy_index'][0]


def get_variant_data(date=None):
    engine = get_engine()

    if date is not None:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        week = date_obj.isocalendar()[1]
        year_week_str = str(date_obj.year) + '-' + str(week if week >= 10 else str(week).zfill(2))

        query_sql = 'SELECT variant ' \
                    'FROM ecdc_varient_data ' \
                    'WHERE year_week = "%s" AND percent_variant > 0 ' \
                    'ORDER BY year_week, percent_variant DESC ' \
                    'LIMIT 1' \
                    % (year_week_str,)

        result = pd.read_sql(query_sql, engine)

        if result.empty:
            query_sql = 'SELECT variant ' \
                        'FROM ecdc_varient_data ' \
                        'WHERE percent_variant > 0 ' \
                        'ORDER BY year_week, percent_variant DESC ' \
                        'LIMIT 1'
            result = pd.read_sql(query_sql, engine)
            print('no variant data for the given date, latest available week data is selected!')

        return result['variant'][0]

    else:
        query_sql = 'SELECT variant ' \
                    'FROM ecdc_varient_data ' \
                    'WHERE percent_variant > 0 ' \
                    'ORDER BY year_week, percent_variant DESC ' \
                    'LIMIT 1'
        result = pd.read_sql(query_sql, engine)
        print('no variant data for the given date, latest available week data is selected!')

        return result['variant'][0]


def get_mobility_data(district, date=None):
    engine = get_engine()

    if date is not None:
        query_sql = 'SELECT "%s" '\
                    'FROM destatis_mobility_data ' \
                    'WHERE Kreisname = "%s" ' % (date, district,)

        result = pd.read_sql(query_sql, engine)

        if result.empty:
            # couldnt find a quesry to get the last column data
            # therefore, read the whole table and prepare the df to do the task
            #end_date = [*mob_data.columns[-1:]][0]
            mob_data = get_all_table_data(table_name='destatis_mobility_data')
            dist_mobility = mob_data.loc[mob_data['Kreisname'] == district]
            result = dist_mobility.iloc[:, -1].iloc[0]
            print('no variant data for the given date, latest available week data is selected!')
            return result

        return result[date].tolist()[0]

    else:
        # couldnt find a quesry to get the last column data
        # therefore, read the whole table and prepare the df to do the task
        # end_date = [*mob_data.columns[-1:]][0]
        mob_data = get_all_table_data(table_name='destatis_mobility_data')
        dist_mobility = mob_data.loc[mob_data['Kreisname'] == district]
        result = dist_mobility.iloc[:, -1].iloc[0]
        print('no variant data for the given date, latest available week data is selected!')
        return result


def get_weather_data(district, date=None):
    if district == 'Garmisch-Partenkirchen':
        district == 'Weilheim-Schongau'
    location = get_district_data(district, [Column.LATITUDE, Column.LONGITUDE])
    dist_lat = float(location[Column.LATITUDE].iloc[0])
    dist_lon = float(location[Column.LONGITUDE].iloc[0])
    date_obj = datetime.strptime(date, '%Y-%m-%d')

    district_loc = Point(dist_lat, dist_lon)
    data = Daily(district_loc, date_obj, date_obj)
    data = data.fetch()

    temperature = data['tavg'][0]
    wind = data['wspd'][0]

    return temperature, wind


def clean_create_model_store():
    connection = get_db_connection()
    cursor = connection.cursor()

    # clean
    cursor.executescript('DROP TABLE IF EXISTS prediction;')
    cursor.executescript('DROP TABLE IF EXISTS param_and_start_vals;')
    cursor.executescript('DROP TABLE IF EXISTS pipeline;')

    create_pipeline_sql = "CREATE TABLE IF NOT EXISTS pipeline( " \
                          "pipeline_id INTEGER PRIMARY KEY, " \
                          "end_date TEXT NOT NULL," \
                          "val_duration INTEGER NOT NULL," \
                          "visualize BOOLEAN NOT NULL," \
                          "validate BOOLEAN NOT NULL," \
                          "verbose BOOLEAN NOT NULL," \
                          "started_on TEXT NOT NULL);"
    cursor.executescript(create_pipeline_sql)

    create_param_sql = "CREATE TABLE IF NOT EXISTS param_and_start_vals( " \
                       "pipeline_id INTEGER NOT NULL," \
                       "district_name TEXT NOT NULL," \
                       "population INTEGER NOT NULL," \
                       "vaccinated INTEGER NOT NULL," \
                       "recovered INTEGER NOT NULL," \
                       "beta REAL NOT NULL," \
                       "gamma_I REAL NOT NULL," \
                       "gamma_U REAL NOT NULL," \
                       "delta REAL NOT NULL," \
                       "theta REAL NOT NULL," \
                       "rho REAL NOT NULL," \
                       "PRIMARY KEY (district_name, pipeline_id)" \
                       "FOREIGN KEY(pipeline_id) " \
                       "REFERENCES pipeline(pipeline_id));"
    cursor.execute(create_param_sql)

    create_prediction_sql = "CREATE TABLE IF NOT EXISTS prediction( " \
                            "prediction_id INTEGER PRIMARY KEY," \
                            "pipeline_id INTEGER NOT NULL," \
                            "district_name TEXT NOT NULL," \
                            "date TEXT NOT NULL," \
                            "cases REAL NOT NULL," \
                            "FOREIGN KEY(district_name, pipeline_id) " \
                            "REFERENCES param_and_start_vals(district_name, pipeline_id));"
    cursor.executescript(create_prediction_sql)

    connection.close()


def start_pipeline(end_date, validation_duration, visualize, validate, verbose):
    connection = get_db_connection()
    cursor = connection.cursor()
    sql_srt = 'INSERT INTO pipeline (end_date, ' \
              'val_duration, ' \
              'visualize, ' \
              'verbose, ' \
              'validate, ' \
              'started_on) values (?, ?, ?, ?, ?, ?)'
    cursor.execute(sql_srt, (end_date, validation_duration, visualize, validate, verbose, datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")))
    cursor.execute('SELECT MAX(pipeline_id) FROM pipeline;')
    pipeline_id = cursor.fetchone()[0]

    connection.commit()
    connection.close()

    return pipeline_id


def insert_param_and_start_vals(pipeline_id, district_name, start_vals, model_params):
    connection = get_db_connection()
    cursor = connection.cursor()

    # prepare the list
    param_list = [pipeline_id, district_name] + list(start_vals) + list(model_params.values())

    sql_srt = 'INSERT INTO param_and_start_vals (' \
              'pipeline_id,' \
              'district_name, ' \
              'population,' \
              'vaccinated,' \
              'recovered,' \
              'beta,' \
              'gamma_I,' \
              'gamma_U,' \
              'delta,' \
              'theta,' \
              'rho) ' \
              'values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
    cursor.execute(sql_srt, param_list)
    connection.commit()
    connection.close()


def insert_prediction_vals(pipeline_id, district_name, predictions, train_end_date):
    connection = get_db_connection()
    cursor = connection.cursor()
    current_day = datetime.strptime(train_end_date, '%Y-%m-%d')
    # next day is the validation/prediction start date
    predictions = pd.DataFrame(data=predictions)

    for i, cases in predictions.iterrows():
        current_day = current_day + timedelta(days=1)
        current_day_str = current_day.strftime('%Y-%m-%d')
        # prepare the list
        param_list = ()

        sql_srt = 'INSERT INTO prediction (' \
                  'district_name, ' \
                  'pipeline_id,' \
                  'date,' \
                  'cases) ' \
                  'values (?, ?, ?, ?)'
        cursor.execute(sql_srt, (district_name, pipeline_id, current_day_str, cases[0]))
    connection.commit()
    connection.close()


if __name__ == '__main__':
    # get_table_data_by_duration('Bremen', '2020-10-25', '2020-11-22', attributes=[Column.ADJ_ACT_CASES.value,
    #                                                                              Column.VACCINATION_PERCENTAGE.value,
    #                                                                              Column.CURRENT_INFECTIOUS.value])
    # get_table_data_by_duration()
    clean_create_model_store()
