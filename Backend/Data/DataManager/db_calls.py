import math
import sqlite3
from datetime import datetime, timedelta

import sqlalchemy
import pandas as pd
from meteostat import Point, Daily

from Backend.Data.DataManager.data_util import format_name, date_str_to_int, validate_dates_for_query, validate_date, \
    Column

"""
    only methods that directly connects/query the DB. All method returns are dataframes
"""


def get_engine():
    engine = sqlalchemy.create_engine('sqlite:///Assets/Data/covcast.db')
    return engine


def get_db_connection():
    return sqlite3.connect('Assets/Data/covcast.db')


def update_db(table_name, dataframe, replace=True):
    """
       store a dataframe in a table with a given table name. if the table not exists, sqlalchemy handles the table
       creation internally. if the table exists, records and be appended or replaced(truncate and store)
    """
    table_name = format_name(table_name)
    exec_type = 'replace'
    # prepare_table(table_name) this will not need to be used
    if not replace:
        exec_type = 'append'
    engine = get_engine()
    dataframe.to_sql(table_name, engine, if_exists=exec_type, index=False)


def drop_table_by_name(table_name):
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.executescript('DROP TABLE IF EXISTS '+table_name+';')
    connection.close()


def update_db_with_index(table_name, dataframe, index_label):
    """
       store a dataframe in a table with a given table name with specified df column as the index label(s).
       if the table not exists, sqlalchemy handles the table creation internally.
       if the table exists, records and be appended or replaced(truncate and store).
    """
    table_name = format_name(table_name)
    engine = get_engine()
    dataframe.to_sql(table_name, engine, if_exists='replace', index=True, index_label=index_label)


def get_table_data_by_duration(table='Münster', start_date='2020-03-01',
                               end_date=datetime.today().strftime('%Y-%m-%d'),
                               duration=0, attributes=None):
    """
        exclusively can be used for querying covid data tables (by district name as table name) by time period and
        list of attributes of interest.
    """

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
    """
        exclusively can be used for querying covid data tables (by district name as table name) by a specific day and
        list of attributes of interest.
    """
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
    """
        get other-data (ex: population, center lat/lang, etc) of a district of interest
    """
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


def get_table_data(table, attributes):
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

    return pd.read_sql(query_sql, engine)


def get_all_table_data(table_name):
    """
        get all the data from a table
    """
    table_name = format_name(table_name)
    engine = get_engine()
    return pd.read_sql(table_name, engine)


def get_policy_data(date=None):
    """
        get intervention-policy index for a date given.
        index value is at country level therefore common for any district
    """
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
    """
        get the dominant variant name for a date given.
        variant is at country level therefore common for any district
    """
    engine = get_engine()
    data_start_date_obj = datetime.strptime('2020-09-28', '%Y-%m-%d') # start date of the 2020-40th week

    if date is not None:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        week = date_obj.isocalendar()[1]
        year_week_str = str(date_obj.year) + '-' + str(week if week >= 10 else str(week).zfill(2))

        query_sql = 'SELECT variant ' \
                    'FROM ecdc_variant_data ' \
                    'WHERE year_week = "%s" AND percent_variant > 0 ' \
                    'ORDER BY year_week, percent_variant DESC ' \
                    'LIMIT 1' \
                    % (year_week_str,)

        result = pd.read_sql(query_sql, engine)

        if result.empty:
            if date_obj < data_start_date_obj:
                query_sql = 'SELECT variant ' \
                            'FROM ecdc_variant_data ' \
                            'WHERE percent_variant > 0 ' \
                            'ORDER BY year_week, percent_variant ASC ' \
                            'LIMIT 1'
                result = pd.read_sql(query_sql, engine)
                print('data starting date is ahead for the given date, starting date data is selected!')

            else:
                query_sql = 'SELECT variant ' \
                            'FROM ecdc_variant_data ' \
                            'WHERE percent_variant > 0 ' \
                            'ORDER BY year_week, percent_variant DESC ' \
                            'LIMIT 1'
                result = pd.read_sql(query_sql, engine)
                print('no variant data for the given date, latest available week data is selected!')

        return result['variant'][0]

    else:
        query_sql = 'SELECT variant ' \
                    'FROM ecdc_variant_data ' \
                    'WHERE percent_variant > 0 ' \
                    'ORDER BY year_week, percent_variant DESC ' \
                    'LIMIT 1'
        result = pd.read_sql(query_sql, engine)
        print('no variant data for the given date, latest available week data is selected!')

        return result['variant'][0]


def get_mobility_data(district, date=None):
    """
        get the mobility data of a district for a date given.
        mobility data is at district level therefore, specific for each district
    """
    engine = get_engine()

    if date is not None:
        query_sql = 'SELECT "%s" ' \
                    'FROM destatis_mobility_data ' \
                    'WHERE Kreisname = "%s" ' % (date, district,)

        result = pd.read_sql(query_sql, engine)

        if result.iloc[:,0].tolist()[0] == date:
            # couldnt find a quesry to get the last column data
            # therefore, read the whole table and prepare the df to do the task
            # end_date = [*mob_data.columns[-1:]][0]
            mob_data = get_all_table_data(table_name='destatis_mobility_data')
            dist_mobility = mob_data.loc[mob_data['Kreisname'] == district]
            result = dist_mobility.iloc[:, -1].iloc[0]
            print('no mobility data for the given date, latest available week data is selected!')
            return result

        return result[date].tolist()[0]

    else:
        # couldnt find a quesry to get the last column data
        # therefore, read the whole table and prepare the df to do the task
        # end_date = [*mob_data.columns[-1:]][0]
        mob_data = get_all_table_data(table_name='destatis_mobility_data')
        dist_mobility = mob_data.loc[mob_data['Kreisname'] == district]
        result = dist_mobility.iloc[:, -1].iloc[0]
        print('no mobility data for the given date, latest available week data is selected!')
        return result


def get_weather_data(district, date=None):
    """
        get the weather( avg. temperature, avg. wind) of a district for a date given.
        weather data is at district level therefore, specific for each district
        LAT LON of the district's main city is retrieved from district_details table
    """
    temp_filler = 15.0
    wind_filler = 10.0
    if district == 'Garmisch-Partenkirchen':
        district = 'Weilheim-Schongau'
    location = get_district_data(district, [Column.LATITUDE, Column.LONGITUDE])
    dist_lat = float(location[Column.LATITUDE].iloc[0])
    dist_lon = float(location[Column.LONGITUDE].iloc[0])
    date_obj = datetime.strptime(date, '%Y-%m-%d')

    district_loc = Point(dist_lat, dist_lon)
    data = Daily(district_loc, date_obj, date_obj)
    data = data.fetch()

    if data.empty:
        temperature = temp_filler
        wind = wind_filler
    else:
        temperature = data['tavg'][0]
        wind = data['wspd'][0]

        if math.isnan(data['tavg'][0]):
            temperature = temp_filler
        if math.isnan(data['wspd'][0]):
            wind = wind_filler

    return temperature, wind


def get_district_forecast_data(district):
    """
        get the latest forecast of a given district from the latest forecast pipeline run results
    """
    engine = get_engine()

    query_sql = 'SELECT * ' \
                'FROM district_forecast ' \
                'WHERE district_name = "%s" ' \
                'AND pipeline_id = (' \
                'SELECT MAX(df2.pipeline_id) ' \
                'FROM district_forecast df2 WHERE district_name = "%s")' \
                % (district, district)

    return pd.read_sql(query_sql, engine)


def get_all_latest_forecasts():
    """
        get the latest forecast of of all the districts from the latest forecast pipeline run results
    """
    engine = get_engine()

    query_sql = 'SELECT * ' \
                'FROM district_forecast ' \
                'WHERE pipeline_id = (' \
                'SELECT MAX(fp.pipeline_id) ' \
                'FROM forecast_pipeline fp WHERE full_run = TRUE AND completed = TRUE) ;' \

    return pd.read_sql(query_sql, engine)


def clean_create_validation_store(with_clean=False):
    """
        create the model validation pipeline relational schema tables
    """
    connection = get_db_connection()
    cursor = connection.cursor()

    if with_clean:
        cursor.executescript('DROP TABLE IF EXISTS validation_forecast;')
        cursor.executescript('DROP TABLE IF EXISTS param_and_start_vals;')
        cursor.executescript('DROP TABLE IF EXISTS validation_pipeline;')

    create_pipeline_sql = "CREATE TABLE IF NOT EXISTS validation_pipeline( " \
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
                       "REFERENCES validation_pipeline(pipeline_id));"
    cursor.execute(create_param_sql)

    create_prediction_sql = "CREATE TABLE IF NOT EXISTS validation_forecast( " \
                            "prediction_id INTEGER PRIMARY KEY," \
                            "pipeline_id INTEGER NOT NULL," \
                            "district_name TEXT NOT NULL," \
                            "date TEXT NOT NULL," \
                            "cases REAL NOT NULL," \
                            "FOREIGN KEY(district_name, pipeline_id) " \
                            "REFERENCES param_and_start_vals(district_name, pipeline_id));"
    cursor.executescript(create_prediction_sql)

    connection.close()


def create_forecast_store(with_clean=False):
    """
        create the model forecast pipeline relational schema tables
    """
    connection = get_db_connection()
    cursor = connection.cursor()

    if with_clean:
        cursor.executescript('DROP TABLE IF EXISTS district_forecast;')
        cursor.executescript('DROP TABLE IF EXISTS forecast_pipeline;')

    create_sql = "CREATE TABLE IF NOT EXISTS forecast_pipeline( " \
                 "pipeline_id INTEGER PRIMARY KEY, " \
                 "train_start_date TEXT NOT NULL," \
                 "train_end_date TEXT NOT NULL," \
                 "frcst_start_date TEXT NOT NULL," \
                 "frcst_end_date TEXT NOT NULL," \
                 "full_run BOOLEAN NOT NULL," \
                 "started_on TEXT NOT NULL," \
                 "completed BOOLEAN NOT NULL," \
                 "ended_on TEXT );"
    cursor.executescript(create_sql)

    create_prediction_sql = "CREATE TABLE IF NOT EXISTS district_forecast( " \
                            "prediction_id INTEGER PRIMARY KEY," \
                            "pipeline_id INTEGER NOT NULL," \
                            "district_name TEXT NOT NULL," \
                            "date TEXT NOT NULL," \
                            "cases REAL," \
                            "y_pred_seirv_last_beta_mean TEXT," \
                            "y_pred_seirv_last_beta_upper TEXT," \
                            "y_pred_seirv_last_beta_lower TEXT," \
                            "y_pred_seirv_ml_beta_mean TEXT," \
                            "y_pred_seirv_ml_beta_upper TEXT," \
                            "y_pred_seirv_ml_beta_lower TEXT," \
                            "y_pred_sarima_mean TEXT," \
                            "y_pred_sarima_upper TEXT," \
                            "y_pred_sarima_lower TEXT," \
                            "y_pred_ensemble_mean TEXT," \
                            "y_pred_ensemble_upper TEXT," \
                            "y_pred_ensemble_lower TEXT," \
                            "FOREIGN KEY(pipeline_id) " \
                            "REFERENCES forecast_pipeline(pipeline_id));"
    cursor.executescript(create_prediction_sql)

    connection.close()


def start_validation_pipeline(end_date, validation_duration, visualize, validate, verbose):
    """
        creates an entry in 'validation_pipeline' to start the new pipeline run and returns the pipeline id
    """
    connection = get_db_connection()
    cursor = connection.cursor()
    sql_srt = 'INSERT INTO validation_pipeline (end_date, ' \
              'val_duration, ' \
              'visualize, ' \
              'verbose, ' \
              'validate, ' \
              'started_on,' \
              'completed,' \
              'ended_on) values (?, ?, ?, ?, ?, ?, ?, ?)'
    cursor.execute(sql_srt, (
        end_date,
        validation_duration,
        visualize,
        validate,
        verbose,
        datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"),
        False))
    cursor.execute('SELECT MAX(pipeline_id) FROM validation_pipeline;')
    pipeline_id = cursor.fetchone()[0]

    connection.commit()
    connection.close()

    return pipeline_id


def start_forecast_pipeline(t_start_date, t_end_date, f_start_date, f_end_date, full_run):
    """
        creates an entry in 'forecast_pipeline' to start the new pipeline run and returns the pipeline id
    """
    connection = get_db_connection()
    cursor = connection.cursor()
    sql_srt = 'INSERT INTO forecast_pipeline (' \
              'train_start_date, ' \
              'train_end_date, ' \
              'frcst_start_date, ' \
              'frcst_end_date, ' \
              'full_run, ' \
              'started_on,' \
              'completed) values (?, ?, ?, ?, ?, ?, ?)'
    cursor.execute(sql_srt, (
        t_start_date,
        t_end_date,
        f_start_date,
        f_end_date,
        full_run,
        datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"),
        False))
    cursor.execute('SELECT MAX(pipeline_id) FROM forecast_pipeline;')
    pipeline_id = cursor.fetchone()[0]

    connection.commit()
    connection.close()

    return pipeline_id


def end_forecast_pipeline(pipeline_id):
    """
        end the forecast-pipeline run under the given pipeline
    """
    connection = get_db_connection()
    cursor = connection.cursor()

    query_sql = 'UPDATE forecast_pipeline ' \
                'SET completed = ?, ended_on = ? ' \
                'WHERE pipeline_id = ? ;' \

    cursor.execute(query_sql, (
        True,
        datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"),
        pipeline_id))

    connection.commit()
    connection.close()


def insert_param_and_start_vals(pipeline_id, district_name, start_vals, model_params):
    """
        stores the parameters and starting values of the validation pipeline run
    """
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


def insert_forecast_vals(pipeline_id, district_name, predictions, train_end_date):
    """
        insert the forecasts generated for a district from the validation pipeline run under the given pipeline_id
    """
    connection = get_db_connection()
    cursor = connection.cursor()
    current_day = datetime.strptime(train_end_date, '%Y-%m-%d')
    # next day is the validation/prediction start date
    predictions = pd.DataFrame(data=predictions)

    for i, cases in predictions.iterrows():
        current_day = current_day + timedelta(days=1)
        current_day_str = current_day.strftime('%Y-%m-%d')

        sql_srt = 'INSERT INTO validation_forecast (' \
                  'district_name, ' \
                  'pipeline_id,' \
                  'date,' \
                  'cases) ' \
                  'values (?, ?, ?, ?)'
        cursor.execute(sql_srt, (district_name, pipeline_id, current_day_str, cases[0]))
    connection.commit()
    connection.close()


if __name__ == '__main__':
    """
        at the very beginning, below two methods
    """
    # clean_create_validation_store()
    # clean_create_forecast_store()
    """
        must be executed once to create the corresponding DB tables
    """
