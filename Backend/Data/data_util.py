import datetime
from enum import Enum


class Column(Enum):
    # columns of tables named after 'districts'
    DATE = 'date'
    DAILY_INFECTIONS = 'daily_infec'
    CURRENT_INFECTIOUS = 'curr_infectious'
    SEVEN_DAY_SMOOTHEN = 'seven_day_infec'
    CUM_INFECTIONS = 'cum_infec'
    DAILY_DEATHS = 'daily_deaths'
    CUM_DEATHS = 'cum_deaths'
    DAILY_RECOVERIES = 'daily_rec'
    CUM_RECOVERIES = 'cum_rec'
    ACTIVE_CASES = 'active_cases'
    ADJ_ACT_CASES = 'adjusted_active_cases'
    SEVEN_DAY_INCIDENTS = 'daily_incidents_rate'
    DAILY_VACCINATED = 'daily_vacc'
    CUM_VACCINATED = 'cum_vacc'
    VACCINATION_PERCENTAGE = 'vacc_percentage'

    # columns of 'district_details' table
    STATE = 'state'
    DISTRICT = 'district'
    POPULATION = 'population'
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'


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

    return string


def validate_dates_for_query(start_date, end_date):
    """
    this function will validate date parameters given for the db query
    :param start_date: should be in DD-MM-YYYY format
    :param end_date: should be in DD-MM-YYYY format
    """
    if validate_date(start_date) and validate_date(end_date):
        today = int(datetime.datetime.today().strftime('%Y%m%d'))
        from_date = date_str_to_int(start_date)
        to_date = date_str_to_int(end_date)

        if today >= to_date > from_date:
            return True

    else:
        print('invalid query parameter dates!')
        print('start: ' + start_date + ' end: ' + end_date)
        return False


def validate_date(date):
    date_positions = date.split("-")
    current_year = int(datetime.datetime.today().strftime('%Y'))

    if 0 < int(date_positions[0]) <= current_year \
            and 0 < int(date_positions[1]) <= 12 \
            and 0 < int(date_positions[2]) <= 31:
        return True
    else:
        return False


def date_str_to_int(date_str):
    date = int(str(date_str).replace("-", ""))

    return date


def date_int_str(date_int):
    date = datetime.datetime.strptime(date_int, '%Y%m%d')
    date = date.strftime('%Y-%m-%d')

    return date
