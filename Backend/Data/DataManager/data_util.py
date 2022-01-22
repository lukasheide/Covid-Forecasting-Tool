import datetime
import sys
import os


class Column:
    # columns of tables named after 'districts'
    DATE = 'date'
    DAILY_INFECTIONS = 'daily_infec'
    CURRENT_INFECTIOUS = 'curr_infectious'
    SEVEN_DAY_SMOOTHEN = 'seven_day_infec'
    CUM_INFECTIONS = 'cum_infec'
    DAILY_DEATHS = 'daily_deaths'
    CUM_DEATHS = 'cum_deaths'
    DAILY_RECOVERIES = 'daily_rec'
    CUM_RECOVERIES = 'R'
    ACTIVE_CASES = 'active_cases'
    ADJ_ACT_CASES = 'adjusted_active_cases'
    SEVEN_DAY_INCIDENTS = 'daily_incidents_rate'
    DAILY_VACCINATED = 'daily_vacc'
    CUM_VACCINATED = 'V'
    VACCINATION_PERCENTAGE = 'vacc_percentage'
    VACCINATION_EFFICIENCY = 'vacc_eff'

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


def get_correct_district_name(wrong_name):
    if wrong_name == 'Flensburg':
        return 'Flensburg, Stadt'
    elif wrong_name == 'Kiel':
        return 'Kiel, Landeshauptstadt'
    elif wrong_name == 'Lübeck':
        return 'Lübeck, Hansestadt'
    elif wrong_name == 'Neumünster':
        return 'Neumünster, Stadt'
    elif wrong_name == 'Region Hannover':
        return 'Hannover'
    elif wrong_name == 'Nienburg (Weser)':
        return 'Nienburg/Weser'
    elif wrong_name == 'Oldenburg (Oldb)':
        return 'Oldenburg, Stadt'
    elif wrong_name == 'Oldenburg':
        return 'Oldenburg, Kreis'
    elif wrong_name == 'Osnabrück, Landkreis':
        return 'Osnabrück, Kreis'
    elif wrong_name == 'Städteregion Aachen':
        return 'Aachen'
    elif wrong_name == 'Darmstadt':
        return 'Kreisfreie Stadt Darmstadt'
    elif wrong_name == 'Frankfurt am Main':
        return 'Kreisfreie Stadt Frankfurt am Main'
    elif wrong_name == 'Offenbach am Main':
        return 'Kreisfreie Stadt Offenbach am Main'
    elif wrong_name == 'Wiesbaden':
        return 'Landeshauptstadt Wiesbaden'
    elif wrong_name == 'Hochtaunuskreis':
        return 'Hochtaunus'
    elif wrong_name == 'Main-Kinzig-Kreis':
        return 'Main-Kinzig'
    elif wrong_name == 'Main-Taunus-Kreis':
        return 'Main-Taunus'
    elif wrong_name == 'Rheingau-Taunus-Kreis':
        return 'Rheingau-Taunus'
    elif wrong_name == 'Wetteraukreis':
        return 'Wetterau'
    elif wrong_name == 'Lahn-Dill-Kreis':
        return 'Lahn-Dill'
    elif wrong_name == 'Vogelsbergkreis':
        return 'Vogelsberg'
    elif wrong_name == 'Kassel, Stadt':
        return 'Kreisfreie Stadt Kassel'
    elif wrong_name == 'Kassel, Landkreis':
        return 'Kassel'
    elif wrong_name == 'Schwalm-Eder-Kreis':
        return 'Schwalm-Eder'
    elif wrong_name == 'Werra-Meißner-Kreis':
        return 'Werra-Meißner'
    elif wrong_name == 'Koblenz':
        return 'Stadt Koblenz'
    elif wrong_name == 'Altenkirchen (Westerwald)':
        return 'Altenkirchen (Ww)'
    elif wrong_name == 'Trier':
        return 'Stadt Trier'
    elif wrong_name == 'Frankenthal (Pfalz)':
        return 'Stadt Frankenthal (Pfalz)'
    elif wrong_name == 'Kaiserslautern, Stadt':
        return 'Stadt Kaiserslautern'
    elif wrong_name == 'Landau in der Pfalz':
        return 'Stadt Landau in der Pfalz'
    elif wrong_name == 'Ludwigshafen am Rhein':
        return 'Stadt Ludwigshafen a. Rh.'
    elif wrong_name == 'Mainz':
        return 'Stadt Mainz'
    elif wrong_name == 'Neustadt an der Weinstraße':
        return 'Südliche Weinstraße'
    elif wrong_name == 'Pirmasens':
        return 'Stadt Pirmasens'
    elif wrong_name == 'Speyer':
        return 'Stadt Speyer'
    elif wrong_name == 'Worms':
        return 'Stadt Worms'
    elif wrong_name == 'Zweibrücken':
        return 'Stadt Zweibrücken'
    elif wrong_name == 'Kaiserslautern, Landkreis':
        return 'Kaiserslautern'
    elif wrong_name == 'Heilbronn, Landkreis':
        return 'Heilbronn, Kreis'
    elif wrong_name == 'Karlsruhe, Landkreis':
        return 'Karlsruhe, Kreis'
    elif wrong_name == 'München, Stadt':
        return 'München, Landeshauptstadt'
    elif wrong_name == 'Mühldorf a. Inn':
        return 'Mühldorf a.Inn'
    elif wrong_name == 'München, Landkreis':
        return 'München, Kreis'
    elif wrong_name == 'Pfaffenhofen a.d. Ilm':
        return 'Pfaffenhofen a.d.Ilm'
    elif wrong_name == 'Rosenheim, Landkreis':
        return 'Rosenheim, Kreis'
    elif wrong_name == 'Landshut, Landkreis':
        return 'Landshut, Kreis'
    elif wrong_name == 'Passau, Landkreis':
        return 'Passau, Kreis'
    elif wrong_name == 'Weiden i.d. OPf.':
        return 'Weiden i.d.OPf.'
    elif wrong_name == 'Neumarkt i.d. OPf.':
        return 'Neumarkt i.d.OPf.'
    elif wrong_name == 'Neustadt a.d. Waldnaab':
        return 'Neustadt a.d.Waldnaab'
    elif wrong_name == 'Regensburg, Landkreis':
        return 'Regensburg, Kreis'
    elif wrong_name == 'Bamberg, Landkreis':
        return 'Bamberg, Kreis'
    elif wrong_name == 'Bayreuth, Landkreis':
        return 'Bayreuth, Kreis'
    elif wrong_name == 'Coburg, Landkreis':
        return 'Coburg, Kreis'
    elif wrong_name == 'Hof, Landkreis':
        return 'Hof, Kreis'
    elif wrong_name == 'Wunsiedel i. Fichtelgebirge':
        return 'Wunsiedel i.Fichtelgebirge'
    elif wrong_name == 'Ansbach, Landkreis':
        return 'Ansbach, Kreis'
    elif wrong_name == 'Fürth, Landkreis':
        return 'Fürth, Kreis'
    elif wrong_name == 'Neustadt a.d. Aisch-Bad Windsheim':
        return 'Neustadt a.d.Aisch-Bad Windsheim'
    elif wrong_name == 'Aschaffenburg, Landkreis':
        return 'Aschaffenburg, Kreis'
    elif wrong_name == 'Schweinfurt, Landkreis':
        return 'Schweinfurt, Kreis'
    elif wrong_name == 'Würzburg, Landkreis':
        return 'Würzburg, Kreis'
    elif wrong_name == 'Augsburg, Landkreis':
        return 'Augsburg, Kreis'
    elif wrong_name == 'Dillingen a.d. Donau':
        return 'Dillingen a.d.Donau'
    elif wrong_name == 'Merzig-Wadern':
        return 'Landkreis Merzig-Wadern'
    elif wrong_name == 'Neunkirchen':
        return 'Landkreis Neunkirchen'
    elif wrong_name == 'Saarlouis':
        return 'Landkreis Saarlouis'
    elif wrong_name == 'St. Wendel':
        return 'Landkreis St. Wendel'
    elif wrong_name == 'Rostock, Stadt':
        return 'Rostock, Hansestadt'
    elif wrong_name == 'Schwerin':
        return 'Schwerin, Landeshauptstadt'
    elif wrong_name == 'Rostock, Landkreis':
        return 'Landkreis Rostock'
    elif wrong_name == 'Leipzig, Landkreis':
        return 'Leipzig, Kreis'
    else:
        return wrong_name


def compute_end_date_of_validation_period(train_end_date, duration):
    current_day = datetime.datetime.strptime(train_end_date, '%Y-%m-%d')
    current_day = current_day + datetime.timedelta(days=duration)

    return current_day.strftime('%Y-%m-%d')


def print_progress(completed, total, extra=''):
    progress = round((completed / total) * 100 / 10)
    progress_str = "["

    for i in range(progress):
        progress_str = progress_str + "="
    for i in range(10 - progress):
        progress_str = progress_str + " "

    progress_str = progress_str + "]"
    sys.stdout.write('\r')
    sys.stdout.write('\r' + progress_str + " " + str(round(completed / total * 100, 2)) + "% " + extra)
