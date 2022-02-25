from Backend.Data.DataManager.data_util import Column, date_int_str, date_str_to_int
from Backend.Data.DataManager.db_calls import get_table_data_by_duration, get_table_data_by_day, get_district_data, \
    get_table_data
from Backend.Modeling.Differential_Equation_Modeling.model_params import params_SEIURV_fixed

"""
methods below are serving different pipeline specific data retrieval tasks from DB   
"""


def get_smoothen_cases(district, end_date, duration):
    """
        get seven day moving average of infections of a district for a given duration of time
    """
    data_result = get_table_data_by_duration(table=district,
                                             end_date=end_date,
                                             duration=duration - 1,
                                             # otherwise for a duration of two days we would get data from e.g. 15th Dec - 17th Dec
                                             attributes=[Column.DATE, Column.SEVEN_DAY_SMOOTHEN])

    return data_result


# assumes DB tables are up-to_date
def get_starting_values(district, train_start_date):
    """
        get vaccinated, recovered, population values of a given district in for a given date.
    """
    district_status = get_table_data_by_day(table=district, date=train_start_date,
                                            attributes=[Column.CUM_VACCINATED, Column.CUM_RECOVERIES])
    district_details = get_district_data(district=district, attributes=[Column.POPULATION])
    vaccinated, recovered, population = district_status[Column.CUM_VACCINATED].to_list()[0], \
                                        district_status[Column.CUM_RECOVERIES].to_list()[0], \
                                        district_details[Column.POPULATION].to_list()[0]

    fixed_start_vals = {
        'N': population,
        'V': vaccinated,
        'R': recovered,
    }

    return fixed_start_vals


def get_model_params(district, train_start_date):
    """
        Get theta(vaccination efficiency value) of a given district for a given date.
    """
    theta = get_table_data_by_day(table=district, date=train_start_date,
                                  attributes=[Column.VACCINATION_EFFICIENCY])
    theta = theta[Column.VACCINATION_EFFICIENCY].tolist()[0]
    model_params = {
        # Get fixed model params:
        'gamma_I': params_SEIURV_fixed['gamma_I']['mean'],
        'gamma_U': params_SEIURV_fixed['gamma_U']['mean'],
        'delta': params_SEIURV_fixed['delta']['mean'],
        'theta': theta,
        'rho': params_SEIURV_fixed['rho']['mean'],
    }

    return model_params


def eval_and_get_latest_possible_starting_date(given_date):
    """
        once a forecast start date is given to the forecast start pipeline,
        this method determines whether it is possible to start the forecasting from that day,
        based on the latest input data we have in the DB.
    """

    dates_df = get_table_data('Münster', Column.DATE)
    dates_list = dates_df[Column.DATE].to_list()
    dates_list.sort()
    latest_start_date = dates_list[-1]

    if given_date is None:
        latest_start_date = date_int_str(latest_start_date)
        print("starting with the latest possible date --> " + latest_start_date)
        return latest_start_date
    else:
        given_date_int = date_str_to_int(given_date)
        if int(given_date_int) > int(latest_start_date):
            print("given start date should not be head of today!")
            latest_start_date = date_int_str(latest_start_date)
            print("starting with the latest possible date --> " + latest_start_date)
            return latest_start_date
        else:
            return given_date


if __name__ == '__main__':
    # only for testing methods locally
    print(get_smoothen_cases('Münster', '2021-03-31', 30))
