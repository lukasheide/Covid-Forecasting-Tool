from Backend.Data.data_util import Column
from Backend.Data.db_calls import get_table_data_by_duration, get_table_data_by_day, get_district_data
from Backend.Modeling.Differential_Equation_Modeling.model_params import params_SEIRV_fixed


def get_smoothen_cases(district, end_date, duration):
    data_result = get_table_data_by_duration(table=district,
                                             end_date=end_date,
                                             duration=duration-1,       # otherwise for a duration of two days we would get data from e.g. 15th Dec - 17th Dec
                                             attributes=[Column.DATE.value, Column.SEVEN_DAY_SMOOTHEN.value])

    return data_result


# assumes DB tables are up-to_date
def get_starting_values(district, train_start_date):
    district_status = get_table_data_by_day(table=district, date=train_start_date,
                                            attributes=[Column.CUM_VACCINATED.value, Column.CUM_RECOVERIES.value])
    district_details = get_district_data(district=district, attributes=[Column.POPULATION.value])
    vaccinated, recovered, population = district_status[Column.CUM_VACCINATED.value].to_list()[0], \
                                        district_status[Column.CUM_RECOVERIES.value].to_list()[0], \
                                        district_details[Column.POPULATION.value].to_list()[0]

    return population, vaccinated, recovered


def get_model_params(district, train_start_date):

    model_params = {
        # Get fixed model params:
        'gamma_I': params_SEIRV_fixed['gamma_I']['mean'],
        'gamma_U': params_SEIRV_fixed['gamma_U']['mean'],
        'delta': params_SEIRV_fixed['delta']['mean'],
        'rho': params_SEIRV_fixed['rho']['mean'],
        # Get theta for train_start_date from db:
        'theta': get_table_data_by_day(table=district, date=train_start_date,
                                       attributes=[Column.VACCINATION_EFFICIENCY.value])}

    return model_params



#
#
# # should only be executed once
# def prepare_model_store():
#
#     clean_create_model_store()


if __name__ == '__main__':
    # get_table_data('Bremen', '2020-10-22', '2020-11-22', [Column.ADJ_ACT_CASES.value,
    #                                                       Column.VACCINATION_PERCENTAGE.value,
    #                                                       Column.CURRENT_INFECTIOUS.value])
    # should this be like 'date','seven_day_infec' ?
    print(get_smoothen_cases('Münster', '2021-03-31', 30))
