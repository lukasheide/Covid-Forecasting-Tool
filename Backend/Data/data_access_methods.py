from Backend.Data.data_util import Column
from Backend.Data.db_calls import get_table_data_by_duration, get_table_data_by_day, get_district_data


def get_smoothen_cases(district, end_date, duration):
    data_result = get_table_data_by_duration(table=district,
                                             end_date=end_date,
                                             duration=duration,
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


if __name__ == '__main__':
    # get_table_data('Bremen', '2020-10-22', '2020-11-22', [Column.ADJ_ACT_CASES.value,
    #                                                       Column.VACCINATION_PERCENTAGE.value,
    #                                                       Column.CURRENT_INFECTIOUS.value])
    # should this be like 'date','seven_day_infec' ?
    print(get_smoothen_cases('Münster', '2021-03-31', 30))
