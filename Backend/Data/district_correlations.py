import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from Backend.Data.db_functions import get_table_data, update_district_matrices

# updating all_district_data should be executed before

list_of_districts = get_table_data("district_list", 0, 0, "district", False)
list_of_districts = list_of_districts['district'].to_list()

district_incidents_matrix = {}

# terminator = 1

for district in list_of_districts:

    district_data = get_table_data(district, 0, 0, "daily_incidents_rate", False)

    district_data = district_data['daily_incidents_rate'].to_numpy()
    district_data = [float(i) for i in district_data]

    district_incidents_matrix[district] = district_data

    # if terminator == 10:
    #     break
    #
    # terminator += 1

all_data_df = pd.DataFrame(district_incidents_matrix)

update_district_matrices('district', 'incidents', all_data_df)

# all_districts_correlation = all_data_df.corr(method='pearson')
# print(all_districts_correlation['Warendorf'][:5])
# reloaded_data = get_table_data('cor_matrix_incidents_districts', 0, 0, 'Warendorf', True)
# reloaded_data = reloaded_data.sort_values('Warendorf', ascending = False)
# print(reloaded_data['Warendorf'][:10])
# print(reloaded_data['Warendorf'][:5].index.tolist())
# update_district_matrices('districts', 'incidents', all_districts_correlation)

# sn.heatmap(all_districts_correlation, annot=True)
# plt.show()

# print(all_data_df)

# city1 = get_table_data("Münster", 0, 0, "daily_incidents_rate")
# city2 = get_table_data("Hamm", 0, 0, "daily_incidents_rate")
#
# city1 = city1['daily_incidents_rate'].to_numpy()
# city1 = [float(i) for i in city1]
# print(city1)
#
# city2 = city2['daily_incidents_rate'].to_numpy()
# city2 = [float(i) for i in city2]
# print(city2)
#
# correlation_btw_cities = np.corrcoef(city1, city2)

# print(district_incidents_matrix)
