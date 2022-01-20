import json

import pandas as pd
import numpy as np
import math
import plotly.express as px

from Backend.Data.DataManager.db_functions import get_table_data, update_district_matrices, get_filtered_table_data

# updating all_district_data should be executed before
list_of_districts = get_table_data("district_list", 0, 0, "district", False)
list_of_districts = list_of_districts['district'].to_list()

list_of_dates = get_table_data("Münster", 0, 0, "date", False)
list_of_dates = list_of_dates['date'].to_list()


def dist_correlation_by_incidents():
    district_incidents_matrix = {}

    for district in list_of_districts:
        district_data = get_table_data(district, 0, 0, "daily_incidents_rate", False)

        district_data = district_data['daily_incidents_rate'].to_numpy()
        district_data = [float(i) for i in district_data]

        district_incidents_matrix[district] = district_data

    all_data_df = pd.DataFrame(district_incidents_matrix)
    all_districts_correlation = all_data_df.corr(method='pearson')

    update_district_matrices('districts', 'incidents', all_districts_correlation, 'district_name')


def date_correlation_by_vac_incidents():
    vac_incidents_matrix = {}
    all_incidents = {}
    all_vacc = {}

    all_incidents['date'] = list_of_dates
    all_vacc['date'] = list_of_dates

    for district in list_of_districts:
        dist_data = get_table_data(district, 0, 0, ['daily_incidents_rate', 'vacc_percentage'], False)
        all_incidents[district] = dist_data['daily_incidents_rate'].to_list()
        all_vacc[district] = dist_data['vacc_percentage'].to_list()

    all_incidents_df = pd.DataFrame(all_incidents)
    all_vacc_df = pd.DataFrame(all_vacc)

    for date in list_of_dates:
        incident_row = all_incidents_df.loc[all_incidents_df['date'] == date]
        incident_row = ((incident_row.loc[:, incident_row.columns != 'date']).values.tolist())[0]
        incident_row = [float(i) for i in incident_row]

        vacc_row = all_vacc_df.loc[all_vacc_df['date'] == date]
        vacc_row = ((vacc_row.loc[:, vacc_row.columns != 'date']).values.tolist())[0]
        vacc_row = [float(i) for i in vacc_row]

        correl_coef = np.corrcoef(incident_row, vacc_row)[0, 1]

        if math.isnan(correl_coef):
            # need to decide a value here
            correl_coef = 0.0

        vac_incidents_matrix[date] = correl_coef

    all_data_df = pd.DataFrame(vac_incidents_matrix, index=[0])
    all_data_df = all_data_df.T
    all_data_df.rename(columns={all_data_df.columns[0]: 'correlation'}, inplace=True)
    update_district_matrices('date', 'vac_to_incidents', all_data_df, 'date')


def dist_daily_incidents_vacc_corre():
    dist_correl_values = {}

    for district in list_of_districts:
        dist_data = get_table_data(district, 0, 0, ['date', 'daily_incidents_rate', 'vacc_percentage'], False)

        incidents_array = []
        vacc_array = []
        daily_correl_values = {}

        for index, row in dist_data.iterrows():

            incidents_array.append(float(row['daily_incidents_rate']))
            vacc_array.append(float(row['vacc_percentage']))

            correl_coef = np.corrcoef(incidents_array, vacc_array)[0, 1]
            if math.isnan(correl_coef):
                # need to decide a value here
                correl_coef = 0.0

            daily_correl_values[row['date']] = correl_coef
        dist_correl_values[district] = daily_correl_values

    all_data_df = pd.DataFrame(dist_correl_values)
    update_district_matrices('dist_daily', 'vac_to_incidents', all_data_df, 'date')
    # all_district_data = pd.DataFrame(attributes[:], columns=['date', 'value', 'attribute'])


def dist_incidents_vacc_corre():
    dist_correl_values = {}

    for district in list_of_districts:
        dist_data = get_table_data(district, 0, 0, ['date', 'daily_incidents_rate', 'vacc_percentage'], False)

        incidents_array = dist_data['daily_incidents_rate'].to_list()
        vacc_array = dist_data['vacc_percentage'].to_list()

        previous_val = 0.0
        counter = 0

        for value in vacc_array:
            if float(value) > 0.0 and previous_val > 00.0:
                counter = counter+1
                previous_val = float(value)
                break

            previous_val = float(value)
            counter = counter + 1

        incidents_array = incidents_array[counter:]
        vacc_array = vacc_array[counter:]

        correl_coef = np.corrcoef([float(i) for i in incidents_array], [float(i) for i in vacc_array])[0, 1]
        if math.isnan(correl_coef):
            # need to decide a value here
            correl_coef = 0.0
        if previous_val > 00.0:
            dist_correl_values[district] = correl_coef

    all_data_df = pd.DataFrame(dist_correl_values, index=[0])
    all_data_df = all_data_df.T
    all_data_df.rename(columns={all_data_df.columns[0]: 'correlation'}, inplace=True)
    all_data_df['district'] = all_data_df.index

    german_districts = json.load(open("simplified_geo_data.geojson", 'r', encoding='utf-8'))
    dist_id = 1000

    state_id_map = {}
    for feature in german_districts["features"]:
        feature["id"] = dist_id
        state_id_map[feature["properties"]["GEN"]] = feature["id"]
        dist_id = dist_id + 1

    all_data_df["id"] = all_data_df["district"].apply(lambda x: state_id_map[x])

    fig = px.choropleth_mapbox(
        all_data_df,
        locations="id",
        geojson=german_districts,
        color='correlation',
        hover_name='district',
        hover_data=['correlation'],
        title="District wise of Incident to Vaccination Percentage",
        mapbox_style="carto-positron",
        center={"lat": 51.1657, "lon": 10.4515},
        zoom=4.8,
        opacity=0.9,
        width=700,
        height=700,

    )
    fig.show()
    # update_district_matrices('dist_daily', 'vac_to_incidents', all_data_df, 'date')
    # all_district_data = pd.DataFrame(attributes[:], columns=['date', 'value', 'attribute'])


if __name__ == '__main__':
    # date_correlation_by_vac_incidents()
    # dist_daily_incidents_vacc_corre()
    dist_incidents_vacc_corre()

    # df = get_table_data('cor_matrix_vac_to_incidents_date', 0, 0, True, False)
    # fig = px.line(df, x="date", y="correlation", title='correlation of incidents to vaccination percentage')
    # fig.show()

    # all_districts_correlation = all_data_df.corr(method='pearson')
    # print(all_districts_correlation['Warendorf'][:5])
    # reloaded_data = get_table_data('cor_matrix_incidents_districts', 0, 0, 'Warendorf', True)
    # reloaded_data = reloaded_data.sort_values('Warendorf', ascending = False)
    # print(reloaded_data['Warendorf'][:10])
    # print(reloaded_data['Warendorf'][:5].index.tolist())
    # update_district_matrices('districts', 'incidents', all_districts_correlation)
