import json

import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px

from Backend.Data.db_functions import get_table_data

app = dash.Dash(__name__)

district_list = get_table_data('district_list', 0, 0, "district", False)

# data for the chloropath maps
german_districts = json.load(open("simplified_geo_data.geojson", 'r', encoding='utf-8'))
# 'all' for bluff to correct later
all_dis_correl_data = get_table_data('cor_matrix_incidents_districts', 0, 0, 'all', True)
dist_id = 1000

state_id_map = {}
for feature in german_districts["features"]:
    feature["id"] = dist_id
    state_id_map[feature["properties"]["GEN"]] = feature["id"]
    dist_id = dist_id + 1

all_district_data = pd.DataFrame()
attr_list = ['daily_infec',
             'cum_infec',
             'daily_deaths',
             'cum_deaths',
             'daily_rec',
             'cum_rec',
             'adjusted_active_cases',
             'daily_incidents_rate',
             'daily_vacc',
             'cum_vacc',
             'vacc_percentage']


def create_data_set(reloaded_data):
    frames = []

    for district in reloaded_data:
        dis_data = get_table_data(district, 0, 0, ['date',
                                                   'daily_infec',
                                                   'cum_infec',
                                                   'daily_deaths',
                                                   'cum_deaths',
                                                   'daily_rec',
                                                   'cum_rec',
                                                   'adjusted_active_cases',
                                                   'daily_incidents_rate',
                                                   'daily_vacc',
                                                   'cum_vacc',
                                                   'vacc_percentage'], False)
        dis_data['district'] = district
        frames.append(dis_data)

    global all_district_data
    all_district_data = pd.concat(frames)
    all_district_data[['daily_infec',
                       'cum_infec',
                       'daily_deaths',
                       'cum_deaths',
                       'daily_rec',
                       'cum_rec',
                       'adjusted_active_cases',
                       'daily_incidents_rate',
                       'daily_vacc',
                       'cum_vacc',
                       'vacc_percentage']] = all_district_data[['daily_infec',
                                                                'cum_infec',
                                                                'daily_deaths',
                                                                'cum_deaths',
                                                                'daily_rec',
                                                                'cum_rec',
                                                                'adjusted_active_cases',
                                                                'daily_incidents_rate',
                                                                'daily_vacc',
                                                                'cum_vacc',
                                                                'vacc_percentage']].apply(pd.to_numeric)


def get_top_relation(district):
    reloaded_data = get_table_data('cor_matrix_incidents_districts', 0, 0, district, True)
    reloaded_data = reloaded_data.sort_values(district, ascending=False)
    reloaded_data = reloaded_data[district][:10].index.tolist()

    create_data_set(reloaded_data)

    return reloaded_data


get_top_relation('Münster')

app.layout = html.Div([
    dcc.Dropdown(
        id='district-dropdown',
        options=[{'label': k, 'value': k} for k in district_list.district.unique()],
        multi=False,
        value='Münster'
    ),

    html.Hr(),

    html.Div([
        dcc.Graph(id='incidents-correl-heat', figure={})

    ]),

    dcc.Dropdown(id='top-dist-dropdown', multi=True, value='Münster'),

    html.Hr(),

    dcc.Dropdown(id='attr-list-dropdown',
                 multi=False,
                 options=[{'label': k, 'value': k} for k in attr_list],
                 value='cum_vacc'),

    html.Hr(),

    html.Div([
        dcc.Graph(id='attr-graph', figure={}, clickData=None, hoverData=None,
                  # I assigned None for demo purposes. By default, these are None, unless you specify otherwise.
                  config={
                      'staticPlot': False,  # True, False
                      'scrollZoom': True,  # True, False
                      'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                      'showTips': False,  # True, False
                      'displayModeBar': True,  # True, False, 'hover'
                      'watermark': True,
                      # 'modeBarButtonsToRemove': ['pan2d','select2d'],
                  }
                  )
    ])
])


@app.callback(
    dash.dependencies.Output('top-dist-dropdown', 'options'),
    [dash.dependencies.Input('district-dropdown', 'value')])
def set_cities_options(selected_country):
    return [{'label': i, 'value': i} for i in get_top_relation(selected_country)]


@app.callback(
    Output(component_id='attr-graph', component_property='figure'),
    Input(component_id='district-dropdown', component_property='value'),
    Input(component_id='top-dist-dropdown', component_property='value'),
    Input(component_id='attr-list-dropdown', component_property='value'),
)
def update_graph(dist, dists_chosen, attr_chosen):
    if type(dists_chosen) is str:
        dists_chosen = [dists_chosen]

    if type(dists_chosen) is None:
        dists_chosen = [dist]

    dff = all_district_data[all_district_data.district.isin(dists_chosen)]
    fig = px.line(data_frame=dff, x='date', y=attr_chosen, color='district',
                  custom_data=['daily_vacc', 'cum_vacc', 'district'])
    fig.update_traces(mode='lines+markers')
    return fig


@app.callback(
    Output(component_id='incidents-correl-heat', component_property='figure'),
    Input(component_id='district-dropdown', component_property='value')
)
def get_correlation_chloropath(district):
    district_name = district
    dis_correl_data = all_dis_correl_data[district_name].to_frame()
    dis_correl_data['district'] = dis_correl_data.index

    dis_correl_data["id"] = dis_correl_data["district"].apply(lambda x: state_id_map[x])

    print(dis_correl_data)

    fig = px.choropleth_mapbox(
        dis_correl_data,
        locations="id",
        geojson=german_districts,
        color=district_name,
        hover_name='district',
        hover_data=[district_name],
        title="Correlation By Incident Number",
        mapbox_style="carto-positron",
        center={"lat": 51.1657, "lon": 10.4515},
        zoom=4.8,
        opacity=0.9,
        width=700,
        height=700,

    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
