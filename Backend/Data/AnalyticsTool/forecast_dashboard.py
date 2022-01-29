import json

import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Backend.Data.DataManager.data_util import create_dates_array
from Backend.Data.DataManager.db_calls import get_district_forecast_data
from Backend.Data.DataManager.db_functions import get_table_data


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# data for the chloropath maps and set ids
german_districts = json.load(open("Data/AnalyticsTool/simplified_geo_data.geojson", 'r', encoding='utf-8'))
dist_id = 1000
state_id_map = {}
for feature in german_districts["features"]:
    feature["id"] = dist_id
    state_id_map[feature["properties"]["GEN"]] = feature["id"]
    dist_id = dist_id + 1

# forecast data loading
all_district_forecasts = pd.DataFrame()


# default dist_list subset
# district_list = ['Münster', 'Potsdam', 'Segeberg', 'Rosenheim, Kreis', 'Hochtaunus', 'Dortmund', 'Essen', 'Bielefeld',
#                  'Warendorf', 'München, Landeshauptstadt']

district_list = get_table_data("district_list", 0, 0, "district", False)
district_list = district_list.sort_values("district", ascending=True)
district_list = district_list['district']


########### app layout is defined here ##############

app.layout = html.Div([
    html.Hr(),
    html.Div(
        className="row",
        children=[
            html.Div(
                className="six columns",
                children=[html.Div([
                            html.Div(
                                children=[
                                            dcc.Dropdown(
                                                    id='district-dropdown',
                                                    options=[{'label': k, 'value': k} for k in district_list],
                                                    multi=False,
                                                    value='Münster'
                                                ),
                                            html.Hr(),
                                            dcc.RadioItems(
                                                id='model-radio',
                                                options=[
                                                    {'label': 'SEVIR + last beta', 'value': 'sevir_last_beta'},
                                                    {'label': 'SEVIR + ML beta', 'value': 'sevir_ml_beta'},
                                                    {'label': 'SARIMA', 'value': 'sarima'},
                                                    {'label': 'Ensemble', 'value': 'ensemble'},
                                                ],
                                                value='sevir_last_beta', className="six columns"
                                            ),
                                            dcc.RadioItems(
                                                id='show-type-radio',
                                                options=[
                                                    {'label': 'show intervals', 'value': 'intervals'},
                                                    {'label': 'show fitted', 'value': 'fitted'}
                                                ],
                                                value='intervals',
                                            ),
                                        ]),
                            html.Hr(),
                            dcc.Graph(
                                id='dist-forecast-graph',
                                figure={
                                    'layout': {
                                        'height': 400,
                                        'margin': {'l': 10, 'b': 10, 't': 10, 'r': 10},
                                        'paper_bgcolor': '#7FDBFF',
                                        'plot_bgcolor': '#7FDBFF',
                                    }
                                }
                            ),

                        ])]
            ),
            html.Div(
                className="six columns",
                children=[
                    html.Div(
                        children=dcc.Graph(id='forecast-chloropath', figure={})
                    )
                ]
            ),
        ]
    )
])


@app.callback(
    Output(component_id='dist-forecast-graph', component_property='figure'),
    Input(component_id='district-dropdown', component_property='value'),
    Input(component_id='model-radio', component_property='value'),
    Input(component_id='show-type-radio', component_property='value')
)
def get_dist_forecast_plot(district, model, show_type):
    dist_forecast_df = get_district_forecast_data(district)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates_array = create_dates_array(start_date_str=dist_forecast_df['date'][0],
                                     num_days=len(dist_forecast_df['cases'].dropna()) + 14,
                                     month_day_only=False)

    # Add traces
    y_common_train = dist_forecast_df['cases'].dropna()
    fig.add_trace(
        go.Scatter(x=dates_array[:28], y=y_common_train, name="train data"),
        secondary_y=False,
    )

    if model == 'sevir_last_beta':
        y = dist_forecast_df['y_pred_seirv_last_beta_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       name="forecast data"),
            secondary_y=False,
        )

    if model == 'sevir_ml_beta':
        y = dist_forecast_df['y_pred_seirv_ml_beta_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       name="forecast data"),
            secondary_y=False,
        )

    if model == 'sarima':
        y = dist_forecast_df['y_pred_sarima_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       name="forecast data"),
            secondary_y=False,
        )

    if model == 'ensemble':
        y = dist_forecast_df['y_pred_ensemble_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       name="forecast data"),
            secondary_y=False,
        )

    # Add figure title
    fig.update_layout(
        title_text="Forecast for " + district
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Dates")

    return fig


# @app.callback(
#     Output(component_id='forecast-chloropath', component_property='figure'),
#     Input(component_id='district-dropdown', component_property='value')
# )
# def create_chloropath_map(district):
#     # Create figure
#     fig = go.Figure()
#
#     # Add traces, one for each slider step
#     for step in np.arange(0, 5, 0.1):
#         fig.add_trace(
#             go.Scatter(
#                 visible=False,
#                 line=dict(color="#00CED1", width=6),
#                 name="𝜈 = " + str(step),
#                 x=np.arange(0, 10, 0.01),
#                 y=np.sin(step * np.arange(0, 10, 0.01))))
#
#     # Make 10th trace visible
#     fig.data[10].visible = True
#
#     # Create and add slider
#     steps = []
#     for i in range(len(fig.data)):
#         step = dict(
#             method="update",
#             args=[{"visible": [False] * len(fig.data)},
#                   {"title": "Slider switched to step: " + str(i)}],  # layout attribute
#         )
#         step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
#         steps.append(step)
#
#     sliders = [dict(
#         active=10,
#         currentvalue={"prefix": "Frequency: "},
#         pad={"t": 50},
#         steps=steps
#     )]
#
#     fig.update_layout(
#         sliders=sliders
#     )
#
#     return fig



####################################################

# app.layout = html.Div([
#     dcc.Dropdown(
#         id='district-dropdown',
#         options=[{'label': k, 'value': k} for k in district_list],
#         multi=False,
#         value='Münster'
#     ),
#
#     html.Hr(),
#
#     html.Div([
#         dcc.RadioItems(
#                 options=[
#                     {'label': 'New York City', 'value': 'NYC'},
#                     {'label': 'Montréal', 'value': 'MTL'},
#                     {'label': 'San Francisco', 'value': 'SF'}
#                 ],
#                 value='MTL', className='five columns'
#             ),
#         dcc.Graph(id='incidents-correl-heat', figure={}, className='five columns',)
#
#     ])
    #
    # dcc.Dropdown(id='top-dist-dropdown', multi=True, value='Münster'),
    #
    # html.Hr(),
    #
    # dcc.Dropdown(id='attr-list-dropdown',
    #              multi=False,
    #              options=[{'label': k, 'value': k} for k in attr_list],
    #              value='cum_vacc'),
    #
    # html.Hr(),
    #
    # html.Div([
    #     dcc.Graph(id='attr-graph', figure={}, clickData=None, hoverData=None,
    #               # I assigned None for demo purposes. By default, these are None, unless you specify otherwise.
    #               config={
    #                   'staticPlot': False,  # True, False
    #                   'scrollZoom': True,  # True, False
    #                   'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
    #                   'showTips': False,  # True, False
    #                   'displayModeBar': True,  # True, False, 'hover'
    #                   'watermark': True,
    #                   # 'modeBarButtonsToRemove': ['pan2d','select2d'],
    #               }
    #               )
    # ])

#
# @app.callback(
#     dash.dependencies.Output('top-dist-dropdown', 'options'),
#     [dash.dependencies.Input('district-dropdown', 'value')])
# def set_cities_options(selected_country):
#     return [{'label': i, 'value': i} for i in get_top_relation(selected_country)]
#
#
# @app.callback(
#     Output(component_id='attr-graph', component_property='figure'),
#     Input(component_id='district-dropdown', component_property='value'),
#     Input(component_id='top-dist-dropdown', component_property='value'),
#     Input(component_id='attr-list-dropdown', component_property='value'),
# )
# def update_graph(dist, dists_chosen, attr_chosen):
#     if type(dists_chosen) is str:
#         dists_chosen = [dists_chosen]
#
#     if type(dists_chosen) is None:
#         dists_chosen = [dist]
#
#     dff = all_district_data[all_district_data.district.isin(dists_chosen)]
#     fig = px.line(data_frame=dff, x='date', y=attr_chosen, color='district',
#                   custom_data=['daily_vacc', 'cum_vacc', 'district'])
#     fig.update_traces(mode='lines+markers')
#     return fig


if __name__ == '__main__':
    app.run_server()
