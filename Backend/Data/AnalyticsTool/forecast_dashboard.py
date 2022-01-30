import json

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import plotly
import plotly.offline as offline
import plotly.io as pio

from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
from Backend.Data.DataManager.data_util import create_dates_array
from Backend.Data.DataManager.db_calls import get_district_forecast_data, get_all_latest_forecasts
from Backend.Data.DataManager.db_functions import get_table_data

pio.templates.default = 'plotly_dark'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', '../Assets/style.css']
#external_stylesheets = ["../Assets/style_dark.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app = dash.Dash(external_stylesheets=[dbc.themes.VAPOR])
# data for the chloropath maps and set ids
german_districts = json.load(open("Data/AnalyticsTool/simplified_geo_data.geojson", 'r', encoding='utf-8'))
dist_id = 1000
state_id_map = {}
for feature in german_districts["features"]:
    feature["id"] = dist_id
    state_id_map[feature["properties"]["GEN"]] = feature["id"]
    dist_id = dist_id + 1

# forecast data loading
all_district_forecasts = get_all_latest_forecasts()



# default dist_list subset
# district_list = ['Münster', 'Potsdam', 'Segeberg', 'Rosenheim, Kreis', 'Hochtaunus', 'Dortmund', 'Essen', 'Bielefeld',
#                  'Warendorf']

district_list = get_table_data("district_list", 0, 0, "district", False)
district_list = district_list.sort_values("district", ascending=True)
district_list = district_list['district']

dates_list = all_district_forecasts['date'].unique()[-14:]


########### app layout is defined here ##############

app.layout = html.Div([
    html.Hr(style={'backgroundColor':'#111111'},),
    html.H2('Regional COVID-19 Forecasting Tool', style={'backgroundColor':'#111111', 'color':'white'}),
    html.Div(
        className="row",
        style={'backgroundColor':'#111111', 'color':'white'},
        children=[
            html.Div(
                className="six columns",
                children=[
                    #html.H2('Dash - STOCK PRICES'),
                    html.Div([
                            html.Div(
                                children=[
                                            dcc.Dropdown(
                                                    id='district-dropdown',
                                                    options=[{'label': k, 'value': k} for k in district_list],
                                                    multi=False,
                                                    value='Münster',
                                                    style={'backgroundColor':'#111111', 'color':'#ffffff'},
                                                    #style=dict(color='green'),,
                                                ),
                                            html.Hr(),
                                            # dcc.RadioItems(
                                            #     id='model-radio',
                                            #     options=[
                                            #         {'label': 'SEVIR + last beta', 'value': 'sevir_last_beta'},
                                            #         {'label': 'SEVIR + ML beta', 'value': 'sevir_ml_beta'},
                                            #         {'label': 'SARIMA', 'value': 'sarima'},
                                            #         {'label': 'Ensemble', 'value': 'ensemble'},
                                            #     ],
                                            #     value='sevir_last_beta', className="six columns"
                                            # ),
                                            dcc.Checklist(
                                                id='model-check',
                                                options=[
                                                    {'label': 'SEVIR(last beta)', 'value': 'sevir_last_beta'},
                                                    {'label': 'SEVIR(ML beta)', 'value': 'sevir_ml_beta'},
                                                    {'label': 'ARIMA', 'value': 'sarima'},
                                                    {'label': 'Ensemble', 'value': 'ensemble'},
                                                ],
                                                value='sevir_last_beta',
                                            ),
                                            dcc.Checklist(
                                                id='show-interval-check',
                                                options=[
                                                    {'label': 'show intervals', 'value': 'intervals'},
                                                ],
                                                value='intervals',
                                            ),
                                        ]),
                            html.Hr(),
                            dcc.Graph(
                                id='dist-forecast-graph',
                                figure={
                                    'layout': {
                                        'height': 500,
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
                        children=[
                            dcc.Dropdown(
                                id='map-forecast-model',

                                options=[{'label': 'SEIRV(Last beta)', 'value': 'y_pred_seirv_last_beta_mean'},
                                         {'label': 'SEIRV(ML beta)', 'value': 'y_pred_seirv_ml_beta_mean'},
                                         {'label': 'SARIMA', 'value': 'y_pred_sarima_mean'},
                                         {'label': 'Ensemble', 'value': 'y_pred_ensemble_mean'}],
                                multi=False,
                                value='y_pred_seirv_last_beta_mean',
                                style={'backgroundColor':'#111111','font-color':'white'},
                            ),
                            dcc.Graph(id='forecast-chloropath', figure={}),
                        ]
                    )
                ]
            ),
        ]
    )
], style={'backgroundColor': '#111111'})


@app.callback(
    Output(component_id='dist-forecast-graph', component_property='figure'),
    Input(component_id='district-dropdown', component_property='value'),
    # Input(component_id='model-radio', component_property='value'),
    Input(component_id='model-check', component_property='value'),
    Input(component_id='show-interval-check', component_property='value')
)
def get_dist_forecast_plot(district, checkbox, show_interval):
    dist_forecast_df = get_district_forecast_data(district)
    training_len = len(dist_forecast_df['cases'].dropna())
    shown = 21
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates_array = create_dates_array(start_date_str=dist_forecast_df['date'][0],
                                     num_days=len(dist_forecast_df['cases'].dropna()) + 14)

    # Add traces
    y_common_train = dist_forecast_df['cases'][training_len-shown:training_len].dropna()
    fig.add_trace(
        go.Scatter(x=dates_array[training_len-shown:training_len], y=y_common_train, mode='lines+markers', name='training'),
        secondary_y=False,
    )

    if ('sevir_last_beta' in checkbox):
        y = dist_forecast_df['y_pred_seirv_last_beta_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       name="SEIURV last beta",
                       line_color='rgb(0,100,80)',
                       mode='lines'),
            secondary_y=False,
        )
    if ('sevir_ml_beta' in checkbox):
        y = dist_forecast_df['y_pred_seirv_ml_beta_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       name="SEIURV ML beta",
                       line_color='rgb(0,176,246)',
                       mode='lines'),
            secondary_y=False,
        )
    if ('sarima' in checkbox):
        y = dist_forecast_df['y_pred_sarima_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       line_color='rgb(231,107,243)',
                       name="ARIMA",
                       mode='lines'),
                       # line = dict(color='green')),
            secondary_y=False,
        )
    if ('ensemble'in checkbox):
        y = dist_forecast_df['y_pred_ensemble_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       line_color='rgb(230,171,2)',
                       name="Ensemble",
                       mode='lines'),

            secondary_y=False,
        )

    # add upper and lower bounds
    if ('sevir_last_beta' in checkbox and 'intervals'in show_interval):
        y_upper = dist_forecast_df['y_pred_seirv_last_beta_upper'].dropna().tolist()
        y_lower = dist_forecast_df['y_pred_seirv_last_beta_lower'].dropna().tolist()
        y_lower = y_lower[::-1]
        x = list(dates_array[-15:])
        x_rev = x[::-1]
        x = x+x_rev
        y_interval = y_upper + y_lower
        fig.add_trace(go.Scatter(x=x,
                                 y=y_interval,
                                 fill='toself',
                                 fillcolor='rgba(0,100,80,0.2)',
                                 line_color='rgba(255,255,255,0)',
                                 name="SEIURV last beta",
                                 showlegend=False))
    if ('sevir_ml_beta' in checkbox and 'intervals'in show_interval):
        y_upper = dist_forecast_df['y_pred_seirv_ml_beta_upper'].dropna().tolist()
        y_lower = dist_forecast_df['y_pred_seirv_ml_beta_lower'].dropna().tolist()
        y_lower = y_lower[::-1]
        x = list(dates_array[-15:])
        x_rev = x[::-1]
        x = x + x_rev
        y_interval = y_upper + y_lower
        fig.add_trace(go.Scatter(x=x,
                                 y=y_interval,
                                 fill='toself',
                                 fillcolor='rgba(0,176,246,0.2)',
                                 line_color='rgba(255,255,255,0)',
                                 name="SEIURV ML beta",
                                 showlegend=False))
    if ('sarima' in checkbox and 'intervals'in show_interval):
        y_upper = dist_forecast_df['y_pred_sarima_upper'].dropna().tolist()
        y_lower = dist_forecast_df['y_pred_sarima_lower'].dropna().tolist()
        y_lower = y_lower[::-1]
        x = list(dates_array[-15:])
        x_rev = x[::-1]
        x = x + x_rev
        y_interval = y_upper + y_lower
        fig.add_trace(go.Scatter(x=x,
                                 y=y_interval,
                                 fill='toself',
                                 fillcolor='rgba(231,107,243,0.2)',
                                 line_color='rgba(255,255,255,0)',
                                 name='ARIMA',
                                 showlegend=False))
    if ('ensemble' in checkbox and 'intervals'in show_interval):
        y_upper = dist_forecast_df['y_pred_ensemble_upper'].dropna().tolist()
        y_lower = dist_forecast_df['y_pred_ensemble_lower'].dropna().tolist()
        y_lower = y_lower[::-1]
        x = list(dates_array[-15:])
        x_rev = x[::-1]
        x = x + x_rev
        y_interval = y_upper + y_lower
        fig.add_trace(go.Scatter(x=x,
                                 y=y_interval,
                                 fill='toself',
                                 fillcolor='rgba(230,171,2,0.2)',
                                 line_color='rgba(255,255,255,0)',
                                 name='Ensemble',
                                 showlegend=False))


    # Add figure title
    fig.update_layout(
        title_text="Forecast for " + district
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Dates")

    #set y-axis title
    non_numeric_cols = ['prediction_id', 'pipeline_id', 'district_name', 'date']
    sel_columns = [c for c in list(dist_forecast_df.columns) if c not in non_numeric_cols]
    temp_df = dist_forecast_df[sel_columns].fillna(value=0).apply(pd.to_numeric)
    max_train_data_points = np.max(temp_df.max())

    max_y_axis_value = round(max_train_data_points*1.2/100)*100

    fig.update_yaxes (title_text='7 Day Incidence', range=[0, max_y_axis_value])

    return fig

@app.callback(
    Output(component_id='forecast-chloropath', component_property='figure'),
    Input(component_id='map-forecast-model', component_property='value')
)
def get_dist_forecast_plot(selected_model):
    # data_for_map_df = all_district_forecasts.loc[all_district_forecasts['date'] == dates_list[selected_date]]
    data_for_map_df = all_district_forecasts[all_district_forecasts['cases'].isna()]
    data_for_map_df["id"] = data_for_map_df["district_name"].apply(lambda x: state_id_map[x])
    data_for_map_df[selected_model] = data_for_map_df[selected_model] \
        .apply(pd.to_numeric).round(decimals=2)

    forecast_map = px.choropleth_mapbox(
        data_for_map_df,
        locations="id",
        geojson=german_districts,
        color=selected_model,
        hover_name='district_name',
        hover_data=['district_name'],
        title="Next 14-Day Incident Number",
        mapbox_style="carto-darkmatter",
        # hot blackbody thermal
        # color_continuous_scale=[(0,'#921315'),
        #                         (0.2,'#661313'),
        #                         (0.4,'#d90183'),
        #                         (0.6,'#fe72C5'),
        #                         (0.8,'#ffC5E8'),
        #                         (1,'#780175')],
        # color_continuous_scale= {("0 - 250", "red"), ("251 - 500", "red"),
        #                          ("501 - 750", "green"), ("751 - 1,000", "green"),
        #                          ("1,001 and higher", "blue")},
        color_discrete_map={
            '0 - 250': '#921315',
            '251 - 500': '#661313',
            '501 - 750': '#d90183',
            '751 - 1,000': '#fe72C5',
            '1,001 and higher': '#ffC5E8'},
        # category_orders={
        #     'y_pred_seirv_ml_beta_mean': [
        #         '0 - 250',
        #         '251 - 500',
        #         '501 - 750',
        #         '751 - 1,000',
        #         '1,001 and higher'
        #     ]
        # },
        # range_color=(0, 2000),
        animation_frame='date',
        center={"lat": 51.1657, "lon": 10.4515},
        zoom=5,
        opacity=0.85,
        width=830,
        height=830,

    )

    return forecast_map

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
