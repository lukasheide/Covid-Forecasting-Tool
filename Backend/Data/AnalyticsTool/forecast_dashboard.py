import json

import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import plotly
import plotly.offline as offline

from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
from Backend.Data.DataManager.data_util import create_dates_array
from Backend.Data.DataManager.db_calls import get_district_forecast_data, get_all_latest_forecasts
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
                                value='y_pred_seirv_last_beta_mean'
                            ),
                            dcc.Graph(id='forecast-chloropath', figure={}),
                        ]
                    )
                ]
            ),
        ]
    )
])


@app.callback(
    Output(component_id='dist-forecast-graph', component_property='figure'),
    Input(component_id='district-dropdown', component_property='value'),
    # Input(component_id='model-radio', component_property='value'),
    Input(component_id='model-check', component_property='value'),
    Input(component_id='show-type-radio', component_property='value')
)
def get_dist_forecast_plot(district, checkbox, show_type):
    dist_forecast_df = get_district_forecast_data(district)
    training_len = len(dist_forecast_df['cases'].dropna())
    shown = 21
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates_array = create_dates_array(start_date_str=dist_forecast_df['date'][0],
                                     num_days=len(dist_forecast_df['cases'].dropna()) + 14)

    # Add traces
    y_common_train = dist_forecast_df['cases'][training_len-shown:training_len].dropna()
    fig.add_trace(
        go.Scatter(x=dates_array[training_len-shown:training_len], y=y_common_train, name="train data"),
        secondary_y=False,
    )

    if ('sevir_last_beta' in checkbox):
        y = dist_forecast_df['y_pred_seirv_last_beta_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       name="SEIURV last beta",
                       line_color='rgb(0,100,80)'),
            secondary_y=False,
        )
    if ('sevir_ml_beta' in checkbox):
        y = dist_forecast_df['y_pred_seirv_ml_beta_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       name="SEIURV ML beta",
                       line_color='rgb(0,176,246)'),
            secondary_y=False,
        )
    if ('sarima' in checkbox):
        y = dist_forecast_df['y_pred_sarima_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       line_color='rgb(231,107,243)',
                       name="ARIMA"),
                       # line = dict(color='green')),
            secondary_y=False,
        )
    if ('ensemble'in checkbox):
        y = dist_forecast_df['y_pred_ensemble_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       line_color='rgb(230,171,2)',
                       name="Ensemble"),

            secondary_y=False,
        )

    # add upper and lower bounds
    if ('sevir_last_beta' in checkbox and show_type == 'intervals'):
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
    if ('sevir_ml_beta' in checkbox and show_type == 'intervals'):
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
    if ('sarima' in checkbox and show_type == 'intervals'):
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
    if ('ensemble' in checkbox and show_type == 'intervals'):
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

    def set_cat(row):
        if row[selected_model] > 0 and row[selected_model] < 251:
            return '0 - 250'
        if row[selected_model] > 251 and row[selected_model] < 501:
            return '251 - 500'
        if row[selected_model] > 501 and row[selected_model] < 751:
            return '501 - 750'
        if row[selected_model] > 751 and row[selected_model] < 1001:
            return '751 - 1,000'
        if row[selected_model] > 1001:
            return '1,000 and higher'

    # data_for_map_df['range'] = data_for_map_df.apply(set_cat, axis=1)

    forecast_map = px.choropleth_mapbox(
        data_for_map_df,
        locations="id",
        geojson=german_districts,
        # color='range',
        color=selected_model,
        hover_name='district_name',
        hover_data=['district_name', selected_model],
        title="Next 14-Day Incident Number",
        mapbox_style="carto-darkmatter",
        # hot blackbody thermal
        color_continuous_scale="thermal",
        # color_discrete_map={
        #     '0 - 250': '#921315',
        #     '251 - 500': '#661313',
        #     '501 - 750': '#D90183',
        #     '751 - 1,000': '#FE72C5',
        #     '1,001 and higher': '#620042'},
        # category_orders={
        #     'range': [
        #         '0 - 250',
        #         '251 - 500',
        #         '501 - 750',
        #         '751 - 1,000',
        #         '1,001 and higher'
        #     ]
        # },
        range_color=(0, 2000),
        animation_frame='date',
        center={"lat": 51.1657, "lon": 10.4515},
        zoom=5,
        # opacity=0.95,
        width=830,
        height=830,

    )

    return forecast_map


if __name__ == '__main__':
    app.run_server()
