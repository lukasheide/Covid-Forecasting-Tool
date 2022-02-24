import json

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

from Backend.Data.DataManager.data_util import create_dates_array
from Backend.Data.DataManager.db_calls import get_district_forecast_data, get_all_latest_forecasts, get_all_table_data

# dark template
pio.templates.default = 'plotly_dark'

# loading css files
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'Assets/style.css']
#external_stylesheets = ["../Assets/style_dark.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app = dash.Dash(external_stylesheets=[dbc.themes.VAPOR])
# geojson data for the chloropath maps and set ids
german_districts = json.load(open("Frontend/AnalyticsTool/Assets/simplified_geo_data.geojson", 'r', encoding='utf-8'))
# starting value of the ids
dist_id = 1000
state_id_map = {}
for feature in german_districts["features"]:
    feature["id"] = dist_id
    state_id_map[feature["properties"]["GEN"]] = feature["id"]
    dist_id = dist_id + 1

# load forecast data from the distrct_forecast table
all_district_forecasts = get_all_latest_forecasts()

# default dist_list subset used for debuging purposes
# district_list = ['Münster', 'Potsdam', 'Segeberg', 'Rosenheim, Kreis', 'Hochtaunus', 'Dortmund', 'Essen', 'Bielefeld',
#                  'Warendorf']

# load the district list from district_data table
district_data = get_all_table_data(table_name='district_list')
district_list = district_data['district'].tolist()
district_list.sort()

# create the district list of latest 14 days of predictions in the prediction table
dates_list = all_district_forecasts['date'].unique()[-14:]

########### creating all three forcast maps #############
# resolution and zoom
map_width = 700
map_height = 700
map_zoom = 4.5

### create common data ###
data_for_map_df = all_district_forecasts[all_district_forecasts['cases'].isna()]
data_for_map_df["id"] = data_for_map_df["district_name"].apply(lambda x: state_id_map[x])

### SEIRV lastweek-Beta Map ###
data_for_map_df['y_pred_seirv_last_beta_mean'] = data_for_map_df['y_pred_seirv_last_beta_mean'] \
    .apply(pd.to_numeric).round(decimals=2)
seirv_lastweek_beta_forecast = px.choropleth_mapbox(
    data_for_map_df,
    locations="id",
    geojson=german_districts,
    # color='range',
    color='y_pred_seirv_last_beta_mean',
    hover_name='district_name',
    hover_data=['district_name', 'y_pred_seirv_last_beta_mean'],
    title="7 Day Incidence Forecast",
    mapbox_style="carto-darkmatter",
    color_continuous_scale="Redor",
    labels={'y_pred_seirv_last_beta_mean': ''},
    range_color=(0, 2500),
    animation_frame='date',
    center={"lat": 51.1657, "lon": 10.4515},
    zoom=map_zoom,
    opacity=0.8,
    width=map_width,
    height=map_height,
)

### SEIRV ML-Beta Map ###
data_for_map_df['y_pred_seirv_ml_beta_mean'] = data_for_map_df['y_pred_seirv_ml_beta_mean'] \
    .apply(pd.to_numeric).round(decimals=2)
seirv_ml_forecast = px.choropleth_mapbox(
    data_for_map_df,
    locations="id",
    geojson=german_districts,
    # color='range',
    color='y_pred_seirv_ml_beta_mean',
    hover_name='district_name',
    hover_data=['district_name', 'y_pred_seirv_ml_beta_mean'],
    title="7 Day Incidence Forecast",
    mapbox_style="carto-darkmatter",
    color_continuous_scale="Redor",
    labels={'y_pred_seirv_ml_beta_mean': ''},
    range_color=(0, 2500),
    animation_frame='date',
    center={"lat": 51.1657, "lon": 10.4515},
    zoom=map_zoom,
    opacity=0.8,
    width=map_width,
    height=map_height,
)

### ARIMA Map ###
data_for_map_df['y_pred_sarima_mean'] = data_for_map_df['y_pred_sarima_mean'] \
    .apply(pd.to_numeric).round(decimals=2)
arima_forecast = px.choropleth_mapbox(
    data_for_map_df,
    locations="id",
    geojson=german_districts,
    # color='range',
    color='y_pred_sarima_mean',
    hover_name='district_name',
    hover_data=['district_name', 'y_pred_sarima_mean'],
    title="7 Day Incidence Forecast",
    mapbox_style="carto-darkmatter",
    color_continuous_scale="Redor",
    labels={'y_pred_sarima_mean': ''},
    range_color=(0, 2500),
    animation_frame='date',
    center={"lat": 51.1657, "lon": 10.4515},
    zoom=map_zoom,
    opacity=0.8,
    width=map_width,
    height=map_height,
)


### Ensemble Map ###
data_for_map_df['y_pred_ensemble_mean'] = data_for_map_df['y_pred_ensemble_mean'] \
    .apply(pd.to_numeric).round(decimals=2)
ensemble_forecast = px.choropleth_mapbox(
    data_for_map_df,
    locations="id",
    geojson=german_districts,
    # color='range',
    color='y_pred_ensemble_mean',
    hover_name='district_name',
    hover_data=['district_name', 'y_pred_ensemble_mean'],
    title="7 Day Incidence Forecast",
    mapbox_style="carto-darkmatter",
    color_continuous_scale="Redor",
    labels={'y_pred_ensemble_mean': ''},
    range_color=(0, 2500),
    animation_frame='date',
    center={"lat": 51.1657, "lon": 10.4515},
    zoom=map_zoom,
    opacity=0.8,
    width=map_width,
    height=map_height,
)


########### app layout is defined here ##############

app.layout = html.Div([
    # html.Hr(style={'backgroundColor':'#111111'},),
    html.H2('Regional COVID-19 Forecasting Tool',
            style={'backgroundColor': '#111111', 'color': 'white', 'text-align': 'center', 'padding-top': '3px', 'padding-bottom': '20px', 'padding-top': '20px'}),
    html.Div(
        className="row",
        style={'backgroundColor': '#111111', 'color': 'white',},
        children=[
            html.Div(
                className="six columns",
                children=[
                    html.Div([
                            html.Div(
                                children=[
                                            # html.H6('District Forecast ', style={'backgroundColor':'#111111', 'color':'white'}),
                                            dcc.Dropdown(
                                                    id='district-dropdown',
                                                    options=[{'label': k, 'value': k} for k in district_list],
                                                    multi=False,
                                                    value='Münster',
                                                    style={'backgroundColor':'#111111', 'color':'#ffffff'},
                                                ),
                                            html.Div([

                                                html.Div([
                                                    html.H6('Forecasting Models', style={'backgroundColor':'#111111', 'color':'white'}),
                                                    dcc.Checklist(
                                                        id='model-check',
                                                        options=[
                                                            {'label': 'SEIURV Last Beta', 'value': 'sevir_last_beta'},
                                                            {'label': 'SEIURV ML Beta', 'value': 'sevir_ml_beta'},
                                                            {'label': 'ARIMA', 'value': 'sarima'},
                                                            {'label': 'Ensemble', 'value': 'ensemble'},
                                                        ],
                                                        value='sevir_ml_beta',
                                                    )], className="six columns",
                                                        style={'verticalAlign': 'top'}),

                                                html.Div([
                                                    html.H6('Prediction Intervals', style={'backgroundColor':'#111111', 'color':'white'}),
                                                    dcc.Checklist(
                                                        id='show-interval-check',
                                                        options=[
                                                            {'label': 'Show Intervals', 'value': 'intervals'},
                                                        ],
                                                        value='intervals',
                                                    )], className='six columns',
                                                        style={'verticalAlign': 'top'}),
                                                ], className='row', style={'padding-left': '30px', 'padding-top': '30px'}),
                                        ]),
                            dcc.Graph(
                                id='dist-forecast-graph',
                                figure={
                                    'layout': {
                                        # 'height': 800,
                                        # 'width': 900,
                                        'margin': {'l': 10, 'b': 10, 't': 10, 'r': 10},
                                        'paper_bgcolor': '#7FDBFF',
                                        'plot_bgcolor': '#7FDBFF',
                                    }
                                }
                            ),

                        ])],
                style={'padding-right': '30px', 'padding-left': '30px', 'padding-top': '10px', 'padding-bottom': '10px'}
            ),
            html.Div(
                className="six columns",
                children=[
                    html.Div(
                        children=[
                            # html.H6('may be add a title here', style={'backgroundColor':'#111111', 'color':'white'}),
                            dcc.Dropdown(
                                id='map-forecast-model',

                                options=[{'label': 'SEIURV Last Beta', 'value': 'y_pred_seirv_last_beta_mean'},
                                         {'label': 'SEIURV ML beta', 'value': 'y_pred_seirv_ml_beta_mean'},
                                         {'label': 'ARIMA', 'value': 'y_pred_sarima_mean'},
                                         {'label': 'Ensemble', 'value': 'y_pred_ensemble_mean'}],
                                multi=False,
                                value='y_pred_ensemble_mean',
                                style={'backgroundColor':'#111111','font-color':'white'},
                            ),
                            dcc.Graph(id='forecast-chloropath', figure={}),
                        ]
                    )
                ],
                style={'padding-right': '30px', 'padding-left': '30px', 'padding-top': '10px', 'padding-bottom': '10px', 'justify-content': 'center'}
            ),
        ]
    )
], style={'backgroundColor': '#111111'})


@app.callback(
    Output(component_id='dist-forecast-graph', component_property='figure'),
    Output(component_id='district-dropdown', component_property='value'),
    Output(component_id='forecast-chloropath', component_property='clickData'),
    Input(component_id='district-dropdown', component_property='value'),
    # Input(component_id='model-radio', component_property='value'),
    Input(component_id='model-check', component_property='value'),
    Input(component_id='show-interval-check', component_property='value'),
    Input(component_id='forecast-chloropath', component_property='clickData'),
)
def get_dist_forecast_plot(district, checkbox, show_interval, click_data):
    line_width = 4

    # if the choropleth map is loaded and the callback on user click on a map-district update accordingly
    if click_data is not None and district != click_data['points'][0]['hovertext']:
        district = click_data['points'][0]['hovertext']

    # retrieve forecast data for the selected district
    dist_forecast_df = get_district_forecast_data(district)
    training_len = len(dist_forecast_df['cases'].dropna())
    # specify the length of the training period to be shown in the graph (range 1-28)
    shown = 21
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates_array = create_dates_array(start_date_str=dist_forecast_df['date'][0],
                                     num_days=len(dist_forecast_df['cases'].dropna()) + 14)

    # Add traces
    y_common_train = dist_forecast_df['cases'][training_len-shown:training_len].dropna()
    max_key = max(y_common_train.keys())
    max_train = y_common_train[max_key]
    fig.add_trace(
        go.Scatter(x=dates_array[training_len-shown:training_len],
                   y=y_common_train,
                   mode='lines+markers',
                   name='Training',
                   line=dict(width=line_width)),
        secondary_y=False,
    )
    fig.update_traces(marker={'size': 12})
    #last_train_list = y_common_train[-1]

    # add upper and lower bounds
    if ('sevir_last_beta' in checkbox and 'intervals' in show_interval):
        y_upper = dist_forecast_df['y_pred_seirv_last_beta_upper'].dropna().tolist()
        y_lower = dist_forecast_df['y_pred_seirv_last_beta_lower'].dropna().tolist()
        x = list(dates_array[-15:])
        y_lower = y_lower[::-1]
        y_lower = np.append(y_lower, max_train)
        y_upper = np.append(max_train, y_upper)
        y_upper = list(y_upper)
        y_lower = list(y_lower)
        x_rev = x[::-1]
        x = x + x_rev
        y_interval = y_upper + y_lower
        fig.add_trace(go.Scatter(x=x,
                                 y=y_interval,
                                 fill='toself',
                                 fillcolor='rgba(0,100,80,0.2)',
                                 # line_color='rgba(255,255,255,0)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name="SEIURV Last Beta",
                                 showlegend=False))
    if ('sevir_ml_beta' in checkbox and 'intervals' in show_interval):
        y_upper = dist_forecast_df['y_pred_seirv_ml_beta_upper'].dropna().tolist()
        y_lower = dist_forecast_df['y_pred_seirv_ml_beta_lower'].dropna().tolist()
        x = list(dates_array[-15:])
        y_lower = y_lower[::-1]
        y_lower = np.append(y_lower, max_train)
        y_upper = np.append(max_train, y_upper)
        y_upper = list(y_upper)
        y_lower = list(y_lower)
        x_rev = x[::-1]
        x = x + x_rev
        y_interval = y_upper + y_lower
        fig.add_trace(go.Scatter(x=x,
                                 y=y_interval,
                                 fill='toself',
                                 fillcolor='rgba(0,176,246,0.2)',
                                 # line_color='rgba(255,255,255,0)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name="SEIURV ML beta",
                                 showlegend=False))
    if ('sarima' in checkbox and 'intervals' in show_interval):
        y_upper = dist_forecast_df['y_pred_sarima_upper'].dropna().tolist()
        y_lower = dist_forecast_df['y_pred_sarima_lower'].dropna().tolist()
        x = list(dates_array[-15:])
        y_lower = y_lower[::-1]
        y_lower = np.append(y_lower, max_train)
        y_upper = np.append(max_train, y_upper)
        y_upper = list(y_upper)
        y_lower = list(y_lower)
        x_rev = x[::-1]
        x = x + x_rev
        y_interval = y_upper + y_lower
        fig.add_trace(go.Scatter(x=x,
                                 y=y_interval,
                                 fill='toself',
                                 fillcolor='rgba(231,107,243,0.2)',
                                 # line_color='rgba(255,255,255,0)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name='ARIMA',
                                 showlegend=False))
    if ('ensemble' in checkbox and 'intervals' in show_interval):
        y_upper = dist_forecast_df['y_pred_ensemble_upper'].dropna().tolist()
        y_lower = dist_forecast_df['y_pred_ensemble_lower'].dropna().tolist()
        x = list(dates_array[-15:])
        y_lower = y_lower[::-1]
        y_lower = np.append(y_lower, max_train)
        y_upper = np.append(max_train, y_upper)
        y_upper = list(y_upper)
        y_lower = list(y_lower)
        x_rev = x[::-1]
        x = x + x_rev
        y_interval = y_upper + y_lower
        fig.add_trace(go.Scatter(x=x,
                                 y=y_interval,
                                 fill='toself',
                                 fillcolor='rgba(230,171,2,0.2)',
                                 # line_color='rgba(255,255,255,0)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name='Ensemble',
                                 showlegend=False))

    if ('sevir_last_beta' in checkbox):
        y = dist_forecast_df['y_pred_seirv_last_beta_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       name="SEIURV Last Beta",
                       # line_color='rgb(0,100,80)',
                       line=dict(color='rgb(0,100,80)', width=line_width),
                       mode='lines'),
            secondary_y=False,
        )
    if ('sevir_ml_beta' in checkbox):
        y = dist_forecast_df['y_pred_seirv_ml_beta_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       name="SEIURV ML beta",
                       # line_color='rgb(0,176,246)',
                       line=dict(color='rgb(0,176,246)', width=line_width),
                       mode='lines'),
            secondary_y=False,
        )
    if ('sarima' in checkbox):
        y = dist_forecast_df['y_pred_sarima_mean'].dropna()
        y_fixed = pd.concat([pd.Series(y_common_train.iloc[-1]), y])
        fig.add_trace(
            go.Scatter(x=dates_array[-15:], y=y_fixed,
                       # line_color='rgb(231,107,243)',
                       line=dict(color='rgb(231,107,243)', width=line_width),
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
                       # line_color='rgb(230,171,2)',
                       line=dict(color='rgb(230,171,2)', width=line_width),
                       name="Ensemble",
                       mode='lines'),

            secondary_y=False,
        )

    # Add figure title
    fig.update_layout(
        title_text="Forecast for " + district,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Time")

    #set y-axis title
    non_numeric_cols = ['prediction_id', 'pipeline_id', 'district_name', 'date']
    sel_columns = [c for c in list(dist_forecast_df.columns) if c not in non_numeric_cols]
    temp_df = dist_forecast_df[sel_columns].fillna(value=0).apply(pd.to_numeric)
    max_train_data_points = np.max(temp_df.max())

    max_y_axis_value = round(max_train_data_points*1.2/100)*100

    fig.update_yaxes (title_text='7 Day Incidence', range=[0, max_y_axis_value])

    return fig, district, None

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

    # can be used to color by range (currently not in use)
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

    forecast_map = px.choropleth_mapbox()

    # preloaded model is assigned based on the selected model
    if selected_model == 'y_pred_seirv_last_beta_mean':
        forecast_map = seirv_lastweek_beta_forecast
    if selected_model == 'y_pred_seirv_ml_beta_mean':
        forecast_map = seirv_ml_forecast
    if selected_model == 'y_pred_sarima_mean':
        forecast_map = arima_forecast
    if selected_model == 'y_pred_ensemble_mean':
        forecast_map = ensemble_forecast

    return forecast_map


if __name__ == '__main__':
    app.run_server()
