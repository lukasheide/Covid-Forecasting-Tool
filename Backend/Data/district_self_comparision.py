import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px

from Backend.Data.db_functions import get_table_data

app = dash.Dash(__name__)

district_list = get_table_data('district_list', 0, 0, "district", False)

all_district_data = pd.DataFrame()
attr_list = ['daily_infec',
             'seven_day_infec',
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

app.layout = html.Div([
    dcc.Dropdown(
        id='district-dropdown',
        options=[{'label': k, 'value': k} for k in district_list.district.unique()],
        multi=False,
        value='Münster'
    ),

    html.Hr(),

    dcc.Dropdown(id='attr-list-dropdown',
                 multi=True,
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


def create_district_data(district):
    attributes = []

    dis_data = get_table_data(district, 0, 0, ['date',
                                               'daily_infec',
                                               'seven_day_infec',
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
    for index, row in dis_data.iterrows():
        attributes.append([row['date'], row['daily_infec'], 'daily_infec'])
        attributes.append([row['date'], row['seven_day_infec'], 'seven_day_infec'])
        attributes.append([row['date'], row['cum_infec'], 'cum_infec'])
        attributes.append([row['date'], row['daily_deaths'], 'daily_deaths'])
        attributes.append([row['date'], row['cum_deaths'], 'cum_deaths'])
        attributes.append([row['date'], row['daily_rec'], 'daily_rec'])
        attributes.append([row['date'], row['cum_rec'], 'cum_rec'])
        attributes.append([row['date'], row['adjusted_active_cases'], 'adjusted_active_cases'])
        attributes.append([row['date'], row['daily_incidents_rate'], 'daily_incidents_rate'])
        attributes.append([row['date'], row['daily_vacc'], 'daily_vacc'])
        attributes.append([row['date'], row['cum_vacc'], 'cum_vacc'])
        attributes.append([row['date'], row['vacc_percentage'], 'vacc_percentage'])

    global all_district_data
    all_district_data = pd.DataFrame(attributes[:], columns=['date', 'value', 'attribute'])
    all_district_data['value'] = pd.to_numeric(all_district_data['value'])


create_district_data('Münster')

@app.callback(
    Output(component_id='attr-graph', component_property='figure'),
    Input(component_id='attr-list-dropdown', component_property='value'),
)
def update_graphs(attr_chosen):
    if type(attr_chosen) is str:
        attr_chosen = [attr_chosen]

    if type(attr_chosen) is None:
        attr_chosen = []

    dff = all_district_data[all_district_data.attribute.isin(attr_chosen)]
    fig = px.line(data_frame=dff, x='date', y='value', color='attribute',
                  custom_data=['date', 'value', 'attribute'])
    fig.update_traces(mode='lines+markers')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
