# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily
import plotly.express as px
import plotly.graph_objects as go

# Set time period
start = datetime(2020, 3, 1)
end = datetime.today()

# Create Point for Muenster, DE
district = Point(51.9625101, 7.6251879)

# Get daily data for 2018
data = Daily(district, start, end)
data = data.fetch()
data['tsun'] = data['tsun'].apply(lambda x: float(x) / 60)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['tavg'].rolling(15).mean(), name='Avg. Temp',
                         mode='lines'))
# fig.add_trace(go.Scatter(x=data.index, y=data['wspd'].rolling(15).mean(), name='Avg. Wind Speed',
#                          mode='lines'))
fig.add_trace(go.Scatter(x=data.index, y=data['tsun'].rolling(15).mean(), name='Avg. Hours of Sun',
                         mode='lines'))
fig.show()

# Plot line chart including average, minimum and maximum temperature
# data.plot(y=['tavg', 'tmin', 'tmax'])
# plt.show()
