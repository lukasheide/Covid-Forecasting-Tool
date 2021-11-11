import requests as req
import pandas as pd
from datetime import datetime
import time

rki_res = req.get(
    'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/rki_history_hubv/FeatureServer/0/query?where=%20(AdmUnitId%20%3D%205515%20OR%20AdmUnitId%20%3D%205516)%20&outFields=*&outSR=4326&f=json')
data = rki_res.json()
data['features']

records = []

for rec in data['features']:
    records.append(rec['attributes'])

dataRKI = pd.DataFrame(records)

dataRKI = dataRKI.sort_values(by='Datum')

datetime.today().strftime('%d%m%y')

dataRKI.to_csv(datetime.today().strftime('%d%m%y')+'.csv')