import os
import urllib.request

from datetime import datetime

"""
    downloading the latest updates from ECDC, DESTATIS and OXCGRT
"""


def download_ecdc_variant_data():
    full_file_name = 'Assets/Data/Scraped/ecdc/' + datetime.today().strftime('%d%m%y') + '.csv'
    os.makedirs(os.path.dirname(full_file_name), exist_ok=True)

    urllib.request.urlretrieve("https://opendata.ecdc.europa.eu/covid19/virusvariant/csv/data.csv", full_file_name)
    print('latest ecdc data downloaded!')


def download_destatis_mobility_data():
    full_file_name = 'Assets/Data/Scraped/destatis/' + datetime.today().strftime('%d%m%y') + '.csv'
    os.makedirs(os.path.dirname(full_file_name), exist_ok=True)

    urllib.request.urlretrieve("https://service.destatis.de/DE/maps/2020/data/map_reg.csv", full_file_name)
    print('latest destatis data downloaded!')


def download_oxcgrt_policy_data():
    full_file_name = 'Assets/Data/Scraped/oxcgrt/' + datetime.today().strftime('%d%m%y') + '.csv'
    os.makedirs(os.path.dirname(full_file_name), exist_ok=True)

    urllib.request.urlretrieve("https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries"
                               "/containment_health_index.csv", full_file_name)
    print('latest oxcgrt data downloaded!')
