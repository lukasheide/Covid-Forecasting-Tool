import urllib.request

from datetime import datetime


def update_ecdc_variant_data():
    full_file_name = '../Assets/Data/Scraped/ecdc/'+datetime.today().strftime('%d%m%y')+'.csv'
    urllib.request.urlretrieve("https://opendata.ecdc.europa.eu/covid19/virusvariant/csv/data.csv", full_file_name)
    print('ecdc data updated!')


def update_destatis_mobility_data():
    full_file_name = '../Assets/Data/Scraped/destatis/'+datetime.today().strftime('%d%m%y')+'.csv'
    urllib.request.urlretrieve("https://service.destatis.de/DE/maps/2020/data/map_reg.csv", full_file_name)
    print('destatis data updated!')


def update_oxcgrt_policy_data():
    full_file_name = '../Assets/Data/Scraped/oxcgrt/'+datetime.today().strftime('%d%m%y')+'.csv'
    urllib.request.urlretrieve("https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries"
                               "/containment_health_index.csv", full_file_name)
    print('oxcgrt data updated!')


if __name__ == '__main__':
    # update_ecdc_variant_data()
    # update_destatis_mobility_data()
    update_oxcgrt_policy_data()
