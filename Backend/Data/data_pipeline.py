"""
    this pipeline is dedicated to prepare all the required data to generate a forecast
"""
from Backend.Data.DataManager.opendaten_data_extractor import update_all_district_data
from Backend.Data.DataScraping.other_data_extractor import extract_all_other_data


def run_data_pipeline():
    # updates all the covid related data for all the districts
    update_all_district_data()

    # updates all the related data from remaining sources
    extract_all_other_data()


if __name__ == '__main__':
    """
        execute below method to update all the data sources
    """
    run_data_pipeline()
