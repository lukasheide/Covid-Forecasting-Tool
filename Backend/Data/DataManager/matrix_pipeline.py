"""
    this pipeline is dedicated to create the data matrix
    that is used to train the machine learning layer which is used to predict beta
"""
from Backend.Data.DataManager.matrix_data import create_complete_matrix_data


def run_matrix_creation_pipeline():
    create_complete_matrix_data()


if __name__ == '__main__':
    """
        run this pipeline to gathers all the required data from the DB tables, 
        fetch weather data using the meteostat library and prepares the data matrix
    """
    run_matrix_creation_pipeline()
