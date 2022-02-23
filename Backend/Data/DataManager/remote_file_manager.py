from paramiko import Transport, SFTPClient

from Backend.Data.DataManager.data_util import print_progress
from Backend.Data.DataManager.properties import Server
import shutil
import os
from pathlib import Path

from datetime import datetime

"""
    these methods are used to handle the files which are needed be keep up to date but cannot be used with git
"""


def download_db_file():
    """
        downloading sqlite .db file
    """
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # create a copy of the existing database file
    original = 'Assets/Data/opendaten.db'
    target = 'Assets/Data/opendaten_backup.db'
    shutil.copy2(original, target)

    remote_path = "Assets/Databasefile/opendaten.db"
    local_path = "Assets/Data/opendaten.db"

    print("downloading the database file from the server:")
    sftp.get(remotepath=remote_path, localpath=local_path, callback=print_progress)

    sftp.close()
    transport.close()


def upload_db_file():
    """
        uploading sqlite .db file
    """
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # create a local copy of the existing database file
    original = 'Assets/Data/opendaten.db'
    target = 'Assets/Data/opendaten_backup.db'
    shutil.copy2(original, target)

    remote_path = "Assets/Databasefile/opendaten.db"
    local_path = "Assets/Data/opendaten.db"

    print("uploading the database file to the server:")
    sftp.put(remotepath=remote_path, localpath=local_path, callback=print_progress)

    timestamp = datetime.now()
    remote_path_backup = f"Assets/Databasefile/opendaten_backup_{timestamp}.db"
    sftp.put(remotepath=remote_path_backup, localpath=local_path)

    # save backup

    sftp.close()
    transport.close()


def download_pred_intervals_file():
    """
        downloading prediction_intervals.csv file
    """
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # check file if exist in the path and create if not
    os.makedirs(os.path.dirname("Assets/Forecasts/PredictionIntervals/prediction_intervals.csv"), exist_ok=True)

    # create a copy of the existing database file
    original = 'Assets/Forecasts/PredictionIntervals/prediction_intervals.csv'
    target = 'Assets/Forecasts/PredictionIntervals/prediction_intervals_backup.csv'
    shutil.copy2(original, target)

    remote_path = "Assets/DataUtils/prediction_intervals.csv"
    local_path = "Assets/Forecasts/PredictionIntervals/prediction_intervals.csv"

    print("downloading the prediction_intervals file from the server:")
    sftp.get(remotepath=remote_path, localpath=local_path, callback=print_progress)

    sftp.close()
    transport.close()


def upload_pred_intervals_file():
    """
        downloading prediction_intervals.csv file
    """
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # check file if exist in the path and create if not
    os.makedirs(os.path.dirname("Assets/Forecasts/PredictionIntervals/prediction_intervals.csv"), exist_ok=True)

    # create a local copy of the existing database file
    original = 'Assets/Forecasts/PredictionIntervals/prediction_intervals.csv'
    target = 'Assets/Forecasts/PredictionIntervals/prediction_intervals_backup.csv'
    shutil.copy2(original, target)

    remote_path = "Assets/DataUtils/prediction_intervals.csv"
    local_path = "Assets/Forecasts/PredictionIntervals/prediction_intervals.csv"

    print("uploading the prediction_intervals file to the server:")
    sftp.put(remotepath=remote_path, localpath=local_path, callback=print_progress)

    timestamp = datetime.now()
    remote_path_backup = f"Assets/DataUtils/prediction_intervals_{timestamp}.csv"
    sftp.put(remotepath=remote_path_backup, localpath=local_path)

    # save backup

    sftp.close()
    transport.close()


if __name__ == '__main__':

    task = 'download_pred_intervals'

    if task == 'upload':
        upload_db_file()
    elif task == 'download':
        download_db_file()
    elif task == 'upload_pred_intervals':
        upload_pred_intervals_file()
    elif task == 'download_pred_intervals':
        download_pred_intervals_file()
