from paramiko import Transport, SFTPClient

from Backend.Data.DataManager.data_util import print_progress
from Backend.Data.DataManager.properties import Server
import shutil
import os
from pathlib import Path

from datetime import datetime

"""
These methods are used to upload and download files to the server. This includes our local database files
as well as csv files, for example, regarding the prediction intervals.
"""


def download_db_file():
    """
        downloading sqlite .db file
    """
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # check file if exist in the path and create if not
    os.makedirs(os.path.dirname("Assets/Data/opendaten.db"), exist_ok=True)

    # create a copy of prediction_intervals.csv if exists
    if os.path.isfile('Assets/Data/opendaten.db'):
        original = 'Assets/Data/opendaten.db'
        target = 'Assets/Data/opendaten_backup.db'
        shutil.copy2(original, target)

    remote_path = "Assets/Databasefile/opendaten.db"
    local_path = "Assets/Data/opendaten.db"

    print("downloading the database file from the server:")
    sftp.get(remotepath=remote_path, localpath=local_path, callback=print_progress)
    print("download success!")

    sftp.close()
    transport.close()


def upload_db_file():
    """
        uploading sqlite .db file
    """
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # check file if exist in the path and create if not
    os.makedirs(os.path.dirname("Assets/Data/opendaten.db"), exist_ok=True)

    # check if the file exists
    if os.path.isfile('Assets/Data/opendaten.db'):
        remote_path = "Assets/Databasefile/opendaten.db"
        local_path = "Assets/Data/opendaten.db"

        # first rename the existing remote file as a backup before uploading new local changes
        timestamp = datetime.now()
        remote_path_backup = f"Assets/Databasefile/opendaten_backup_{timestamp}.db"
        sftp.rename(oldpath=remote_path, newpath=remote_path_backup)

        # uploading new local changes
        print("uploading the prediction_intervals.csv file to the server:")
        sftp.put(remotepath=remote_path, localpath=local_path, callback=print_progress)
        print("upload success!")

    else:
        print("no file at 'Assets/Data/opendaten.db'")

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

    # create a copy of prediction_intervals.csv if exists
    if os.path.isfile('Assets/Forecasts/PredictionIntervals/prediction_intervals.csv'):
        original = 'Assets/Forecasts/PredictionIntervals/prediction_intervals.csv'
        target = 'Assets/Forecasts/PredictionIntervals/prediction_intervals_backup.csv'
        shutil.copy2(original, target)

    remote_path = "Assets/DataUtils/prediction_intervals.csv"
    local_path = "Assets/Forecasts/PredictionIntervals/prediction_intervals.csv"

    print("downloading the prediction_intervals file from the server:")
    sftp.get(remotepath=remote_path, localpath=local_path, callback=print_progress)
    print("download success!")

    sftp.close()
    transport.close()


def upload_pred_intervals_file():
    """
        uploading prediction_intervals.csv file
    """
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # check file if exist in the path and create if not
    os.makedirs(os.path.dirname("Assets/Forecasts/PredictionIntervals/prediction_intervals.csv"), exist_ok=True)

    # check if the file exists
    if os.path.isfile('Assets/Forecasts/PredictionIntervals/prediction_intervals.csv'):
        remote_path = "Assets/DataUtils/prediction_intervals.csv"
        local_path = "Assets/Forecasts/PredictionIntervals/prediction_intervals.csv"

        # first rename the existing remote file as a backup before uploading new local changes
        timestamp = datetime.now()
        remote_path_backup = f"Assets/DataUtils/prediction_intervals_backup_{timestamp}.csv"
        sftp.rename(oldpath=remote_path, newpath=remote_path_backup)

        # uploading new local changes
        print("uploading the prediction_intervals.csv file to the server:")
        sftp.put(remotepath=remote_path, localpath=local_path, callback=print_progress)
        print("upload success!")

    else:
        print("no file at 'Assets/Data/Scraped/destatis/prediction_intervals.csv'")

    sftp.close()
    transport.close()


def download_destatis_base_file():
    """
        downloading destatis_base.csv file
    """
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # check file path exists and create if not
    os.makedirs(os.path.dirname("Assets/Data/Scraped/destatis/destatis_base.csv"), exist_ok=True)

    # create a copy of destatis_base.csv if exists
    if os.path.isfile('Assets/Data/Scraped/destatis/destatis_base.csv'):
        original = 'Assets/Data/Scraped/destatis/destatis_base.csv'
        target = 'Assets/Data/Scraped/destatis/destatis_base_backup.csv'
        shutil.copy2(original, target)

    remote_path = "Assets/DataUtils/destatis_base.csv"
    local_path = "Assets/Data/Scraped/destatis/destatis_base.csv"

    print("downloading the destatis_base.csv file from the server:")
    sftp.get(remotepath=remote_path, localpath=local_path, callback=print_progress)
    print("download success!")

    sftp.close()
    transport.close()


def upload_destatis_base_file():
    """
        uploading destatis_base.csv file
    """
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # check file path exists and create if not
    os.makedirs(os.path.dirname("Assets/Data/Scraped/destatis/destatis_base.csv"), exist_ok=True)

    # check if the file exists
    if os.path.isfile('Assets/Data/Scraped/destatis/destatis_base.csv'):
        remote_path = "Assets/DataUtils/destatis_base.csv"
        local_path = "Assets/Data/Scraped/destatis/destatis_base.csv"

        # first rename the existing remote file as a backup before uploading new local changes
        timestamp = datetime.now()
        remote_path_backup = f"Assets/DataUtils/destatis_base_backup_{timestamp}.csv"
        sftp.rename(oldpath=remote_path, newpath=remote_path_backup)

        # uploading new local changes
        print("uploading the destatis_base.csv file to the server:")
        sftp.put(remotepath=remote_path, localpath=local_path, callback=print_progress)
        print("upload success!")

    else:
        print("no file at 'Assets/Data/Scraped/destatis/destatis_base.csv'")

    sftp.close()
    transport.close()


if __name__ == '__main__':

    """
    These methods are used to upload and download files to the server. This includes our local database files
    as well as csv files, for example, regarding the prediction intervals.
    """

    task = 'upload_destatis_base'

    if task == 'upload':
        upload_db_file()
    elif task == 'download':
        download_db_file()
    elif task == 'upload_pred_intervals':
        upload_pred_intervals_file()
    elif task == 'download_pred_intervals':
        download_pred_intervals_file()
    elif task == 'download_destatis_base':
        download_destatis_base_file()
    elif task == 'upload_destatis_base':
        upload_destatis_base_file()
