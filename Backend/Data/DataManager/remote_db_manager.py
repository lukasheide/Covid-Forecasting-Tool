from paramiko import Transport, SFTPClient
from Backend.Data.DataManager.properties import Server
import shutil

from datetime import datetime


def download_db_file():
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # create a copy of the existing database file
    original = '../Assets/Data/opendaten.db'
    target = '../Assets/Data/opendaten_backup.db'
    shutil.copy2(original, target)

    remote_path = "Assets/Databasefile/opendaten.db"
    local_path = "../Assets/Data/opendaten.db"

    sftp.get(remotepath=remote_path, localpath=local_path)

    sftp.close()
    transport.close()


def upload_db_file():
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    # create a copy of the existing database file
    original = '../Assets/Data/opendaten.db'
    target = '../Assets/Data/opendaten_backup.db'
    shutil.copy2(original, target)

    remote_path = "Assets/Databasefile/opendaten.db"
    local_path = "../Assets/Data/opendaten.db"

    sftp.put(remotepath=remote_path, localpath=local_path)

    timestamp = datetime.now()
    remote_path_backup = f"Assets/Databasefile/opendaten_backup_{timestamp}.db"

    # save backup

    sftp.close()
    transport.close()


if __name__ == '__main__':

    task = 'upload'

    if task == 'upload':
        upload_db_file()
    elif task == 'download':
        download_db_file()
