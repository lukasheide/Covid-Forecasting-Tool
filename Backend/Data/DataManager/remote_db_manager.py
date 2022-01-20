from paramiko import Transport, SFTPClient

from Backend.Data.DataManager.data_util import print_progress
from Backend.Data.DataManager.properties import Server
import shutil


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

    print("downloading the database file from the server:")
    sftp.get(remotepath=remote_path, localpath=local_path, callback=print_progress)

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

    print("uploading the database file to the server:")
    sftp.put(remotepath=remote_path, localpath=local_path, callback=print_progress)

    sftp.close()
    transport.close()


if __name__ == '__main__':
    # upload_db_file()
    download_db_file()
