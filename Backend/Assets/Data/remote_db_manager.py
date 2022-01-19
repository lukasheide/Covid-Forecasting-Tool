from paramiko import Transport, SFTPClient


def download_db_file():
    host = "D-3120S33.uni-muenster.de"
    port = 2222
    username = "covcastadmin"
    password = "7vCrSwcnFJXir"

    transport = Transport((host, port))
    transport.connect(username=username, password=password)
    sftp = SFTPClient.from_transport(transport)

    remote_path = "Assets/Databasefile/opendaten_backup.db"
    local_path = "../Assets/Data/opendaten_backup.db"
    # sftp.put(localpath, path)

    sftp.get(remotepath=remote_path, localpath=local_path)

    sftp.close()
    transport.close()


def upload_db_file():
    host = "D-3120S33.uni-muenster.de"
    port = 2222
    username = "covcastadmin"
    password = "7vCrSwcnFJXir"

    transport = Transport((host, port))
    transport.connect(username=username, password=password)
    sftp = SFTPClient.from_transport(transport)

    remote_path = "Assets/Databasefile/opendaten_backup.db"
    local_path = "../Assets/Data/opendaten_backup.db"

    sftp.put(remotepath=remote_path, localpath=local_path)

    sftp.close()
    transport.close()


if __name__ == '__main__':
    # upload_db_file()
    download_db_file()
