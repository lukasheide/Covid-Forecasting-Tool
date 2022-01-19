from paramiko import Transport, SFTPClient
from properties import Server


def download_db_file():
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    remote_path = "Assets/Databasefile/opendaten_backup.db"
    local_path = "../Assets/Data/opendaten_backup.db"
    # sftp.put(localpath, path)

    sftp.get(remotepath=remote_path, localpath=local_path)

    sftp.close()
    transport.close()


def upload_db_file():
    transport = Transport((Server.host, Server.port))
    transport.connect(username=Server.username, password=Server.password)
    sftp = SFTPClient.from_transport(transport)

    remote_path = "Assets/Databasefile/opendaten_backup.db"
    local_path = "../Assets/Data/opendaten_backup.db"

    sftp.put(remotepath=remote_path, localpath=local_path)

    sftp.close()
    transport.close()


if __name__ == '__main__':
    # upload_db_file()
    download_db_file()
