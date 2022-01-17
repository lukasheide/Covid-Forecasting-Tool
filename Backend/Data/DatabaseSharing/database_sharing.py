from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.file import File
from office365.graph_client import GraphClient
import os
import tempfile

from office365.sharepoint.client_context import ClientContext

from confidentials import USERNAME, PASSWORD


def upload_db_file_to_teams():
    pass

def get_db_file_from_teams():
    client = GraphClient(acquire_token_func)



if __name__ == '__main__':
