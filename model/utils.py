import requests
import os
import json

def fetch_gcp_credentials_from_vault(vault_url, vault_token, secret_path):
    """
    Fetch GCP credentials from HashiCorp Vault.

    Args:
        vault_url (str): The URL of the Vault server.
        vault_token (str): The authentication token for Vault.
        secret_path (str): The path to the secret in Vault.

    Returns:
        dict: The credentials retrieved from Vault.
    """
    headers = {
        'X-Vault-Token': vault_token
    }
    response = requests.get(f"{vault_url}/v1/{secret_path}", headers=headers)

    if response.status_code == 200:
        data = response.json()
        credentials = data.get("data", {})
        if not credentials:
            raise Exception("No credentials found in the response.")
        return credentials
    else:
        raise Exception(f"Error fetching credentials from Vault: {response.text}")

def write_gcp_credentials(credentials, output_path="/tmp/gcs_credentials.json"):
    """
    Write GCP credentials to a file and set the environment variable.

    Args:
        credentials (dict): The credentials to write to file.
        output_path (str): Path to the file where credentials will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(credentials, f, indent=4)

    # Set the environment variable for Google Cloud SDK
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = output_path
    print(f"Credentials written to {output_path} and environment variable set.")

def get_env_variable(var_name):
    """
    Get an environment varia

