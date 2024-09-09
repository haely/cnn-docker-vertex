import requests
import os
import json

def fetch_gcp_credentials_from_vault(vault_url, vault_token, secret_path):
    headers = {
        'X-Vault-Token': vault_token
    }
    response = requests.get(f"{vault_url}/v1/{secret_path}", headers=headers)

    if response.status_code == 200:
        credentials = response.json()["data"]
        return credentials
    else:
        raise Exception(f"Error fetching credentials from Vault: {response.text}")

def write_gcp_credentials(credentials, output_path="/tmp/gcs_credentials.json"):
    with open(output_path, 'w') as f:
        json.dump(credentials, f)

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = output_path

