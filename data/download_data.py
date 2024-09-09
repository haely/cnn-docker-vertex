import os
from google.cloud import storage

def download_data(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")

if __name__ == "__main__":
    bucket_name = "your-bucket-name"
    source_blob_name = "path/to/your/data"
    destination_file_name = "local_path/to/store/data"
    download_data(bucket_name, source_blob_name, destination_file_name)
# haely todo - update the paths
