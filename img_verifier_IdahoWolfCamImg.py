import os
from google.cloud import storage

# Setup
bucket_name = "public-datasets-lila"
manifest_path = "/home/greatgilbertsoco/WolfDetect/data/wolf_manifest.txt"
download_dir = "/home/greatgilbertsoco/WolfDetect/data/wolf_images"

client = storage.Client.create_anonymous_client()
bucket = client.bucket(bucket_name)

with open(manifest_path, "r") as f:
    lines = f.readlines()

print(f"Starting download of {len(lines)} images...")

for line in lines:
    blob_path = line.strip().replace("gs://public-datasets-lila/", "")
    blob = bucket.blob(blob_path)
    
    local_file = os.path.join(download_dir, os.path.basename(blob_path))
    
    if not os.path.exists(local_file):
        try:
            blob.download_to_filename(local_file)
            print(f"Downloaded: {local_file}")
        except Exception as e:
            print(f"Failed to download {blob_path}: {e}")
    else:
        # Already downloaded, skip it
        pass

print("Finished!")