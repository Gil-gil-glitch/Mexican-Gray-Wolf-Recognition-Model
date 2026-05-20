import json 
from pathlib import Path
from tqdm import tqdm



# local paths
ANNOTATIONS_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/idaho-camera-traps.json")
OUTPUT_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_manifeest.txt")
BUCKET_PREFIX = "gs://public-datasets-lila/idaho-camera-traps/public"


wolf_category_id = 38


print("Loading masssive JSON annotations file into memory...")
with open(ANNOTATIONS_PATH, "r") as f:
    data = json.load(f)


print("Filtering annotations for wolf entries...")


wolf_image_ids = {
    ann['image_id'] for ann in tqdm(data['annotations'], desc="Annotations Pass") 
    if ann['category_id'] == wolf_category_id
}

print(f"Found matches. Building GCS download paths for {len(wolf_image_ids)} images...")
wolf_file_paths = []
for img in tqdm(data['images'], desc="Images Pass"):
    if img['id'] in wolf_image_ids:
        # Cleanly join paths ensuring no double slashes inside the relative path structure
        clean_file_name = img['file_name'].lstrip('/')
        full_gcs_path = f"{BUCKET_PREFIX}/{clean_file_name}"
        wolf_file_paths.append(full_gcs_path)

print(f"Writing paths to manifest file: {OUTPUT_PATH}")
with open(OUTPUT_PATH, "w") as f:
    for path in wolf_file_paths:
        f.write(f"{path}\n")

print(f"Finished! Manifest created with exactly {len(wolf_file_paths)} target paths.")