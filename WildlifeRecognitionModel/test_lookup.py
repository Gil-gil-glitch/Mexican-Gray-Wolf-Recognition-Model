## test_lookup.py
import json
import pandas as pd
from pathlib import Path

MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
IDAHO_JSON = Path("/home/greatgilbertsoco/WolfDetect/data/idaho-camera-traps.json")
IWILDCAM_JSON = Path("/home/greatgilbertsoco/WolfDetect/data/iwildcam2019_train_annotations.json")

print(f"iWildCam JSON exists: {IWILDCAM_JSON.exists()}")
print(f"Idaho JSON exists: {IDAHO_JSON.exists()}")

# Sample what's inside Idaho if it exists
if IDAHO_JSON.exists():
    with open(IDAHO_JSON, 'r') as f:
        data = json.load(f)
    print("\n--- Raw Idaho Image Sample from JSON ---")
    print(data['images'][0])
    
print("\n--- Manifest Lookups ---")
df = pd.read_csv(MANIFEST_PATH)
canid_df = df[df['category_id'].isin([11, 15, 18])]

for source in ['iwildcam', 'idaho']:
    sample = canid_df[canid_df['dataset_source'] == source].head(2)
    for _, row in sample.iterrows():
        print(f"Manifest Row -> Source: {source} | file_name: {row['file_name']}")