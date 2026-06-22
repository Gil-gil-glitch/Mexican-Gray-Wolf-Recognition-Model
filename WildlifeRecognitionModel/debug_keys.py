## debug_keys.py
import json
import pandas as pd
from pathlib import Path

MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
IDAHO_JSON = Path("/home/greatgilbertsoco/WolfDetect/data/idaho-camera-traps.json")

print("--- MANIFEST SAMPLE ---")
df = pd.read_csv(MANIFEST_PATH)
canid_df = df[df['category_id'].isin([11, 15, 18])]
print(canid_df[['file_name', 'dataset_source']].head(3))

if IDAHO_JSON.exists():
    print("\n--- IDAHO JSON SAMPLE ---")
    with open(IDAHO_JSON, 'r') as f:
        data = json.load(f)
    print("Image entry sample:")
    print(data['images'][:2])