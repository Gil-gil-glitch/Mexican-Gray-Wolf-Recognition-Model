import pandas as pd
from pathlib import Path

# Config paths
MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
CROPS_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/hybrid_crops_v2")

# iWildCam species mapping provided by you
SPECIES_MAP = {
    0: "empty", 1: "deer", 2: "moose", 3: "squirrel", 4: "rodent",
    5: "small_mammal", 6: "elk", 7: "pronghorn_antelope", 8: "rabbit",
    9: "bighorn_sheep", 10: "fox", 11: "coyote", 12: "black_bear",
    13: "raccoon", 14: "skunk", 15: "wolf", 16: "bobcat", 17: "cat",
    18: "dog", 19: "opossum", 20: "bison", 21: "mountain_goat", 22: "mountain_lion"
}

def analyze_active_crops():
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return
    if not CROPS_DIR.exists():
        print(f"Error: Output crops folder not found at {CROPS_DIR}")
        return

    # 1. Read the manifest mapping
    print("Reading manifest mapping...")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Standardize column types for safe lookups (adjust 'id' if your column has a different name)
    df['id'] = df['id'].astype(int)
    
    # Create a quick lookup dictionary: {file_name: species_id}
    # This makes the execution instantaneous even with 160,000 rows
    manifest_lookup = dict(zip(df['file_name'], df['id']))

    # 2. Scan the current files in your active output folder
    print("Scanning active output directory...")
    current_files = list(CROPS_DIR.glob("*.png"))
    total_cropped = len(current_files)
    print(f"Total images successfully cropped so far: {total_cropped}\n")

    # 3. Cross-reference file names to extract species counts
    species_counts = {}
    
    for file_path in current_files:
        # Strip the prefix 'hybrid_wildlife_subject_' or 'hybrid_wolf_' to get the original filename
        name_str = file_path.name
        
        original_name = None
        if name_str.startswith("hybrid_wildlife_subject_"):
            original_name = name_str.replace("hybrid_wildlife_subject_", "")
        elif name_str.startswith("hybrid_wolf_"):
            original_name = name_str.replace("hybrid_wolf_", "")
            
        if original_name:
            # Revert the extension back to whatever the manifest expects (usually .jpg)
            base_stem = Path(original_name).stem
            
            matched_id = None
            for manifest_file, s_id in manifest_lookup.items():
                if Path(manifest_file).stem == base_stem:
                    matched_id = s_id
                    break
            
            if matched_id is not None:
                species_name = SPECIES_MAP.get(matched_id, f"Unknown ({matched_id})")
                species_counts[species_name] = species_counts.get(species_name, 0) + 1
            else:
                species_counts["Unmapped in Manifest"] = species_counts.get("Unmapped in Manifest", 0) + 1

    # 4. Print the leaderboard results
    print("====================================================")
    print("        LIVE SPECIES EXTRACTION BREAKDOWN           ")
    print("====================================================")
    sorted_counts = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
    for species, count in sorted_counts:
        percentage = (count / total_cropped) * 100 if total_cropped > 0 else 0
        print(f" -> {species.ljust(20)}: {str(count).rjust(6)} files ({percentage:.1f}%)")
    print("====================================================")

if __name__ == "__main__":
    analyze_active_crops()