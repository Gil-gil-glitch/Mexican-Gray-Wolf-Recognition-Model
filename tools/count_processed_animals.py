import pandas as pd
from pathlib import Path

MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
CROPS_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/hybrid_crops_v2")

SPECIES_MAP = {
    0: "empty", 1: "deer", 2: "moose", 3: "squirrel", 4: "rodent",
    5: "small_mammal", 6: "elk", 7: "pronghorn_antelope", 8: "rabbit",
    9: "bighorn_sheep", 10: "fox", 11: "coyote", 12: "black_bear",
    13: "raccoon", 14: "skunk", 15: "wolf", 16: "bobcat", 17: "cat",
    18: "dog", 19: "opossum", 20: "bison", 21: "mountain_goat", 22: "mountain_lion"
}

def analyze_active_crops_fast():
    if not MANIFEST_PATH.exists() or not CROPS_DIR.exists():
        print("Paths misaligned. Check manifest/crops directory configuration.")
        return

    print("Reading manifest mapping...")
    df = pd.read_csv(MANIFEST_PATH)
    
    # --- OPTIMIZATION HACK ---
    # Strip extensions from manifest names immediately and build a direct hash map.
    # This completely eliminates the inner sequential loop!
    print("Building high-speed index mapping...")
    manifest_lookup = {Path(row['file_name']).stem: int(row['category_id']) for _, row in df.iterrows()}

    print("Scanning active output directory...")
    current_files = list(CROPS_DIR.glob("*.png"))
    total_cropped = len(current_files)
    print(f"Total images successfully cropped so far: {total_cropped}\n")

    species_counts = {}
    
    for file_path in current_files:
        name_str = file_path.name
        
        # Isolate the core filename stem
        original_stem = None
        if name_str.startswith("hybrid_wildlife_subject_"):
            original_stem = name_str.replace("hybrid_wildlife_subject_", "").split('.')[0]
        elif name_str.startswith("hybrid_wolf_"):
            original_stem = name_str.replace("hybrid_wolf_", "").split('.')[0]
            
        if original_stem:
            # High-speed O(1) hash map retrieval
            matched_id = manifest_lookup.get(original_stem)
            
            if matched_id is not None:
                species_name = SPECIES_MAP.get(matched_id, f"Unknown ({matched_id})")
                species_counts[species_name] = species_counts.get(species_name, 0) + 1
            else:
                species_counts["Unmapped in Manifest"] = species_counts.get("Unmapped in Manifest", 0) + 1

    print("====================================================")
    print("        LIVE SPECIES EXTRACTION BREAKDOWN           ")
    print("====================================================")
    sorted_counts = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
    for species, count in sorted_counts:
        percentage = (count / total_cropped) * 100 if total_cropped > 0 else 0
        print(f" -> {species.ljust(20)}: {str(count).rjust(6)} files ({percentage:.1f}%)")
    print("====================================================")

if __name__ == "__main__":
    analyze_active_crops_fast()