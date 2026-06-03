#
## eda_pipeline.py
#
#  This script performs an exploratory data analysis (EDA) on the results of the cascaded 
#  dynamic inference pipeline and the cascaded inference pipelinefor wildlife recognition. 
#  It analyzes the performance of the gatekeeper stage by comparing the dropped samples 
#  against a randomly sampled validation set from the master manifest. The EDA focuses 
#  on identifying patterns in false negatives, species-specific leakage, and potential 
#  dataset biases to inform future improvements in the pipeline's gating strategy. 
#


import pandas as pd
import sys
from pathlib import Path

# Path to the manifest file used in your cascaded inference script
MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")

def analyze_pipeline_leakage(log_path=None):
    if log_path:
        DROPPED_LOG_PATH = Path(log_path)

    print("=====================================================================")
    # Visual check to ensure files exist before executing analysis
    print(f"RUNNING PIPELINE GATEKEEPER EDA")
    print("=====================================================================")
    
    if not DROPPED_LOG_PATH.exists():
        print(f"[CRITICAL] '{DROPPED_LOG_PATH}' not found. Please ensure your previous 1,000-image simulation run completed successfully.")
        return

    # 1Data loading and preparation
    master_df = pd.read_csv(MANIFEST_PATH)
    dropped_df = pd.read_csv(DROPPED_LOG_PATH)
    
    # Reconstruct the 1,000-image sampled validation set
    # We use the exact same random state (101) and filtering logic used in simulation
    target_categories = [11, 15, 18]
    full_sample_set = master_df[master_df['category_id'].isin(target_categories)].sample(n=1000, random_state=101)
    
    # Create clean lookup mappings matching your script's architecture
    id_to_name = {15: "Mexican Gray Wolf", 11: "Coyote", 18: "Domestic Dog"}
    dropped_filenames = set(dropped_df['dropped_file_names'].tolist())
    
    # Classify each sample in the validation set as either 'Processed' or 'Dropped'
    analysis_records = []
    for _, row in full_sample_set.iterrows():
        fname = row['file_name']
        cat_id = int(row['category_id'])
        species_name = id_to_name[cat_id]
        
        status = "Dropped (False Negative)" if fname in dropped_filenames else "Processed (Success Gate)"
        
        analysis_records.append({
            'file_name': fname,
            'species': species_name,
            'category_id': cat_id,
            'pipeline_status': status,
            'dataset_source': row['dataset_source']
        })
        
    summary_df = pd.DataFrame(analysis_records)
    
    # =====================================================================
    # METRIC 1: GLOBAL DROPPED VS PROCESSED BREAKDOWN
    # =====================================================================
    print("\nGLOBAL GATEKEEPER PERFORMANCE")
    print("-" * 50)
    global_counts = summary_df['pipeline_status'].value_counts()
    global_pct = summary_df['pipeline_status'].value_counts(normalize=True) * 100
    for idx in global_counts.index:
        print(f" * {idx}: {global_counts[idx]} images ({global_pct[idx]:.2f}%)")
        
    # =====================================================================
    # METRIC 2: SPECIES-SPECIFIC LEAKAGE CROSS-TABULATION
    # =====================================================================
    print("\nSPECIES-SPECIFIC CROSS-TABULATION")
    print("-" * 50)
    cross_tab = pd.crosstab(summary_df['species'], summary_df['pipeline_status'])
    cross_tab_pct = pd.crosstab(summary_df['species'], summary_df['pipeline_status'], normalize='index') * 100
    
    for species in cross_tab.index:
        processed_num = cross_tab.loc[species, "Processed (Success Gate)"]
        processed_pct = cross_tab_pct.loc[species, "Processed (Success Gate)"]
        dropped_num = cross_tab.loc[species, "Dropped (False Negative)"]
        dropped_pct = cross_tab_pct.loc[species, "Dropped (False Negative)"]
        total_instances = processed_num + dropped_num
        
        print(f"Species: {species:<20} | Total Samples: {total_instances}")
        print(f"   ↳ Passed Gate: {processed_num:<4} ({processed_pct:.2f}%)")
        print(f"   ↳ Dropped:     {dropped_num:<4} ({dropped_pct:.2f}%)")
        print("-" * 50)

    # =====================================================================
    # METRIC 3: DATASET SOURCE BIAS ANALYSIS (iWildCam vs. Idaho Wolves)
    # =====================================================================
    print("\nENVIRONMENTAL SOURCE BIAS ANALYSIS")
    print("-" * 50)
    source_tab = pd.crosstab(summary_df['dataset_source'], summary_df['pipeline_status'], normalize='index') * 100
    for source in source_tab.index:
        print(f" * Source Location: {source:<12} -> Passed: {source_tab.loc[source, 'Processed (Success Gate)']:.2f}% | Dropped: {source_tab.loc[source, 'Dropped (False Negative)']:.2f}%")
    print("=====================================================================\n")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_dropped_samples_csv>")
        sys.exit(1)
    
    log_path_arg = sys.argv[1] # Expecting the path to the 'dynamic_dropped_samples_log.csv' generated from the cascaded_dynamic_inference.py run
    
    analyze_pipeline_leakage(log_path_arg)