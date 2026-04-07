import os
import pandas as pd
import random
import shutil
from pathlib import Path

def create_subset_pandas(source_dir, target_dir, n_patients=1000):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if not source_dir.exists():
        print(f"Source {source_dir} not found.")
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    # 1. Select n_patients random subject_ids
    # Search for patients file in various possible locations/casings
    possible_patient_files = [
        source_dir / "hosp/patients.csv",
        source_dir / "hosp/patients.csv.gz",
        source_dir / "PATIENTS.csv",
        source_dir / "PATIENTS.csv.gz",
        source_dir / "patients.csv",
        source_dir / "patients.csv.gz"
    ]
    
    patients_file = None
    for pf in possible_patient_files:
        if pf.exists():
            patients_file = pf
            break

    if not patients_file:
        print(f"Error: Patients file not found in {source_dir}")
        return

    print(f"Selecting patients from {patients_file}...")
    patients_df = pd.read_csv(patients_file)
    # Handle both subject_id and SUBJECT_ID
    sid_col = 'subject_id' if 'subject_id' in patients_df.columns else 'SUBJECT_ID'
    
    all_subject_ids = patients_df[sid_col].unique().tolist()
    selected_subject_ids = set(random.sample(all_subject_ids, min(n_patients, len(all_subject_ids))))
    print(f"Selected {len(selected_subject_ids)} patients.")

    # 2. Iterate and filter using pandas
    for root, dirs, files in os.walk(source_dir):
        rel_path = Path(root).relative_to(source_dir)
        target_root = target_dir / rel_path
        target_root.mkdir(parents=True, exist_ok=True)

        for file in files:
            source_file = Path(root) / file
            
            # Handle .csv and .csv.gz
            is_csv = file.lower().endswith(".csv")
            is_gz = file.lower().endswith(".csv.gz")
            
            if not (is_csv or is_gz):
                target_file = target_root / file
                shutil.copy2(source_file, target_file)
                continue

            # Target will always be .csv for simplicity in dev mode
            if is_gz:
                target_file = target_root / file[:-3]
            else:
                target_file = target_root / file

            print(f"Processing {source_file.relative_to(source_dir)}...")
            # Use chunks to handle potentially large files
            try:
                # Read header only first
                header_df = pd.read_csv(source_file, nrows=0)
                cols = [c.lower() for c in header_df.columns]
                
                if 'subject_id' in cols:
                    # Find actual casing
                    actual_col = header_df.columns[cols.index('subject_id')]
                    chunks = pd.read_csv(source_file, chunksize=100000)
                    first_chunk = True
                    for chunk in chunks:
                        filtered_chunk = chunk[chunk[actual_col].isin(selected_subject_ids)]
                        if not filtered_chunk.empty:
                            filtered_chunk.to_csv(target_file, mode='a', index=False, header=first_chunk)
                            first_chunk = False
                    if first_chunk: # Create empty with header if no match
                        header_df.to_csv(target_file, index=False)
                else:
                    # No subject_id, copy as-is (but decompress if gz)
                    if is_gz:
                        with pd.read_csv(source_file, chunksize=100000) as reader:
                            for i, chunk in enumerate(reader):
                                chunk.to_csv(target_file, mode='a', index=False, header=(i==0))
                    else:
                        shutil.copy2(source_file, target_file)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                shutil.copy2(source_file, target_file)

if __name__ == "__main__":
    print("--- Creating MIMIC-IV dev subset ---")
    create_subset_pandas("data/root/mimiciv", "data/root/dev_mimiciv", n_patients=1000)
    
    print("\n--- Creating MIMIC-III dev subset ---")
    create_subset_pandas("data/root/mimiciii", "data/root/dev_mimiciii", n_patients=1000)
