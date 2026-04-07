import os
import pandas as pd
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def create_subset(source_dir, target_dir, n_patients=1000):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Create target directory structure
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    (target_dir / "hosp").mkdir()
    (target_dir / "icu").mkdir()

    # 1. Select 1000 random subject_ids from patients.csv
    patients_file = source_dir / "hosp" / "patients.csv"
    if not patients_file.exists():
        print(f"Error: {patients_file} not found.")
        return

    print(f"Loading patients from {patients_file}...")
    patients_df = pd.read_csv(patients_file)
    all_subject_ids = patients_df['subject_id'].unique().tolist()
    
    # Take 1000 random patients, or all if less than 1000
    selected_subject_ids = set(random.sample(all_subject_ids, min(n_patients, len(all_subject_ids))))
    print(f"Selected {len(selected_subject_ids)} patients.")

    # 2. Iterate through all files and subdirectories
    for root, dirs, files in os.walk(source_dir):
        # Skip hidden files or specific non-data files if necessary
        rel_path = Path(root).relative_to(source_dir)
        target_root = target_dir / rel_path
        target_root.mkdir(exist_ok=True)

        for file in files:
            if not file.endswith(".csv"):
                # Copy metadata files as-is
                shutil.copy2(Path(root) / file, target_root / file)
                continue
            
            source_file = Path(root) / file
            target_file = target_root / file
            
            print(f"Processing {source_file.relative_to(source_dir)}...")
            
            # Read header to check if subject_id is present
            header = pd.read_csv(source_file, nrows=0).columns.tolist()
            
            if 'subject_id' in header:
                # Filter by subject_id
                # Use chunking for large files to be memory efficient
                chunks = pd.read_csv(source_file, chunksize=100000)
                first_chunk = True
                for chunk in chunks:
                    filtered_chunk = chunk[chunk['subject_id'].isin(selected_subject_ids)]
                    if not filtered_chunk.empty:
                        filtered_chunk.to_csv(target_file, mode='a', index=False, header=first_chunk)
                        first_chunk = False
                # If no rows were matched, create an empty file with header
                if first_chunk:
                    pd.DataFrame(columns=header).to_csv(target_file, index=False)
            else:
                # Copy metadata files as-is
                shutil.copy2(source_file, target_file)

    # 3. Handle symlinks if they exist in the root (based on previous ls output)
    # The ls showed admissions.csv@ etc. in the root, which are likely symlinks to hosp/
    # We will recreate them as symlinks in dev_mimiciv if they point to files we created
    for item in source_dir.iterdir():
        if item.is_symlink():
            link_target = item.readlink()
            new_link = target_dir / item.name
            # Adjust the link target to be relative within the new structure if possible
            if not new_link.exists():
                os.symlink(link_target, new_link)

if __name__ == "__main__":
    create_subset("data/root/mimiciv", "data/root/dev_mimiciv", n_patients=1000)
    print("Subset creation complete.")
