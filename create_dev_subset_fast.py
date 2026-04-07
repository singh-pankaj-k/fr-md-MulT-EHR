import os
import pandas as pd
import random
import shutil
import subprocess
from pathlib import Path

def create_subset_fast(source_dir, target_dir, n_patients=1000):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    (target_dir / "hosp").mkdir()
    (target_dir / "icu").mkdir()

    # 1. Select 1000 random subject_ids
    patients_file = source_dir / "hosp" / "patients.csv"
    print(f"Selecting patients from {patients_file}...")
    patients_df = pd.read_csv(patients_file)
    selected_subject_ids = random.sample(patients_df['subject_id'].unique().tolist(), min(n_patients, len(patients_df)))
    selected_subject_ids_str = set(map(str, selected_subject_ids))
    
    # Save selected IDs to a temp file for grep -f
    with open("selected_ids.txt", "w") as f:
        for sid in selected_subject_ids_str:
            f.write(f",{sid},\n|^{sid},\n,{sid}$\n")

    print(f"Selected {len(selected_subject_ids)} patients.")

    # 2. Iterate and filter
    # ONLY these tables are needed for the current project pipeline as per graph_constructor.py
    essential_tables = [
        "hosp/patients.csv",
        "hosp/admissions.csv",
        "hosp/diagnoses_icd.csv",
        "hosp/procedures_icd.csv",
        "hosp/prescriptions.csv",
        # "hosp/labevents.csv" # Skipping labevents for extra speed if requested, but let's keep it if we can
    ]
    
    # Actually, let's just skip labevents in the dev subset to make it extremely fast
    print("Skipping labevents for maximum speed in dev subset creation.")

    for table_rel in essential_tables:
        source_file = source_dir / table_rel
        target_file = target_dir / table_rel
        
        if not source_file.exists():
            print(f"Warning: {source_file} not found, skipping.")
            continue

        print(f"Processing {table_rel}...")
        with open(source_file, 'r') as f:
            header = f.readline()
        
        with open(target_file, 'w') as f_out:
            f_out.write(header)
            cmd = f"grep -F -f selected_ids.txt {source_file}"
            subprocess.run(cmd, shell=True, stdout=f_out)

    # Recreate symlinks in the root for the essential tables
    for table_rel in essential_tables:
        name = Path(table_rel).name
        target_link = target_dir / name
        if not target_link.exists():
            os.symlink(table_rel, target_link)

    os.remove("selected_ids.txt")

if __name__ == "__main__":
    create_subset_fast("data/root/mimiciv", "data/root/dev_mimiciv")
    print("Fast subset creation complete (labevents skipped).")
