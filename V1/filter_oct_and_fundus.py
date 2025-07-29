# filter_oct_and_fundus.py

"""
Script to filter diabetic retinopathy (DR)-positive images (NPDR/PDR) from the
OCT-AND-EYE-FUNDUS dataset and copy them into your project’s fundus and OCT folders.
"""

# ---------- CONFIGURATION ------------
RAW_DATA_DIR         = r"D:\New folder (2)\OCT\OCT-AND-EYE-FUNDUS-DATASET-main"
EYE_CSV              = "EYE FUNDUS.csv"
OCT_CSV              = "OCT.csv"
SOURCE_FUNDUS_FOLDER = "EYE FUNDUS"
SOURCE_OCT_FOLDER    = "OCT"
OUT_FUNDUS           = r"D:\projects\retina-gan\data\fundus\trainA"
OUT_OCT              = r"D:\projects\retina-gan\data\oct\trainB"
DR_COLUMN            = "DR"
FNAME_COLUMN         = "Name"
EXT_COLUMN           = "Format"
DR_VALUES            = ["NPDR", "PDR"]
# -------------------------------------

import os
import pandas as pd
import shutil
from pathlib import Path


def build_file_map(src_folder):
    """
    Recursively builds a map of filename -> full path for images in src_folder.
    """
    file_map = {}
    for p in Path(src_folder).rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            file_map[p.name.lower()] = str(p)
    return file_map


def filter_and_copy(csv_path, src_folder, out_folder):
    """
    Reads csv at csv_path, filters for rows where DR_COLUMN is in DR_VALUES,
    then copies matching files from src_folder (recursive) into out_folder.
    """
    df = pd.read_csv(csv_path)
    print(f"CSV columns ({os.path.basename(csv_path)}): {df.columns.tolist()}")
    df_filtered = df[df[DR_COLUMN].isin(DR_VALUES)]
    print(f"Filtered {len(df_filtered)} rows with {DR_COLUMN} in {DR_VALUES}")

    os.makedirs(out_folder, exist_ok=True)
    file_map = build_file_map(os.path.join(RAW_DATA_DIR, src_folder))

    copied = 0
    missing = 0
    for _, row in df_filtered.iterrows():
        name = str(row[FNAME_COLUMN])
        ext = str(row.get(EXT_COLUMN, '')).lower()
        # Construct filename
        filename = f"{name}.{ext}" if ext and not name.lower().endswith(f".{ext}") else name
        filename_lower = filename.lower()
        src_path = file_map.get(filename_lower)
        if src_path:
            dst_path = os.path.join(out_folder, filename)
            shutil.copy(src_path, dst_path)
            copied += 1
        else:
            missing += 1
    print(f"Copied {copied} files to {out_folder}. {missing} missing.")


if __name__ == '__main__':
    print("\nProcessing fundus images...")
    filter_and_copy(os.path.join(RAW_DATA_DIR, EYE_CSV), SOURCE_FUNDUS_FOLDER, OUT_FUNDUS)

    print("\nProcessing OCT images...")
    filter_and_copy(os.path.join(RAW_DATA_DIR, OCT_CSV), SOURCE_OCT_FOLDER, OUT_OCT)
