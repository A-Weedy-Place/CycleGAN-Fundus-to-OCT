import os
import zipfile
import shutil
import random
import argparse
import re
from pathlib import Path

def extract_archives(raw_dir: str, extract_dir: str):
    """
    Extracts all .zip archives (including multi-part) from raw_dir into extract_dir.
    """
    raw_path = Path(raw_dir)
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)

    # Group parts by base zip name
    parts_map = {}
    for f in raw_path.iterdir():
        if f.is_file():
            # single .zip
            if f.suffix == '.zip':
                parts_map.setdefault(f.name, []).append(f)
            # multi-part e.g., train.zip.001
            else:
                m = re.match(r'(.+\.zip)\.(\d+)$', f.name)
                if m:
                    parts_map.setdefault(m.group(1), []).append(f)

    for base_name, files in parts_map.items():
        files_sorted = sorted(files, key=lambda p: (p.suffix != '.zip', p.name))
        if len(files_sorted) == 1 and files_sorted[0].suffix == '.zip':
            # single zip
            with zipfile.ZipFile(files_sorted[0], 'r') as zf:
                zf.extractall(extract_path)
        else:
            # combine parts then extract
            combined_zip = extract_path / base_name
            with open(combined_zip, 'wb') as wfd:
                for part in files_sorted:
                    with open(part, 'rb') as fd:
                        shutil.copyfileobj(fd, wfd)
            with zipfile.ZipFile(combined_zip, 'r') as zf:
                zf.extractall(extract_path)
            combined_zip.unlink()


def split_images(extracted_dir: str, train_dir: str, val_dir: str,
                 train_ratio: float = 0.8, seed: int = 42):
    """
    Splits images from extracted_dir into train_dir and val_dir per train_ratio.
    """
    ext_path = Path(extracted_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    # collect images
    img_exts = {'.jpg', '.jpeg', '.png'}
    images = [p for p in ext_path.rglob('*') if p.suffix.lower() in img_exts]

    # shuffle and split
    random.seed(seed)
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # move files
    for img in train_images:
        shutil.move(str(img), str(train_path / img.name))
    for img in val_images:
        shutil.move(str(img), str(val_path / img.name))

    print(f"Total images: {len(images)}")
    print(f"Train: {len(train_images)} -> {train_path}")
    print(f"Val:   {len(val_images)} -> {val_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract multi-part zips and split images into train/val'
    )
    parser.add_argument('--raw_dir',     required=True, help='Folder with raw zip files')
    parser.add_argument('--extract_dir', required=True, help='Temp folder to extract archives')
    parser.add_argument('--train_dir',   required=True, help='Output folder for training images')
    parser.add_argument('--val_dir',     required=True, help='Output folder for validation images')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Fraction of images for training')
    parser.add_argument('--seed',        type=int,   default=42,  help='Random seed')
    args = parser.parse_args()

    extract_archives(args.raw_dir, args.extract_dir)
    split_images(args.extract_dir, args.train_dir, args.val_dir,
                 args.train_ratio, args.seed)
