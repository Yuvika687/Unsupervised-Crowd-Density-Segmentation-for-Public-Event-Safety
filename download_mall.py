"""
download_mall.py — Download and extract the Mall Crowd Dataset
================================================================
Source:  http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html
Output:  data/mall/

The dataset contains 2000 surveillance camera frames with 13–53 people per
image, ideal for training crowd-density models on sparse crowds.

Usage:
    python3 download_mall.py
"""

import os
import sys
import zipfile
import urllib.request
import urllib.error
import shutil

# ─── Configuration ────────────────────────────────────────────────────────────

DATASET_URL = "http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip"
DOWNLOAD_DIR = "data"
EXTRACT_DIR = os.path.join(DOWNLOAD_DIR, "mall")
ZIP_PATH = os.path.join(DOWNLOAD_DIR, "mall_dataset.zip")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def download_with_progress(url, dest):
    """Download a file with a console progress bar."""
    print(f"Downloading from:\n  {url}")
    print(f"Saving to:\n  {dest}\n")

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"({mb_done:.1f} / {mb_total:.1f} MB)"
            )
            sys.stdout.flush()
        else:
            mb_done = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded {mb_done:.1f} MB")
            sys.stdout.flush()

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', -1))
            block_size = 8192
            block_num = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                block_num += 1
                out_file.write(buffer)
                _reporthook(block_num, block_size, total_size)
    except urllib.error.URLError as e:
        print(f"\n  ✗ ERROR: Failed to download. {e}")
        return False

    print("\n  ✓ Download complete.\n")
    return True

def count_files_by_ext(directory):
    """Count files by extension in a directory tree."""
    counts = {}
    total = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            counts[ext] = counts.get(ext, 0) + 1
            total += 1
    return counts, total

def print_tree(directory, prefix="", max_depth=2, current_depth=0):
    """Print a directory tree up to max_depth."""
    if current_depth >= max_depth:
        return
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return

    dirs = [e for e in entries if os.path.isdir(os.path.join(directory, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(directory, e))]

    # Show up to 5 files then summarize
    for f in files[:5]:
        print(f"{prefix}├── {f}")
    if len(files) > 5:
        print(f"{prefix}├── ... and {len(files) - 5} more files")

    for i, d in enumerate(dirs):
        connector = "└── " if i == len(dirs) - 1 else "├── "
        extension = "    " if i == len(dirs) - 1 else "│   "
        print(f"{prefix}{connector}{d}/")
        print_tree(
            os.path.join(directory, d),
            prefix=prefix + extension,
            max_depth=max_depth,
            current_depth=current_depth + 1,
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Mall Crowd Dataset — Downloader")
    print("=" * 60)
    print()

    # Check if already extracted
    if os.path.exists(EXTRACT_DIR):
        counts, total = count_files_by_ext(EXTRACT_DIR)
        if total > 100:
            print(f"Dataset already exists at {EXTRACT_DIR}/")
            print(f"  Total files: {total}")
            for ext, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"    {ext or '(no ext)':>10}: {cnt}")
            print("\nTo re-download, delete the folder and run again:")
            print(f"  rm -rf {EXTRACT_DIR}")
            return

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Step 1: Download
    if os.path.exists(ZIP_PATH):
        size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
        print(f"Zip already exists: {ZIP_PATH} ({size_mb:.1f} MB)")
        print("Skipping download.\n")
    else:
        success = download_with_progress(DATASET_URL, ZIP_PATH)
        if not success:
            print("\n  ✗ ERROR: Automatic download failed.")
            print("    Please download the dataset manually using this URL:")
            print("      http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html")
            print(f"    Then extract all its contents into exactly this folder:")
            print(f"      {EXTRACT_DIR}/")
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)
            sys.exit(1)

    # Step 2: Extract
    print(f"Extracting to {EXTRACT_DIR}/ ...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            members = zf.namelist()
            print(f"  Archive contains {len(members)} entries")
            zf.extractall(EXTRACT_DIR)
    except zipfile.BadZipFile:
        print("\n  ✗ ERROR: Downloaded file is not a valid zip archive.")
        print("    The server may be down or the URL may have changed.")
        print("    Try downloading manually from:")
        print("      http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html")
        print(f"    Then extract all its contents into exactly this folder:")
        print(f"      {EXTRACT_DIR}/")
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        sys.exit(1)

    print("  ✓ Extraction complete.\n")

    # Step 3: Flatten if needed (some zips nest inside a subfolder, like "mall_dataset")
    subdirs = [
        d for d in os.listdir(EXTRACT_DIR)
        if os.path.isdir(os.path.join(EXTRACT_DIR, d)) and d != "__MACOSX"
    ]
    if len(subdirs) == 1:
        nested = os.path.join(EXTRACT_DIR, subdirs[0])
        nested_contents = os.listdir(nested)
        print(f"  Flattening nested folder: {subdirs[0]}/")
        for item in nested_contents:
            src = os.path.join(nested, item)
            dst = os.path.join(EXTRACT_DIR, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        os.rmdir(nested)
        print("  ✓ Flattened.\n")

    # Step 4: Verify and summarize
    print("─" * 60)
    print("  VERIFICATION SUMMARY")
    print("─" * 60)

    counts, total = count_files_by_ext(EXTRACT_DIR)
    print(f"\n  Location:    {os.path.abspath(EXTRACT_DIR)}/")
    print(f"  Total files: {total}")
    print()

    print("  Files by type:")
    for ext, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {ext or '(no ext)':>10}: {cnt}")

    print(f"\n  Folder structure:")
    print(f"  {EXTRACT_DIR}/")
    print_tree(EXTRACT_DIR, prefix="  ", max_depth=2)

    print()
    print("─" * 60)

    if total > 0:
        print("  ✓ Mall dataset ready for training!")
        print()
        print("  Next steps:")
        print("    1. Review the folder structure above")
        print("    2. Ensure mall_gt.mat and frames are in place.")
    else:
        print("  ✗ WARNING: No files found after extraction.")
        print("    Check the download URL or extract manually.")

    print("─" * 60)

    # Cleanup zip (optional)
    # os.remove(ZIP_PATH)


if __name__ == "__main__":
    main()
