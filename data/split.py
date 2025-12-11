import os
import shutil
import random
from pathlib import Path

# Directory of this script: SPINOFF-MAIN/data
ROOT = Path(__file__).parent

# Source and target directories
SRC = ROOT / "IXI" / "processed" / "png_slices"   # contains .npy files
TRAIN = ROOT / "IXI" / "processed" / "train"
VAL   = ROOT / "IXI" / "processed" / "val"
TEST  = ROOT / "IXI" / "processed" / "test"

# Reset output directories
for d in [TRAIN, VAL, TEST]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# Load NPY files
npy_files = sorted(list(SRC.glob("*.npy")))
print(f"Found {len(npy_files)} NPY slices.")

# Shuffle for randomness
random.seed(42)
random.shuffle(npy_files)

# 70 / 15 / 15 split
n = len(npy_files)
n_train = int(n * 0.70)
n_val = int(n * 0.15)
n_test = n - n_train - n_val

train_files = npy_files[:n_train]
val_files   = npy_files[n_train : n_train + n_val]
test_files  = npy_files[n_train + n_val :]

# Move files
def move_files(files, dest):
    for f in files:
        shutil.move(str(f), str(dest / f.name))

move_files(train_files, TRAIN)
move_files(val_files, VAL)
move_files(test_files, TEST)

print("\nDone moving NPY files:")
print(f"Train: {len(train_files)}")
print(f"Val:   {len(val_files)}")
print(f"Test:  {len(test_files)}")
