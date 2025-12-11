import shutil
from pathlib import Path

ROOT = Path(__file__).parent

TRAIN = ROOT / "IXI" / "processed" / "train"
VAL   = ROOT / "IXI" / "processed" / "val"
TEST  = ROOT / "IXI" / "processed" / "test"

DEST = ROOT / "IXI" / "processed" / "png_slices"

DEST.mkdir(parents=True, exist_ok=True)

def copy_back(src, dest):
    if not src.exists():
        return
    for f in src.glob("*.npy"):
        shutil.copy2(f, dest / f.name)

print("Copying NPY files back into png_slices…")

copy_back(TRAIN, DEST)
copy_back(VAL, DEST)
copy_back(TEST, DEST)

print("✔ Done. All files restored into png_slices (duplicates allowed).")
