from pathlib import Path
import shutil
import random
import json

# =========================
# CONFIGURATION
# =========================
RAW_ROOT = Path(r"C:\Users\USER\Documents\NutriVision_AI\nutritional_ai\data\raw\food-101")
OUTPUT_ROOT = Path(r"C:\Users\USER\Documents\NutriVision_AI\nutritional_ai\data\processed")

# Choose the classes you want
SELECTED_CLASSES = [
    "pizza",
    "hamburger",
    "fried_rice",
    "steak",
    "omelette",
    "ice_cream",
    "donut",
    "apple_pie",
    "caesar_salad",
    "hot_dog",
    "spaghetti_bolognese",
    "sushi",
    "waffles",
    "french_fries",
    "grilled_cheese_sandwich",
    "chicken_wings",
    "cheesecake",
    "pancakes",
    "ramen",
    "tacos"
]

VAL_SPLIT = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# =========================
# PATHS
# =========================
IMAGES_DIR = RAW_ROOT / "images"
META_DIR = RAW_ROOT / "meta"

TRAIN_TXT = META_DIR / "train.txt"
TEST_TXT = META_DIR / "test.txt"

# =========================
# HELPERS
# =========================
def read_split_file(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def copy_image(rel_path, split):
    cls = rel_path.split("/")[0]
    img_name = rel_path.split("/")[1] + ".jpg"

    src = IMAGES_DIR / f"{rel_path}.jpg"
    dst = OUTPUT_ROOT / split / cls / img_name

    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


# =========================
# MAIN
# =========================
print("Preparing Food-101 subset...")

# Clean previous processed dataset
if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

ensure_dir(OUTPUT_ROOT)

# Read official splits
train_items = read_split_file(TRAIN_TXT)
test_items = read_split_file(TEST_TXT)

# Filter selected classes
train_items = [x for x in train_items if x.split("/")[0] in SELECTED_CLASSES]
test_items = [x for x in test_items if x.split("/")[0] in SELECTED_CLASSES]

# Group training items by class
grouped = {}
for item in train_items:
    cls = item.split("/")[0]
    grouped.setdefault(cls, []).append(item)

# Create train/val split
final_train = []
final_val = []

for cls, items in grouped.items():
    random.shuffle(items)
    n_val = max(1, int(len(items) * VAL_SPLIT))
    final_val.extend(items[:n_val])
    final_train.extend(items[n_val:])

# Copy files
for item in final_train:
    copy_image(item, "train")

for item in final_val:
    copy_image(item, "val")

for item in test_items:
    copy_image(item, "test")

# Save class mapping
class_to_idx = {cls: i for i, cls in enumerate(sorted(SELECTED_CLASSES))}
with open(OUTPUT_ROOT / "class_to_idx.json", "w") as f:
    json.dump(class_to_idx, f, indent=2)

# Summary
print("\nDataset prepared successfully!")
print(f"Classes: {len(SELECTED_CLASSES)}")
print(f"Train images: {len(final_train)}")
print(f"Validation images: {len(final_val)}")
print(f"Test images: {len(test_items)}")
print(f"Saved to: {OUTPUT_ROOT}")