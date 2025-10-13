import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import zipfile
import kaggle

# --- Configuration ---
DATA_DIR = 'data'
# Kaggle dataset identifier
KAGGLE_DATASET = 'jangedoo/utkface-new'
# The Kaggle API downloads a zip file
DATASET_ARCHIVE = os.path.join(DATA_DIR, 'utkface-new.zip') 
# The images are inside this folder after extraction
IMAGE_DIR = os.path.join(DATA_DIR, 'utkface_aligned_cropped', 'UTKFace')
OUTPUT_CSV = os.path.join(DATA_DIR, 'labels.csv')

def download_and_extract():
    """Downloads and extracts the UTKFace dataset using the Kaggle API."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(IMAGE_DIR):
        print("Dataset already exists. Skipping download and extraction.")
        return

    print("Downloading dataset from Kaggle... (this may take a few minutes)")
    try:
        # Use the Kaggle API to download the dataset
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=False)
    except Exception as e:
        print(f"❌ Kaggle API Error: {e}")
        print("Please ensure your kaggle.json token is correctly placed in C:\\Users\\<Your-Username>\\.kaggle\\")
        return

    print(f"Extracting {DATASET_ARCHIVE}...")
    try:
        with zipfile.ZipFile(DATASET_ARCHIVE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
    except Exception as e:
        print(f"Extraction failed: {e}. Please delete the 'data' folder and try again.")
        return
        
    os.remove(DATASET_ARCHIVE) # Clean up the zip file
    print("✅ Dataset ready.")

def mock_label_hair(age, gender):
    """MOCK function to simulate hair length labels."""
    return np.random.choice([0, 1], p=[0.3, 0.7]) if gender == 1 else np.random.choice([0, 1], p=[0.8, 0.2])

def create_labels_csv():
    """Parses filenames and creates a CSV with labels."""
    if not os.path.exists(IMAGE_DIR):
        print("❌ Error: Image directory not found. Download or extraction may have failed.")
        return

    print(f"Parsing filenames to create {OUTPUT_CSV}...")
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]
    records = []

    for filename in tqdm(image_files, desc="Processing files"):
        filepath = os.path.join(IMAGE_DIR, filename)
        try:
            parts = filename.split('_')
            if len(parts) >= 3:
                age, gender = int(parts[0]), int(parts[1])
                if 0 < age < 120 and gender in [0, 1]:
                    hair_length = mock_label_hair(age, gender)
                    records.append([filepath, age, gender, hair_length])
        except (ValueError, IndexError):
            continue
    
    if not records:
        print("\n❌ CRITICAL ERROR: No valid image files were found in the expected directory.")
        print(f"Please check the contents of the '{IMAGE_DIR}' directory.")
        return

    df = pd.DataFrame(records, columns=['filepath', 'age', 'gender', 'hair_length'])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Successfully created labels CSV with {len(df)} entries.")
    print("\n--- Data Sample ---")
    print(df.head())

if __name__ == '__main__':
    download_and_extract()
    create_labels_csv()