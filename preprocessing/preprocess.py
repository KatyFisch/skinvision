from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "final_data.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

# MILK10k
df_milk_truth = pd.read_csv(DATA_DIR / "MILK10k_Training_GroundTruth.csv")
df_milk_meta = pd.read_csv(DATA_DIR / "MILK10k_Training_Metadata.csv")
df_milk_supplement = pd.read_csv(DATA_DIR / "MILK10k_Training_Supplement.csv")

df_milk = df_milk_meta.merge(df_milk_truth, on="lesion_id", how="left", validate="many_to_one")

# ISIC 2017
df_isic_truth = pd.read_csv(DATA_DIR / "ISIC-2017_Training_Part3_GroundTruth.csv")
df_isic_meta = pd.read_csv(DATA_DIR / "ISIC-2017_Training_Data" / "ISIC-2017_Training_Data_metadata.csv")

df_isic = df_isic_meta.merge(df_isic_truth, on="image_id", how="left", validate="many_to_one")


# --------------------------------------------------
# Define diagnosis categories
# --------------------------------------------------

categories = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]

# --------------------------------------------------
# Add diagnosis type
# --------------------------------------------------

df_milk = df_milk.merge(df_milk_supplement[["isic_id", "diagnosis_confirm_type"]], on="isic_id", how="left", validate="many_to_one")

# --------------------------------------------------
# Preprocess MILK10k
# --------------------------------------------------

df_milk["age"] = df_milk["age_approx"]
df_milk["image_id"] = df_milk["isic_id"]

df_milk["path"] = df_milk.apply(lambda row: Path("MILK10k_Training_Input") / row["lesion_id"] / f"{row['isic_id']}.jpg", axis=1)

# MILK10k has no segmentation masks
df_milk["path_segmentation"] = pd.NA

# Add dataset indicator
df_milk["dataset"] = "MILK10k"

# Keep relevant columns
milk_columns = [
    "dataset",
    "lesion_id",
    "image_id",
    "image_type",
    "age",
    "sex",
    "skin_tone_class",
    "diagnosis_confirm_type",
    *categories,
    "path",
    "path_segmentation",
]

df_milk = df_milk[milk_columns]


# --------------------------------------------------
# Preprocess ISIC 2017
# --------------------------------------------------

df_isic["age"] = df_isic["age_approximate"]
df_isic["skin_tone_class"] = pd.NA
df_isic["image_type"] = pd.NA
df_isic["diagnosis_confirm_type"] = pd.NA

df_isic["path"] = df_isic["image_id"].apply(lambda x: Path("ISIC-2017_Training_resizedImages") / f"{x}.jpg")

df_isic["path_segmentation"] = df_isic["image_id"].apply(lambda x: Path("ISIC-2017_Training_resizedSegmentation") / f"{x}_segmentation.png")

df_isic = df_isic.rename(columns={"melanoma": "MEL"})
df_isic["NV"] = ((df_isic["MEL"] == 0) & (df_isic["seborrheic_keratosis"] == 0)).astype(float)

# Create other diagnoses
for category in categories:
    if category not in ["MEL", "NV"]:
        df_isic[category] = 0

# Keep only MEL and NV diagnoses
df_isic = df_isic.loc[(df_isic["NV"] != 0) | (df_isic["MEL"] != 0)].copy()

df_isic["dataset"] = "ISIC2017"
df_isic["lesion_id"] = pd.NA

# Keep relevant columns
isic_columns = [
    "dataset",
    "lesion_id",
    "image_id",
    "image_type",
    "age",
    "sex",
    "skin_tone_class",
    "diagnosis_confirm_type",
    *categories,
    "path",
    "path_segmentation",
]

df_isic = df_isic[isic_columns]


# --------------------------------------------------
# Combine datasets
# --------------------------------------------------

df_final = pd.concat([df_milk, df_isic], ignore_index=True)

# set unknown to NA
df_final["age"] = df_final["age"].replace("unknown", pd.NA)

# Save paths as strings
df_final["path"] = df_final["path"].astype(str)
df_final["path_segmentation"] = df_final["path_segmentation"].astype("string")

# Save final dataset
df_final.to_csv(OUTPUT_PATH, index=False)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print(f"Final dataset saved to: {OUTPUT_PATH}")
print(f"Total images: {len(df_final)}")
print("\nImages per dataset:")
print(df_final["dataset"].value_counts())

print("\nMissing values:")
print(df_final.isna().sum())
