from pathlib import Path

from PIL import Image

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_DIR / "data/ISIC-2017_Training_Data"
MASK_DIR = PROJECT_DIR / "data/ISIC-2017_Training_Part1_GroundTruth"

OUTPUT_IMAGE_DIR = PROJECT_DIR / "data/ISIC-2017_Training_resizedImages"
OUTPUT_MASK_DIR = PROJECT_DIR / "data/ISIC-2017_Training_resizedSegmentation"

# Target dimensions
OUTPUT_SIZE = (600, 450)


def crop_to_4_3(image):
    """Center-crop an image to a 4:3 aspect ratio."""
    width, height = image.size
    target_ratio = 4 / 3
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        box = (left, 0, left + new_width, height)
    else:
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        box = (0, top, width, top + new_height)

    return image.crop(box)


def preprocess_image(input_path, output_path, is_mask=False):
    """Crop to 4:3 and resize an image or segmentation mask."""

    mode = "L" if is_mask else "RGB"
    interpolation = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS

    with Image.open(input_path) as image:
        image = image.convert(mode)
        image = crop_to_4_3(image)
        image = image.resize(OUTPUT_SIZE, interpolation)
        image.save(output_path)


def main():
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    # Process images
    image_files = sorted(IMAGE_DIR.glob("*.jpg"))

    for i, path in enumerate(image_files, 1):
        output_path = OUTPUT_IMAGE_DIR / path.name
        preprocess_image(path, output_path)
        print(f"Image {i}/{len(image_files)}: {path.name}")

    # Process masks
    mask_files = sorted(MASK_DIR.glob("*.png"))

    for i, path in enumerate(mask_files, 1):
        output_path = OUTPUT_MASK_DIR / path.name
        preprocess_image(path, output_path, is_mask=True)
        print(f"Mask {i}/{len(mask_files)}: {path.name}")

    print("\nPreprocessing complete!")
    print(f"Images processed: {len(image_files)}")
    print(f"Masks processed: {len(mask_files)}")


if __name__ == "__main__":
    main()
