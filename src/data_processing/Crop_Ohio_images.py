from pathlib import Path
from PIL import Image


def clean_dataset(input_dir: str, output_dir: str, crop_top: int = 50, crop_bottom: int = 45) -> None:
    """
    Crops the top and bottom off images recursively in a directory and saves them to a new structure.

    Args:
        input_dir: Path to the original dataset.
        output_dir: Path to save the cropped dataset.
        crop_top: Pixels to remove from the top.
        crop_bottom: Pixels to remove from the bottom.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Robustly find all JPG files, regardless of case
    image_paths = [p for p in input_path.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg']]

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images. Starting crop...")

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                width, height = img.size

                # Define the crop box
                left = 0
                top = crop_top
                right = width
                bottom = height - crop_bottom

                # Crop the image
                cropped_img = img.crop((left, top, right, bottom))

                # Reconstruct folder structure
                relative_path = img_path.relative_to(input_path)
                out_file_path = output_path / relative_path
                out_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Bulletproof Save
                cropped_img.save(out_file_path, format='JPEG', quality=80, optimize=True)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print("Processing complete!")


if __name__ == "__main__":
    INPUT_FOLDER = "../../datasets/OhioSmallAnimals"
    OUTPUT_FOLDER = "../../datasets/OhioSmallAnimals_Cleaned"

    clean_dataset(INPUT_FOLDER, OUTPUT_FOLDER)