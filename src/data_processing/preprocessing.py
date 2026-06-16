from pathlib import Path
from typing import Optional, Callable
from PIL import Image
from torchvision import transforms


def preprocess_dataset(input_dir: str,
                       preprocessing_function: Optional[Callable] = None,
                       output_dir: Optional[str] = None) -> None:
    """
    Recursively resize all images in a directory (and its subdirectories).

    Args:
        input_dir: Path to the root directory containing images.
        preprocessing_function: Function to apply to each image.
        output_dir: Where to save resized images. If None, overwrites originals.
    """
    transform = preprocessing_function or transforms.Resize((224, 224))
    input_path = Path(input_dir)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # rglog acts recursively through all subdirectories
    for img_file in input_path.rglob('*'):
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:

            if output_dir:
                relative_path = img_file.relative_to(input_path)
                save_file = output_path / relative_path
                save_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                save_file = img_file

            try:
                with Image.open(img_file) as img:
                    resized_img = transform(img)
                    resized_img.save(save_file)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")


if __name__ == "__main__":
    preprocess_dataset(
        input_dir="../../datasets_original/OhioSmallAnimals/",
        preprocessing_function=transforms.Resize((224, 224)),
        output_dir="../../datasets/OhioSmallAnimals/"
    )