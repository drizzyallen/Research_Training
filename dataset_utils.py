from pathlib import Path
from PIL import Image
from typing import Tuple

def resize_images_in_directory(
    source_directory: str, 
    output_directory: str, 
    size: Tuple[int, int] = (224, 224)
) -> None:
    """
    Reads images from a source directory, resizes them, and saves them to an output directory.

    Args:
        source_directory: The path to the folder containing the original images.
        output_directory: The path to the folder where resized images will be saved.
        size: A tuple representing the target (width, height). Defaults to (224, 224).
    """
    source_dir = Path(source_directory)
    output_dir = Path(output_directory)
    
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Error: The source directory '{source_dir}' does not exist or is not a valid directory.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    
    count = 0
    for img_path in source_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    try:
                        resample_method = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample_method = Image.LANCZOS
                        
                    resized_img = img.resize(size, resample_method)
                    output_path = output_dir / img_path.name
                    resized_img.save(output_path)
                    count += 1
                    
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                
    print(f"Successfully processed {count} images. Saved to {output_dir}")