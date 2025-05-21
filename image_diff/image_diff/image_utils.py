from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor

def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from the given file path and converts it to RGB format.

    Args:
        image_path: The path to the image file.

    Returns:
        A PIL.Image.Image object.

    Raises:
        FileNotFoundError: If the image path does not exist.
        IOError: If the file cannot be opened as an image or is not a supported format.
    """
    try:
        image = Image.open(image_path)
        return image.convert('RGB')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except (IOError, UnidentifiedImageError) as e:
        raise IOError(f"Error opening or reading image file {image_path}: {e}")

def preprocess_image(image: Image.Image, model_name_or_path: str):
    """
    Preprocesses an image using a Hugging Face AutoImageProcessor.

    Args:
        image: A PIL.Image.Image object.
        model_name_or_path: The name or path of the Hugging Face model.

    Returns:
        The processed image tensor (pixel_values).
    """
    try:
        processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        processed_image = processor(images=image, return_tensors='pt')
        return processed_image.pixel_values
    except Exception as e:
        # Broad exception for now, can be refined if specific exceptions from transformers are known
        raise RuntimeError(f"Error preprocessing image with model {model_name_or_path}: {e}")
