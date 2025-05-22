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

def preprocess_image(image: Image.Image, processor):
    """
    Preprocesses a single image using a pre-initialized Hugging Face AutoImageProcessor.

    Args:
        image: A PIL.Image.Image object.
        processor: An initialized Hugging Face AutoImageProcessor instance.

    Returns:
        The processed image tensor (pixel_values).
    """
    try:
        processed_output = processor(images=image, return_tensors='pt')
        return processed_output.pixel_values
    except Exception as e:
        # Broad exception for now, can be refined if specific exceptions from transformers are known
        raise RuntimeError(f"Error during single image preprocessing: {e}")

def preprocess_image_batch(images: list[Image.Image], processor):
    """
    Preprocesses a batch of images using a pre-initialized Hugging Face AutoImageProcessor.

    Args:
        images: A list of PIL.Image.Image objects.
        processor: An initialized Hugging Face AutoImageProcessor instance.

    Returns:
        A tensor containing the processed pixel values for the batch of images.
    """
    try:
        processed_output = processor(images=images, return_tensors='pt')
        return processed_output.pixel_values
    except Exception as e:
        # Broad exception for now, can be refined
        raise RuntimeError(f"Error during batch image preprocessing: {e}")
