from PIL import Image, UnidentifiedImageError
import torch # Added for MTCNN
from facenet_pytorch import MTCNN # Added for MTCNN

# AutoImageProcessor import removed

def load_image(image_path: str, mtcnn: MTCNN = None, device: torch.device = None) -> Image.Image:
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
        image = image.convert('RGB')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except (IOError, UnidentifiedImageError) as e:
        raise IOError(f"Error opening or reading image file {image_path}: {e}")

    if mtcnn and device:
        faces = detect_faces(image, mtcnn, device)
        if faces:
            # For now, return the first (and likely largest or only) detected face.
            # TODO: Add strategy for multiple faces (e.g., largest)
            return faces[0]
        else:
            # Log warning? For now, return original image if no face detected
            # This behavior will be handled by the CLI to inform the user.
            # print(f"Warning: No faces detected in {image_path}. Using original image.") # Placeholder for logging
            return image # Fallback to original image
    return image


def detect_faces(image: Image.Image, mtcnn: MTCNN, device: torch.device) -> list[Image.Image]:
    """
    Detects faces in an image using MTCNN and returns a list of cropped face images.

    Args:
        image: A PIL.Image.Image object.
        mtcnn: An initialized facenet_pytorch.MTCNN model.
        device: The torch.device the model is on.

    Returns:
        A list of PIL.Image.Image objects, each containing a cropped face.
        Returns an empty list if no faces are detected or if an error occurs.
    """
    if not mtcnn or not isinstance(image, Image.Image):
        return []

    try:
        # MTCNN expects a PIL image directly
        # The `boxes` are the bounding boxes of detected faces
        # The `probs` are the probabilities of those detections
        # `landmarks` are also detected but not used here.
        boxes, _ = mtcnn.detect(image)

        face_images = []
        if boxes is not None:
            for box in boxes:
                # Ensure box coordinates are integers and within image bounds
                x1, y1, x2, y2 = [int(b) for b in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.width, x2), min(image.height, y2)

                if x1 < x2 and y1 < y2: # Ensure valid crop dimensions
                    cropped_face = image.crop((x1, y1, x2, y2))
                    face_images.append(cropped_face)
        return face_images
    except Exception as e:
        # print(f"Error during face detection: {e}") # Placeholder for logging
        return []


# preprocess_image function removed

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
