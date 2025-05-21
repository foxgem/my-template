import argparse
import os
import itertools
from PIL import UnidentifiedImageError
import torch # For device check and potential torch-specific exceptions
from transformers import AutoImageProcessor, AutoModel # For pre-loading

# Local package imports
from .image_utils import load_image, preprocess_image
from .similarity import get_image_embedding, calculate_similarity

def main():
    parser = argparse.ArgumentParser(description="Compares images in a folder and reports similar pairs.")
    parser.add_argument("folder_path", type=str, help="Path to the directory containing images.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for reporting image pairs (0.0 to 1.0). Default: 0.9",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="Hugging Face model name or path to a local model for feature extraction. Default: 'google/vit-base-patch16-224-in21k'",
    )
    args = parser.parse_args()

    # Validate folder_path
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder not found: {args.folder_path}")
        return
    if not os.path.isdir(args.folder_path):
        print(f"Error: Provided path is not a directory: {args.folder_path}")
        return

    # Image Discovery
    supported_extensions = ('.png', '.jpg', '.jpeg')
    image_paths = []
    for item in os.listdir(args.folder_path):
        if item.lower().endswith(supported_extensions):
            image_paths.append(os.path.join(args.folder_path, item))

    if len(image_paths) < 2:
        print(f"Found {len(image_paths)} image(s). Need at least two images to compare.")
        return
    
    print(f"Found {len(image_paths)} images in '{args.folder_path}'.")

    # Model and Processor Loading Attempt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        print(f"Loading model '{args.model_name}' and its processor...")
        _ = AutoImageProcessor.from_pretrained(args.model_name)
        _ = AutoModel.from_pretrained(args.model_name)
        print("Model and processor loaded successfully.")
    except OSError as e:
        print(f"Error loading model or processor '{args.model_name}': {e}. Check model name or network connection.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading model or processor '{args.model_name}': {e}")
        return

    # Image Comparison Loop
    similar_pairs_found = 0
    for img_path1, img_path2 in itertools.combinations(image_paths, 2):
        print(f"\nComparing '{os.path.basename(img_path1)}' and '{os.path.basename(img_path2)}'...")
        try:
            # Load and Preprocess Image 1
            image1 = load_image(img_path1)
            # preprocess_image returns the tensor directly (pixel_values)
            processed_image1_tensor = preprocess_image(image1, args.model_name)

            # Load and Preprocess Image 2
            image2 = load_image(img_path2)
            processed_image2_tensor = preprocess_image(image2, args.model_name)

            # Get Embeddings
            # get_image_embedding expects the pixel_values tensor
            embedding1 = get_image_embedding(processed_image1_tensor, args.model_name)
            embedding2 = get_image_embedding(processed_image2_tensor, args.model_name)

            # Calculate Similarity
            similarity_score = calculate_similarity(embedding1, embedding2)

            # Report if above threshold
            if similarity_score >= args.threshold:
                print(f"Found similar pair: {os.path.basename(img_path1)} and {os.path.basename(img_path2)} - Similarity: {similarity_score:.4f}")
                similar_pairs_found += 1
            else:
                print(f"Similarity: {similarity_score:.4f} (Below threshold of {args.threshold})")

        except FileNotFoundError as e:
            print(f"Error: Image not found: {e.filename}")
        except (IOError, UnidentifiedImageError) as e: # IOError can be raised by load_image
            # Try to determine which path caused the error if possible
            path_in_error = ""
            if hasattr(e, 'filename') and e.filename: # FileNotFoundError has filename
                 path_in_error = e.filename
            # For general IOError/UnidentifiedImageError from PIL, it might not have a filename attribute
            # or it might be set by our custom load_image
            # We can infer based on which image was being processed if the error message itself is not specific
            # This part is tricky as the exception might not directly tell us which file it was.
            # The f-string in load_image helps.
            print(f"Error: Could not read or identify image. Details: {e}")
        except RuntimeError as e: # Catch custom RuntimeErrors from our functions
            print(f"Processing error for pair ({os.path.basename(img_path1)}, {os.path.basename(img_path2)}): {e}")
        except Exception as e:
            print(f"An unexpected error occurred processing pair ({os.path.basename(img_path1)}, {os.path.basename(img_path2)}): {e}")

    if similar_pairs_found == 0:
        print("\nNo pairs found above the similarity threshold.")
    else:
        print(f"\nFound {similar_pairs_found} pair(s) above the similarity threshold.")

if __name__ == '__main__':
    main()
