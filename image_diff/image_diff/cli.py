import argparse
import os
import itertools
from PIL import UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModel
import hashlib
import pathlib

# Local package imports
from .image_utils import load_image, preprocess_image_batch
from .similarity import get_image_embeddings, calculate_similarity

# Cache Helper Functions
CACHE_DIR = pathlib.Path(".image_diff_cache")

def get_cache_filepath(model_name: str) -> pathlib.Path:
    safe_model_name = hashlib.md5(model_name.encode()).hexdigest()
    return CACHE_DIR / f"embeddings_cache_{safe_model_name}.pt"

def load_embedding_cache(cache_filepath: pathlib.Path) -> dict:
    if cache_filepath.exists():
        try:
            # Ensure tensors are loaded to CPU by default with torch.load
            return torch.load(cache_filepath, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Warning: Could not load cache file {cache_filepath}: {e}")
    return {}

def save_embedding_cache(cache_filepath: pathlib.Path, cache_data: dict):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(cache_data, cache_filepath)
        print(f"Embeddings cache saved to {cache_filepath}")
    except Exception as e:
        print(f"Warning: Could not save cache file {cache_filepath}: {e}")

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

    # Model and Processor Loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    try:
        print(f"Loading image processor for '{args.model_name}'...")
        processor = AutoImageProcessor.from_pretrained(args.model_name)
        print(f"Loading model '{args.model_name}'...")
        model = AutoModel.from_pretrained(args.model_name)
        model.eval()
        model.to(device)
        print("Model and processor loaded successfully and moved to device.")
    except Exception as e: # Catch broad errors from transformers loading
        print(f"Error loading model or processor '{args.model_name}': {e}. Check model name or network connection.")
        return

    # Initialization for Caching and Processing
    cache_filepath = get_cache_filepath(args.model_name)
    embedding_cache = load_embedding_cache(cache_filepath)
    current_run_embeddings = {} # Stores img_path: {'mtime': mtime, 'embedding': tensor} for this run
    
    paths_for_computation = []
    pils_for_computation = []
    # Pre-allocate list to store final embeddings in order of original image_paths
    ordered_embeddings_list = [None] * len(image_paths)
    
    valid_image_paths_for_comparison = [] # Paths that are valid and will be part of comparison
    valid_image_indices_map = {} # Map original index to new index in valid_image_paths_for_comparison

    print(f"\nProcessing {len(image_paths)} images (checking cache, loading, and preparing for embedding)...")
    for original_idx, img_path in enumerate(image_paths):
        try:
            current_mtime = os.path.getmtime(img_path)
            pil_image = load_image(img_path) # Load PIL image early
        except (FileNotFoundError, IOError, UnidentifiedImageError) as e:
            print(f"Warning: Skipping image {os.path.basename(img_path)} due to error: {e}")
            ordered_embeddings_list[original_idx] = None # Mark as unusable
            continue

        # Cache check
        if img_path in embedding_cache and embedding_cache[img_path]['mtime'] == current_mtime:
            print(f"Using cached embedding for {os.path.basename(img_path)}")
            cached_data = embedding_cache[img_path]
            ordered_embeddings_list[original_idx] = cached_data['embedding'].cpu() # Ensure CPU
            current_run_embeddings[img_path] = cached_data # Add to current run for potential re-save
        else:
            paths_for_computation.append(img_path)
            pils_for_computation.append(pil_image)
            ordered_embeddings_list[original_idx] = "COMPUTE_PENDING" # Mark for computation

        # If image is usable (either from cache or will be computed)
        # This check might seem redundant if errors above 'continue', but kept for clarity
        if ordered_embeddings_list[original_idx] is not None:
             valid_image_indices_map[original_idx] = len(valid_image_paths_for_comparison)
             valid_image_paths_for_comparison.append(img_path)


    # Batch Preprocessing & Inference for paths_for_computation
    if pils_for_computation:
        print(f"\nPreprocessing {len(pils_for_computation)} images and generating new embeddings...")
        try:
            batched_pixel_values = preprocess_image_batch(pils_for_computation, processor)
            computed_embeddings_batch = get_image_embeddings(batched_pixel_values, model, device)
            
            computed_idx = 0
            for original_idx, item_status in enumerate(ordered_embeddings_list):
                if item_status == "COMPUTE_PENDING":
                    img_path_being_filled = image_paths[original_idx]
                    embedding_tensor = computed_embeddings_batch[computed_idx].cpu()
                    ordered_embeddings_list[original_idx] = embedding_tensor
                    current_mtime = os.path.getmtime(img_path_being_filled) # Re-get mtime
                    current_run_embeddings[img_path_being_filled] = {'mtime': current_mtime, 'embedding': embedding_tensor}
                    computed_idx += 1
            print("New embeddings generated successfully.")
        except RuntimeError as e:
            print(f"Error during batch processing or embedding generation for new images: {e}")
            # Note: Images that failed here won't be in current_run_embeddings, so won't be cached.
            # ordered_embeddings_list items will remain "COMPUTE_PENDING" or None for these.
            # We need to filter them out before torch.stack
        except Exception as e:
            print(f"An unexpected error occurred during batch processing for new images: {e}")
            # Similar handling as above
    else:
        print("\nNo new embeddings to compute. All usable images were cached or processed.")

    # Final all_embeddings Preparation
    final_embeddings_for_comparison = [
        emb for emb in ordered_embeddings_list 
        if emb is not None and isinstance(emb, torch.Tensor)
    ]

    if len(final_embeddings_for_comparison) < 2:
        print(f"\nNot enough valid embeddings ({len(final_embeddings_for_comparison)}) available for comparison. Exiting.")
        # Save any newly computed or re-validated cache entries
        embedding_cache.update(current_run_embeddings)
        save_embedding_cache(cache_filepath, embedding_cache)
        return

    all_embeddings = torch.stack(final_embeddings_for_comparison)
    
    # Rebuild valid_image_paths_for_comparison based on successfully processed embeddings
    # This is important if some "COMPUTE_PENDING" images failed during batch processing
    valid_image_paths_for_comparison = []
    temp_valid_indices_map = {}
    new_idx = 0
    for original_idx, item_status in enumerate(ordered_embeddings_list):
        if isinstance(item_status, torch.Tensor): # Only successfully processed/cached images
            valid_image_paths_for_comparison.append(image_paths[original_idx])
            temp_valid_indices_map[original_idx] = new_idx
            new_idx +=1
    # The valid_image_indices_map is not directly used in comparison loop below, but good for consistency

    if len(valid_image_paths_for_comparison) < 2:
         print(f"\nNot enough valid images ({len(valid_image_paths_for_comparison)}) after processing for comparison. Exiting.")
         embedding_cache.update(current_run_embeddings)
         save_embedding_cache(cache_filepath, embedding_cache)
         return

    # Similarity Calculation Loop
    similar_pairs_found = 0
    print("\nCalculating similarities and comparing pairs...")
    # Iterate over indices of the 'all_embeddings' tensor, which corresponds to 'valid_image_paths_for_comparison'
    for i, j in itertools.combinations(range(len(valid_image_paths_for_comparison)), 2):
        img_path1 = valid_image_paths_for_comparison[i]
        img_path2 = valid_image_paths_for_comparison[j]
        
        # all_embeddings are already stacked correctly
        embedding1 = all_embeddings[i].unsqueeze(0) 
        embedding2 = all_embeddings[j].unsqueeze(0)

        try:
            similarity_score = calculate_similarity(embedding1, embedding2)

            similarity_score = calculate_similarity(embedding1, embedding2)

            if similarity_score >= args.threshold:
                print(f"Found similar pair: {os.path.basename(img_path1)} and {os.path.basename(img_path2)} - Similarity: {similarity_score:.4f}")
                similar_pairs_found += 1
        except RuntimeError as e:
            print(f"Error calculating similarity for pair ({os.path.basename(img_path1)}, {os.path.basename(img_path2)}): {e}")
        except Exception as e:
            print(f"Unexpected error calculating similarity for pair ({os.path.basename(img_path1)}, {os.path.basename(img_path2)}): {e}")

    if similar_pairs_found == 0:
        print("\nNo pairs found above the similarity threshold.")
    else:
        print(f"\nFound {similar_pairs_found} pair(s) above the similarity threshold.")

    # Save updated cache
    embedding_cache.update(current_run_embeddings)
    save_embedding_cache(cache_filepath, embedding_cache)

if __name__ == '__main__':
    main()
