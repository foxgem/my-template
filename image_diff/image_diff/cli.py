import argparse
import os
import itertools
# UnidentifiedImageError import removed
import torch
from transformers import AutoImageProcessor, AutoModel
import hashlib
import lancedb
import pyarrow as pa
import json # For JSON output
import csv # For CSV output

# Local package imports
from .image_utils import load_image, preprocess_image_batch
from .similarity import get_image_embeddings, calculate_similarity

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
    parser.add_argument(
        "--output_file",
        type=str,
        default="similar_pairs.csv", # Default to CSV
        help="Path to save the results of similar pairs. E.g., similar_pairs.csv or similar_pairs.json",
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

    # LanceDB Initialization
    db_uri = os.path.join(args.folder_path, ".lancedb")
    db = lancedb.connect(db_uri)
    model_hash = hashlib.md5(args.model_name.encode()).hexdigest()[:12]
    table_name = f"image_embeddings_{model_hash}"
    tbl = None
    try:
        tbl = db.open_table(table_name)
        print(f"Opened existing LanceDB table: {table_name} at {db_uri}")
    except FileNotFoundError: # LanceDB raises FileNotFoundError if table doesn't exist
        print(f"LanceDB table '{table_name}' not found. Will be created if new embeddings are generated.")
    except Exception as e: # Catch other potential lancedb errors during open
        print(f"Error opening LanceDB table '{table_name}': {e}. Proceeding as if table needs creation.")

    embedding_dim = model.config.hidden_size # Get embedding dimension

    # Image Scanning and Embedding Update
    paths_for_computation = []
    pils_for_computation = []
    # Dict to store all data: path -> {'mtime': mtime, 'embedding': tensor, 'basename': basename}
    processed_image_data = {} 

    print(f"\nProcessing {len(image_paths)} images (checking LanceDB, loading, and preparing for embedding)...")
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        try:
            current_mtime = os.path.getmtime(img_path)
        except FileNotFoundError:
            print(f"Warning: File {basename} not found during mtime check. Skipping.")
            continue
        
        found_in_db = False
        if tbl:
            try:
                # Ensure basename is properly escaped for SQL-like WHERE clause if it could contain quotes,
                # though unlikely for typical file basenames. LanceDB might handle this.
                query = f"image_path = '{basename}'" 
                results = tbl.search().where(query).limit(1).to_list()

                if results:
                    db_mtime = results[0]['mtime']
                    if db_mtime == current_mtime:
                        print(f"SUCCESS_CACHE_HIT: Using embedding from LanceDB for {basename} (mtime match: DB={db_mtime}, File={current_mtime})")
                        embedding = torch.tensor(results[0]['embedding']).cpu()
                        processed_image_data[img_path] = {'mtime': current_mtime, 'embedding': embedding, 'basename': basename}
                        found_in_db = True
                    else:
                        print(f"INFO_CACHE_MISS: Mtime mismatch for {basename}. DB_mtime={db_mtime}, File_mtime={current_mtime}. Will recompute.")
                else:
                    print(f"INFO_CACHE_MISS: {basename} not found in LanceDB table '{table_name}'. Will compute.")
            except Exception as e:
                print(f"WARNING_CACHE_QUERY_ERROR: Error querying LanceDB for {basename}: {e}. Will attempt to recompute.")
        else: # tbl is None (table does not exist)
            print(f"INFO_CACHE_MISS: LanceDB table '{table_name}' does not exist. Will compute for {basename}.")

        if not found_in_db:
            try:
                pil_image = load_image(img_path)
                paths_for_computation.append(img_path)
                pils_for_computation.append(pil_image)
            except IOError as e: # UnidentifiedImageError is a subclass of IOError
                print(f"WARNING_IMAGE_LOAD_ERROR: Skipping image {basename} due to loading error: {e}")
            except Exception as e:
                print(f"WARNING_UNEXPECTED_LOAD_ERROR: Unexpected error loading {basename}: {e}. Skipping.")


    # Batch Computation & LanceDB Update
    if pils_for_computation:
        print(f"\nPreprocessing {len(pils_for_computation)} images and generating new embeddings...")
        try:
            batched_pixel_values = preprocess_image_batch(pils_for_computation, processor)
            computed_embeddings_batch = get_image_embeddings(batched_pixel_values, model, device)
            
            data_to_add_to_lancedb = []
            for idx, img_path_comp in enumerate(paths_for_computation):
                basename_comp = os.path.basename(img_path_comp)
                embedding_tensor = computed_embeddings_batch[idx].cpu()
                try:
                    mtime_comp = os.path.getmtime(img_path_comp)
                    processed_image_data[img_path_comp] = {'mtime': mtime_comp, 'embedding': embedding_tensor, 'basename': basename_comp}
                    data_to_add_to_lancedb.append({
                        'image_path': basename_comp, 
                        'mtime': mtime_comp, 
                        'embedding': embedding_tensor.tolist() # Convert tensor to list for LanceDB
                    })
                except FileNotFoundError:
                     print(f"Warning: File {basename_comp} not found during mtime re-check for LanceDB add. Skipping this file.")


            if data_to_add_to_lancedb:
                if tbl is None:
                    schema = pa.schema([
                        pa.field("image_path", pa.string()),
                        pa.field("mtime", pa.float64()),
                        pa.field("embedding", pa.list_(pa.float32(), list_size=embedding_dim))
                    ])
                    try:
                        tbl = db.create_table(table_name, schema=schema)
                        print(f"Created LanceDB table: {table_name}")
                    except Exception as e:
                        print(f"Failed to create table {table_name} (it might already exist or schema mismatch): {e}")
                        try:
                            tbl = db.open_table(table_name) # Try opening again
                        except Exception as e_open:
                            print(f"Fatal: Could not create or open LanceDB table {table_name}: {e_open}")
                            return
                
                if tbl: # Ensure table is usable
                    try:
                        # Delete old entries for images being updated
                        for item_to_add in data_to_add_to_lancedb:
                            tbl.delete(f"image_path = '{item_to_add['image_path']}'")
                        
                        tbl.add(data_to_add_to_lancedb)
                        print(f"Added/updated {len(data_to_add_to_lancedb)} embeddings in LanceDB.")
                    except Exception as e:
                        print(f"Error updating LanceDB table {table_name}: {e}")

        except RuntimeError as e:
            print(f"Error during batch processing or embedding generation for new images: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during batch processing for new images: {e}")
    else:
        print("\nNo new embeddings to compute. All usable images were found in LanceDB or processed.")

    # Prepare for Comparison
    valid_image_paths_for_comparison = list(processed_image_data.keys())
    if len(valid_image_paths_for_comparison) < 2:
        print(f"\nNot enough valid embeddings ({len(valid_image_paths_for_comparison)}) available for comparison. Exiting.")
        return
    
    all_embeddings_list = [processed_image_data[path]['embedding'] for path in valid_image_paths_for_comparison]
    all_embeddings = torch.stack(all_embeddings_list)

    # Similarity Calculation Loop
    similar_pairs_found = 0
    similar_pairs_data = [] # For storing output
    print("\nCalculating similarities and comparing pairs...")
    for i, j in itertools.combinations(range(len(valid_image_paths_for_comparison)), 2):
        img_path1 = valid_image_paths_for_comparison[i]
        img_path2 = valid_image_paths_for_comparison[j]
        
        basename1 = processed_image_data[img_path1]['basename']
        basename2 = processed_image_data[img_path2]['basename']
        
        embedding1 = all_embeddings[i].unsqueeze(0) 
        embedding2 = all_embeddings[j].unsqueeze(0)

        try:
            similarity_score = calculate_similarity(embedding1, embedding2)
            if similarity_score >= args.threshold:
                print(f"Found similar pair: {basename1} and {basename2} - Similarity: {similarity_score:.4f}")
                similar_pairs_data.append({'image1': basename1, 'image2': basename2, 'similarity': round(similarity_score, 4)})
                similar_pairs_found += 1
        except RuntimeError as e:
            print(f"Error calculating similarity for pair ({basename1}, {basename2}): {e}")
        except Exception as e:
            print(f"Unexpected error calculating similarity for pair ({basename1}, {basename2}): {e}")

    # Save Results to Output File
    if similar_pairs_data:
        output_file_path = args.output_file
        file_ext = os.path.splitext(output_file_path)[1].lower()
        try:
            if file_ext == '.csv':
                with open(output_file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['image1', 'image2', 'similarity'])
                    writer.writeheader()
                    writer.writerows(similar_pairs_data)
            elif file_ext == '.json':
                with open(output_file_path, 'w') as f:
                    json.dump(similar_pairs_data, f, indent=4)
            else: # Default to .txt
                with open(output_file_path, 'w') as f:
                    for pair in similar_pairs_data:
                        f.write(f"Image 1: {pair['image1']}, Image 2: {pair['image2']}, Similarity: {pair['similarity']}\n")
            print(f"\nSaved {len(similar_pairs_data)} similar pairs to {output_file_path}")
        except IOError as e:
            print(f"Error writing output file {output_file_path}: {e}")

    elif similar_pairs_found == 0 : # Check if any pairs were printed but not added to data (e.g. if data append was skipped)
        print("\nNo pairs found above the similarity threshold.")
    # If similar_pairs_data is empty and similar_pairs_found is also 0, it means no pairs were found.

if __name__ == '__main__':
    main()
