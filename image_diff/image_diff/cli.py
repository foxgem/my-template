import argparse
import os
import itertools
import logging # Added
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

    logging_group = parser.add_argument_group('Logging Options')
    logging_group.add_argument(
        "--log-file",
        type=str,
        default=None, # Default will be handled to place it in args.folder_path later
        help="Path to save the log file. Defaults to '.image_diff.log' in the target image folder."
    )
    logging_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for console output. Default: INFO."
    )
    args = parser.parse_args()

    # Setup logging
    logger = logging.getLogger("image_diff_cli")
    logger.setLevel(logging.DEBUG) # Set logger to lowest level to capture all messages

    # Console Handler
    ch = logging.StreamHandler() # Defaults to sys.stderr
    console_log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    ch.setLevel(console_log_level)
    ch_formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    # Ensure folder_path exists before trying to create log file in it
    # This validation is also done later, but good to have it before log setup if using folder_path for log file
    if not os.path.exists(args.folder_path):
        logger.error(f"Folder not found: {args.folder_path}") # Logger might only have console here if file path is bad
        return
    if not os.path.isdir(args.folder_path):
        logger.error(f"Provided path is not a directory: {args.folder_path}")
        return

    # Determine log file path
    if args.log_file:
        log_file_path = args.log_file
    else:
        log_file_path = os.path.join(args.folder_path, ".image_diff.log")

    # File Handler (always DEBUG level)
    try:
        fh = logging.FileHandler(log_file_path, mode='w') # Overwrite log each time
        fh.setLevel(logging.DEBUG) 
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
        logger.info(f"Detailed logs will be written to: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to set up file logging to {log_file_path}: {e}")
        # Continue with console logging only

    # Image Discovery
    supported_extensions = ('.png', '.jpg', '.jpeg')
    image_paths = []
    for item in os.listdir(args.folder_path):
        if item.lower().endswith(supported_extensions):
            image_paths.append(os.path.join(args.folder_path, item))

    if len(image_paths) < 2:
        logger.error(f"Found {len(image_paths)} image(s). Need at least two images to compare.")
        return
    
    logger.info(f"Found {len(image_paths)} images in '{args.folder_path}'.")

    # Model and Processor Loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    try:
        logger.info(f"Loading image processor for '{args.model_name}'...")
        processor = AutoImageProcessor.from_pretrained(args.model_name)
        logger.info(f"Loading model '{args.model_name}'...")
        model = AutoModel.from_pretrained(args.model_name)
        model.eval()
        model.to(device)
        logger.info("Model and processor loaded successfully and moved to device.")
    except Exception as e: # Catch broad errors from transformers loading
        logger.error(f"Error loading model or processor '{args.model_name}': {e}. Check model name or network connection.")
        return

    # LanceDB Initialization
    # LanceDB Initialization
    db_uri = os.path.join(args.folder_path, ".lancedb")
    db = lancedb.connect(db_uri)
    model_hash = hashlib.md5(args.model_name.encode()).hexdigest()[:12]
    table_name = f"image_embeddings_{model_hash}"
    logger.info(f"Using LanceDB URI: {db_uri}") # Early logging
    logger.info(f"Target LanceDB table name: {table_name}") # Early logging

    try:
        existing_tables = db.table_names()
        logger.debug(f"Existing tables in LanceDB at {db_uri}: {existing_tables}")
    except Exception as e:
        logger.warning(f"Could not list LanceDB tables at {db_uri}: {e}")

    tbl = None
    try:
        tbl = db.open_table(table_name)
        logger.info(f"SUCCESS_LANCEDB_OPEN: Opened existing LanceDB table: {table_name}")
    except FileNotFoundError: # LanceDB raises FileNotFoundError if table doesn't exist
        logger.info(f"INFO_LANCEDB_NOT_FOUND: LanceDB table '{table_name}' not found by open_table(). Will attempt to create it if new embeddings are generated.")
    except Exception as e: # Catch other potential lancedb errors during open
        logger.warning(f"WARNING_LANCEDB_OPEN_ERROR: Error opening LanceDB table '{table_name}': {e}. Proceeding as if table needs creation.")

    embedding_dim = model.config.hidden_size # Get embedding dimension

    # Image Scanning and Embedding Update
    paths_for_computation = []
    pils_for_computation = []
    # Dict to store all data: path -> {'mtime': mtime, 'embedding': tensor, 'basename': basename}
    processed_image_data = {} 

    logger.info(f"Processing {len(image_paths)} images (checking LanceDB, loading, and preparing for embedding)...")
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        try:
            current_mtime = os.path.getmtime(img_path)
        except FileNotFoundError:
            logger.warning(f"File {basename} not found during mtime check. Skipping.")
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
                        logger.debug(f"SUCCESS_CACHE_HIT: Using embedding from LanceDB for {basename} (mtime match: DB={db_mtime}, File={current_mtime})")
                        embedding = torch.tensor(results[0]['embedding']).cpu()
                        processed_image_data[img_path] = {'mtime': current_mtime, 'embedding': embedding, 'basename': basename}
                        found_in_db = True
                    else:
                        logger.debug(f"INFO_CACHE_MISS: Mtime mismatch for {basename}. DB_mtime={db_mtime}, File_mtime={current_mtime}. Will recompute.")
                else:
                    logger.debug(f"INFO_CACHE_MISS: {basename} not found in LanceDB table '{table_name}'. Will compute.")
            except Exception as e:
                logger.warning(f"WARNING_CACHE_QUERY_ERROR: Error querying LanceDB for {basename}: {e}. Will attempt to recompute.")
        else: # tbl is None (table does not exist)
            logger.debug(f"INFO_CACHE_MISS: LanceDB table '{table_name}' does not exist. Will compute for {basename}.")

        if not found_in_db:
            try:
                pil_image = load_image(img_path)
                paths_for_computation.append(img_path)
                pils_for_computation.append(pil_image)
            except IOError as e: # UnidentifiedImageError is a subclass of IOError
                logger.warning(f"WARNING_IMAGE_LOAD_ERROR: Skipping image {basename} due to loading error: {e}")
            except Exception as e:
                logger.warning(f"WARNING_UNEXPECTED_LOAD_ERROR: Unexpected error loading {basename}: {e}. Skipping.")


    # Batch Computation & LanceDB Update
    if pils_for_computation:
        logger.info(f"Preprocessing {len(pils_for_computation)} images and generating new embeddings...")
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
                     logger.warning(f"File {basename_comp} not found during mtime re-check for LanceDB add. Skipping this file.")


            if data_to_add_to_lancedb:
                if tbl is None:
                    schema = pa.schema([
                        pa.field("image_path", pa.string()),
                        pa.field("mtime", pa.float64()),
                        pa.field("embedding", pa.list_(pa.float32(), list_size=embedding_dim))
                    ])
                    try:
                        logger.info(f"Attempting to create new LanceDB table '{table_name}' with {len(data_to_add_to_lancedb)} records.")
                        tbl = db.create_table(table_name, schema=schema)
                        logger.info(f"SUCCESS_LANCEDB_CREATE: Created LanceDB table: {table_name}")
                    except Exception as e:
                        logger.warning(f"WARNING_LANCEDB_CREATE_ERROR: Failed to create table {table_name} (it might already exist or schema mismatch): {e}")
                        try:
                            tbl = db.open_table(table_name) # Try opening again
                            logger.info(f"INFO_LANCEDB_REOPEN_SUCCESS: Successfully opened table '{table_name}' after create attempt failed.")
                        except Exception as e_open:
                            logger.error(f"FATAL_LANCEDB_REOPEN_ERROR: Could not create or open LanceDB table {table_name}: {e_open}")
                            return
                
                if tbl: # Ensure table is usable
                    try:
                        # Delete old entries for images being updated
                        for item_to_add in data_to_add_to_lancedb:
                            tbl.delete(f"image_path = '{item_to_add['image_path']}'")
                        
                        tbl.add(data_to_add_to_lancedb)
                        logger.info(f"Added/updated {len(data_to_add_to_lancedb)} embeddings in LanceDB.")
                    except Exception as e:
                        logger.error(f"Error updating LanceDB table {table_name}: {e}")

        except RuntimeError as e:
            logger.error(f"Error during batch processing or embedding generation for new images: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during batch processing for new images: {e}")
    else:
        logger.info("No new embeddings to compute. All usable images were found in LanceDB or processed.")

    # Prepare for Comparison
    valid_image_paths_for_comparison = list(processed_image_data.keys())
    if len(valid_image_paths_for_comparison) < 2:
        logger.error(f"Not enough valid embeddings ({len(valid_image_paths_for_comparison)}) available for comparison. Exiting.")
        return
    
    all_embeddings_list = [processed_image_data[path]['embedding'] for path in valid_image_paths_for_comparison]
    all_embeddings = torch.stack(all_embeddings_list)

    # Similarity Calculation Loop
    similar_pairs_found = 0
    similar_pairs_data = [] # For storing output
    logger.info("Calculating similarities and comparing pairs...")
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
                logger.info(f"Found similar pair: {basename1} and {basename2} - Similarity: {similarity_score:.4f}")
                similar_pairs_data.append({'image1': basename1, 'image2': basename2, 'similarity': round(similarity_score, 4)})
                similar_pairs_found += 1
        except RuntimeError as e:
            logger.warning(f"Error calculating similarity for pair ({basename1}, {basename2}): {e}")
        except Exception as e:
            logger.error(f"Unexpected error calculating similarity for pair ({basename1}, {basename2}): {e}")

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
            logger.info(f"Saved {len(similar_pairs_data)} similar pairs to {output_file_path}")
        except IOError as e:
            logger.error(f"Error writing output file {output_file_path}: {e}")

    elif similar_pairs_found == 0 : 
        logger.info("No pairs found above the similarity threshold.")
    # If similar_pairs_data is empty and similar_pairs_found is also 0, it means no pairs were found.

if __name__ == '__main__':
    main()
