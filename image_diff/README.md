# Image Diff Tool

## Description
A command-line tool to find similar images within a folder. It compares images pairwise using vision transformer models and reports pairs exceeding a given similarity threshold.

## Features
*   Pairwise image comparison in a specified folder.
*   **NEW:** Optional face detection mode to compare detected faces within images instead of whole images.
*   Supports PNG and JPG/JPEG image formats.
*   Uses Hugging Face transformer models for feature extraction (for both whole images and detected faces).
*   Allows user selection of any compatible model from Hugging Face Hub or a local path.
*   Adjustable similarity threshold.
*   Persistent embedding storage using LanceDB within the target image folder (`<folder_path>/.lancedb/`). Embeddings for whole images and faces are stored separately to avoid conflicts, reducing re-computation on subsequent runs.
*   Outputs results to console and optionally to a file (CSV, JSON, or TXT).

## Requirements
*   Python 3.8+ and `uv` (recommended, for environment and package management).
    *   Dependencies like `lancedb` and `pyarrow` are managed through `pyproject.toml` and installed with `uv pip install .`.

## Installation

1.  **Install `uv` (if you haven't already):**
    Follow the official instructions at [astral.sh/docs/uv#installation](https://astral.sh/docs/uv#installation) or use pipx/pip:
    ```bash
    # Using pipx (recommended for CLI tools)
    pipx install uv
    # Or using pip
    pip install uv
    ```

2.  **Clone the repository (or download source):**
    ```bash
    git clone <repository_url> # Replace with actual URL
    cd image-diff-tool 
    ```
    *(Note: The directory name might be `image_diff` or `image-diff-tool`. The `pyproject.toml` uses `image-diff-tool` as the project name. Ensure you are in the project root directory).*

3.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create a virtual environment
    uv venv
    # Activate the environment (uv automatically detects it in the current directory)
    # On Windows: .\.venv\Scriptsctivate
    # On macOS/Linux: source .venv/bin/activate
    # Note: With uv, direct activation is often not needed if you prefix commands with `uv run`.

    # Install the package and its dependencies
    uv pip install .
    ```

## Usage

After installation, the tool will be available as `image-diff` in your environment.

```bash
image-diff <folder_path> [options]
```

### Arguments
*   `folder_path`: (Required) Path to the folder containing images.
*   `--threshold VALUE`: (Optional) Similarity threshold for reporting image pairs (a float between 0.0 and 1.0). Default: `0.9`.
*   `--model_name NAME_OR_PATH`: (Optional) Hugging Face model name or path to a local pre-trained model. Default: `'google/vit-base-patch16-224-in21k'`.
*   `--output_file FILE_PATH`: (Optional) Path to save the list of similar image pairs. Output format is determined by the file extension:
    *   `.csv`: Comma-Separated Values (columns: `image1, image2, similarity`).
    *   `.json`: JSON array of objects.
    *   Other/No extension: Plain text.
    *   Default: `similar_pairs.csv`.
*   `--face_diff`: (Optional) If specified, enables face detection mode. The tool will attempt to detect faces in each image and compare these faces. If no face is detected in an image, the whole image might be used as a fallback (current behavior, with a warning logged).

### Examples
1.  Compare whole images in a folder named `my_images` using default settings (output to `similar_pairs.csv`):
    ```bash
    image-diff ./my_images
    ```

2.  **NEW:** Compare detected faces in images in `my_images` and save to `face_results.json`:
    ```bash
    image-diff ./my_images --face_diff --output_file face_results.json
    ```

3.  Compare whole images with a custom threshold and save to a JSON file:
    ```bash
    image-diff ./my_images --threshold 0.85 --output_file results.json
    ```

4.  Compare detected faces using a different model and a specific threshold:
    ```bash
    image-diff ./my_images --face_diff --model_name 'facebook/dinov2-base' --threshold 0.92
    ```

5.  Compare whole images using a different model and save to a text file:
    ```bash
    image-diff ./my_images
    ```

2.  Compare images with a custom threshold and save to a JSON file:
    ```bash
    image-diff ./my_images --threshold 0.85 --output_file results.json
    ```

3.  Compare images using a different model and save to a text file:
    ```bash
    image-diff ./my_images --model_name 'facebook/dinov2-base' --output_file findings.txt
    ```

## How it Works
1.  The tool scans the target folder (specified by `folder_path`) for images with `.png`, `.jpg`, or `.jpeg` extensions.
2.  **Operating Mode:**
    *   **Default (Image Diff):** The entire image is processed.
    *   **Face Diff (`--face_diff`):** The tool first attempts to detect faces using an MTCNN model. If faces are found, the largest detected face (current strategy, may evolve) is used for embedding. If no faces are detected in an image, the tool falls back to using the whole image for that specific file, and a warning is logged.
3.  For each image (or detected face):
    *   Its modification time (of the original file) is checked.
    *   If a valid embedding for the current vision model and operating mode (image vs. face) exists in the local LanceDB store (`<folder_path>/.lancedb/`) and the image hasn't changed, the stored embedding is used. LanceDB tables are named differently for image embeddings vs. face embeddings to prevent conflicts (e.g., `image_embeddings_<model_hash>` vs. `face_embeddings_<model_hash>`).
4.  Otherwise, if the image (or face) is new or the original file is modified:
    *   It's loaded (and cropped if in face mode and a face is found).
    *   It's preprocessed.
    *   Its feature embedding is computed using the selected vision transformer model.
5.  Newly computed embeddings are stored or updated in the appropriate model-specific and mode-specific LanceDB table.
6.  Once all necessary embeddings are retrieved or computed, the tool iterates through all unique pairs of valid items (images or faces). For each pair:
    a.  The cosine similarity is calculated between their embeddings. This score indicates how similar they are in terms of the features learned by the vision model.
    b.  If the calculated similarity score is greater than or equal to the specified `--threshold`, the pair of original image filenames is reported.
7.  Results (similar pairs and their scores) are printed to the console and saved to the file specified by `--output_file` (defaulting to `similar_pairs.csv`).

## Note on Models

### Vision Transformer Models
The default model for feature extraction is `google/vit-base-patch16-224-in21k`. You can use other models from the Hugging Face Hub that are suitable for image feature extraction (e.g., other Vision Transformer (ViT) variants, DeiT, DINOv2, etc.). Ensure the chosen model is compatible with `AutoModel` and `AutoImageProcessor` from the `transformers` library.

The LanceDB storage for embeddings is specific to both the chosen vision model and the operating mode (whole image vs. face). This means embeddings generated by different models or for different modes are stored separately, preventing conflicts and ensuring correct caching.

When you use a new vision model name from Hugging Face Hub for the first time, the tool will download its weights and configuration. This may take some time depending on the model size and your internet connection. Subsequent uses of the same model will load it from the local Hugging Face cache.

### Face Detection Model
When `--face_diff` mode is active, the tool uses an MTCNN (Multi-task Cascaded Convolutional Networks) model for face detection, provided by the `facenet-pytorch` library (which is a port of MTCNN). This model is generally effective for detecting faces under various conditions. The first time you run with `--face_diff`, the MTCNN model weights might also be downloaded if not already cached by `facenet-pytorch`.

### Performance Considerations & Model Choice

The default model (`google/vit-base-patch16-224-in21k`) provides a good balance of accuracy and performance. However, if you need faster processing, especially for a large number of images, consider the following:

*   **Patch Size:** Vision Transformer (ViT) models with larger patch sizes (e.g., `google/vit-base-patch32-224`) can sometimes be faster as they result in shorter sequence lengths for the transformer, at a potential slight cost to fine-grained accuracy.
*   **Smaller Architectures:** Models specifically designed for efficiency, such as MobileViT (e.g., `apple/mobilevit-small`) or other lightweight architectures available on Hugging Face Hub, can offer significant speed improvements. These models might have different embedding characteristics, so the optimal similarity threshold could vary.
*   **Hardware:** Using a GPU (`cuda` device) will dramatically speed up model inference compared to a CPU. The tool will automatically try to use a GPU if PyTorch detects one.
*   **Experimentation:** The best model depends on your specific dataset and requirements for speed versus similarity precision. You might need to experiment with a few options and adjust the `--threshold` accordingly.

When choosing a different model, ensure it's compatible with `AutoModel` and `AutoImageProcessor` for image feature extraction tasks.

### Running Tests

To run the unit tests:
```bash
# Ensure development dependencies (if any were defined as such) are installed
# For this project, tests run with the main dependencies
uv run python -m unittest discover tests -p 'test_*.py'
```
