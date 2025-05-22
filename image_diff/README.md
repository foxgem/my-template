# Image Diff Tool

## Description
A command-line tool to find similar images within a folder. It compares images pairwise using vision transformer models and reports pairs exceeding a given similarity threshold.

## Features
*   Pairwise image comparison in a specified folder.
*   Supports PNG and JPG/JPEG image formats.
*   Uses Hugging Face transformer models for feature extraction.
*   Allows user selection of any compatible model from Hugging Face Hub or a local path.
*   Adjustable similarity threshold.

## Requirements
*   Python 3.8+ and `uv` (recommended, for environment and package management).

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

### Examples
1.  Compare images in a folder named `my_images` using default settings:
    ```bash
    image-diff ./my_images
    ```

2.  Compare images in `my_images` with a custom similarity threshold of `0.85`:
    ```bash
    image-diff ./my_images --threshold 0.85
    ```

3.  Compare images in `my_images` using a different model, for example, `facebook/dinov2-base`:
    ```bash
    image-diff ./my_images --model_name 'facebook/dinov2-base'
    ```

## How it Works
1.  The tool scans the target folder (specified by `folder_path`) for images with `.png`, `.jpg`, or `.jpeg` extensions.
2.  It then iterates through all unique pairs of these images. For each pair:
    a.  The images are loaded and preprocessed to be compatible with the selected vision model.
    b.  A pre-trained vision transformer model (specified by `--model_name`) is used to extract feature embeddings (numerical representations) from each image.
    c.  The cosine similarity is calculated between the two image embeddings. This score indicates how similar the images are in terms of the features learned by the model.
3.  If the calculated similarity score is greater than or equal to the specified `--threshold`, the tool prints the names of the two images and their similarity score.

## Note on Models
The default model is `google/vit-base-patch16-224-in21k`. You can use other models from the Hugging Face Hub that are suitable for image feature extraction (e.g., other Vision Transformer (ViT) variants, DeiT, DINOv2, etc.). Ensure the chosen model is compatible with `AutoModel` and `AutoImageProcessor` from the `transformers` library for image feature extraction tasks.

When you use a new model name from Hugging Face Hub for the first time, the tool will download the model weights and configuration. This may take some time depending on the model size and your internet connection. Subsequent uses of the same model will load it from the local cache.

### Running Tests

To run the unit tests:
```bash
# Ensure development dependencies (if any were defined as such) are installed
# For this project, tests run with the main dependencies
uv run python -m unittest discover tests -p 'test_*.py'
```
