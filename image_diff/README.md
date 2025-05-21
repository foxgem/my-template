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
*   Python 3.7+
*   Libraries listed in `requirements.txt`. These can be installed using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Installation
1.  Clone the repository or download and extract the source code.
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd image_diff
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note on PyTorch Installation:** The `requirements.txt` file includes `torch`. If you need a specific version of PyTorch (e.g., with a particular CUDA toolkit support), you might want to install it separately first. For example:
        ```bash
        # Example for PyTorch with CUDA 11.8 support
        # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        # Then run: pip install -r requirements.txt
        ```
        Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for more options.

## Usage
The tool is run from the command line using the `image_diff` module.

Basic command structure:
```bash
python -m image_diff <folder_path> [options]
```

### Arguments
*   `folder_path`: (Required) Path to the folder containing images.
*   `--threshold VALUE`: (Optional) Similarity threshold for reporting image pairs (a float between 0.0 and 1.0). Default: `0.9`.
*   `--model_name NAME_OR_PATH`: (Optional) Hugging Face model name or path to a local pre-trained model. Default: `'google/vit-base-patch16-224-in21k'`.

### Examples
1.  Compare images in a folder named `my_images` using default settings:
    ```bash
    python -m image_diff ./my_images
    ```

2.  Compare images in `my_images` with a custom similarity threshold of `0.85`:
    ```bash
    python -m image_diff ./my_images --threshold 0.85
    ```

3.  Compare images in `my_images` using a different model, for example, `facebook/dinov2-base`:
    ```bash
    python -m image_diff ./my_images --model_name 'facebook/dinov2-base'
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
