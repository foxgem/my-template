import torch
from transformers import AutoModel
import torch.nn.functional as F

def get_image_embedding(processed_image_tensor, model_name_or_path: str):
    """
    Generates an embedding for a preprocessed image using a Hugging Face model.

    Args:
        processed_image_tensor: The tensor output from preprocess_image (pixel_values).
        model_name_or_path: The name or path of the Hugging Face model.

    Returns:
        A torch.Tensor representing the image embedding.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = AutoModel.from_pretrained(model_name_or_path)
        model.eval()
        model.to(device)

        processed_image_tensor = processed_image_tensor.to(device)

        with torch.no_grad():
            outputs = model(processed_image_tensor)

        # Try to get pooler_output, otherwise use last_hidden_state's CLS token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embedding = outputs.pooler_output
        else:
            embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding.cpu() # Return embedding on CPU

    except Exception as e:
        # Broad exception for now, can be refined
        raise RuntimeError(f"Error generating embedding with model {model_name_or_path}: {e}")

def calculate_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """
    Calculates the cosine similarity between two embedding tensors.

    Args:
        embedding1: The first embedding tensor.
        embedding2: The second embedding tensor.

    Returns:
        A float representing the cosine similarity, ranging from -1 to 1.
    """
    if not isinstance(embedding1, torch.Tensor) or not isinstance(embedding2, torch.Tensor):
        raise TypeError("Embeddings must be torch.Tensor objects.")
    
    if embedding1.ndim == 1: # Ensure embeddings are at least 2D for cosine_similarity (batch_size, features)
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.ndim == 1:
        embedding2 = embedding2.unsqueeze(0)

    try:
        # dim=-1 to compute similarity over the feature dimension
        similarity = F.cosine_similarity(embedding1, embedding2, dim=-1)
        return similarity.item()
    except Exception as e:
        raise RuntimeError(f"Error calculating cosine similarity: {e}")
