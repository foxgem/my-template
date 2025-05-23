import torch
# AutoModel import removed
import torch.nn.functional as F

def get_image_embeddings(batch_pixel_values, model, device):
    """
    Generates embeddings for a batch of preprocessed images using a pre-loaded Hugging Face model.

    Args:
        batch_pixel_values: A tensor of shape (batch_size, num_channels, height, width)
                            containing the preprocessed image data.
        model: An initialized and loaded Hugging Face AutoModel instance, already in eval()
               mode and on the correct device.
        device: The torch.device to which the batch_pixel_values should be moved.

    Returns:
        A torch.Tensor representing the batch of image embeddings, moved to CPU.
        Shape: (batch_size, embedding_dimension).
    """
    try:
        batch_pixel_values = batch_pixel_values.to(device)
        with torch.no_grad():
            outputs = model(batch_pixel_values)

        # Try to get pooler_output, otherwise use last_hidden_state's CLS token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # Assuming CLS token is at index 0 of the sequence for each item in the batch
            embeddings = outputs.last_hidden_state[:, 0, :] 
        
        return embeddings.cpu() # Return batch of embeddings on CPU

    except Exception as e:
        # Broad exception for now, can be refined
        raise RuntimeError(f"Error generating embeddings from batch: {e}")

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
