import unittest
import os
import torch

# Ensure the image_diff package is discoverable
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock
import transformers # For MagicMock spec

from image_diff.similarity import get_image_embeddings, calculate_similarity
# from image_diff.image_utils import load_image, preprocess_image # No longer directly used for setup
# from PIL import UnidentifiedImageError, Image # No longer directly used for setup

class TestSimilarity(unittest.TestCase):

    def setUp(self):
        # Mock AutoModel
        self.mock_model = MagicMock(spec=transformers.AutoModel)
        self.device = torch.device('cpu')
        self.embedding_dim = 768 # Example embedding dimension

    def test_get_image_embeddings_shape_single(self):
        # Test with a single "processed image" tensor
        dummy_pixel_values_single = torch.randn(1, 3, 224, 224) # Batch of 1
        
        # Configure mock_model to return a structure with pooler_output
        # Model output for a batch of 1, embedding dim 768
        expected_embedding = torch.randn(1, self.embedding_dim)
        self.mock_model.return_value = MagicMock(pooler_output=expected_embedding, last_hidden_state=None)

        embedding = get_image_embeddings(dummy_pixel_values_single, self.mock_model, self.device)
        
        self.mock_model.assert_called_once_with(dummy_pixel_values_single.to(self.device))
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.ndim, 2, "Embedding should be 2D (batch_size, features)")
        self.assertEqual(embedding.shape[0], 1, "Batch size for embedding should be 1")
        self.assertEqual(embedding.shape[1], self.embedding_dim)
        self.assertTrue(torch.equal(embedding, expected_embedding.cpu()))

    def test_get_image_embeddings_shape_batch(self):
        batch_size = 3
        dummy_pixel_values_batch = torch.randn(batch_size, 3, 224, 224)
        
        expected_embeddings_batch = torch.randn(batch_size, self.embedding_dim)
        self.mock_model.return_value = MagicMock(pooler_output=expected_embeddings_batch, last_hidden_state=None)

        embeddings = get_image_embeddings(dummy_pixel_values_batch, self.mock_model, self.device)

        self.mock_model.assert_called_once_with(dummy_pixel_values_batch.to(self.device))
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.ndim, 2)
        self.assertEqual(embeddings.shape[0], batch_size)
        self.assertEqual(embeddings.shape[1], self.embedding_dim)
        self.assertTrue(torch.equal(embeddings, expected_embeddings_batch.cpu()))

    def test_get_image_embeddings_uses_last_hidden_state_if_no_pooler(self):
        batch_size = 2
        sequence_length = 50 # Example sequence length for ViT
        feature_dim = self.embedding_dim 
        dummy_pixel_values_batch = torch.randn(batch_size, 3, 224, 224)
        
        # Simulate model output without pooler_output
        mock_last_hidden_state = torch.randn(batch_size, sequence_length, feature_dim)
        self.mock_model.return_value = MagicMock(pooler_output=None, last_hidden_state=mock_last_hidden_state)
        
        # Expected CLS token embeddings (first token of each item in batch)
        expected_cls_embeddings = mock_last_hidden_state[:, 0, :].cpu()

        embeddings = get_image_embeddings(dummy_pixel_values_batch, self.mock_model, self.device)

        self.mock_model.assert_called_once_with(dummy_pixel_values_batch.to(self.device))
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], batch_size)
        self.assertEqual(embeddings.shape[1], feature_dim)
        self.assertTrue(torch.equal(embeddings, expected_cls_embeddings))


    def test_calculate_similarity_identical_embeddings(self):
        # Test with identical dummy embeddings
        embedding1 = torch.randn(1, self.embedding_dim)
        embedding1_copy = embedding1.clone() # Perfect copy
        
        similarity = calculate_similarity(embedding1, embedding1_copy)
        self.assertGreater(similarity, 0.999, "Similarity for identical embeddings should be very close to 1.0")

    def test_calculate_similarity_different_embeddings(self):
        # Test with different dummy embeddings
        embedding1 = torch.randn(1, self.embedding_dim)
        embedding2 = torch.randn(1, self.embedding_dim)
        # Ensure they are not identical by chance (highly unlikely for high dim random tensors)
        while torch.equal(embedding1, embedding2):
            embedding2 = torch.randn(1, self.embedding_dim)

        similarity = calculate_similarity(embedding1, embedding2)
        # For random embeddings, similarity should generally be low, but can vary.
        # We are checking that the function runs and returns a float.
        # A specific range for truly "different" random embeddings is hard to assert without statistical analysis.
        # For now, just check it's not extremely high (like identical)
        self.assertLess(similarity, 0.95, "Similarity for different random embeddings should generally be less than 0.95")


    def test_calculate_similarity_dummy_values(self):
        # Test with simple, predictable tensors (original test_calculate_similarity_dummy_embeddings)
        embedding_a = torch.tensor([[1.0, 0.0, 0.0]])
        embedding_b = torch.tensor([[1.0, 0.0, 0.0]]) # Identical
        embedding_c = torch.tensor([[0.0, 1.0, 0.0]]) # Orthogonal
        embedding_d = torch.tensor([[-1.0, 0.0, 0.0]]) # Opposite

        similarity_ab = calculate_similarity(embedding_a, embedding_b)
        self.assertAlmostEqual(similarity_ab, 1.0, places=5)

        similarity_ac = calculate_similarity(embedding_a, embedding_c)
        self.assertAlmostEqual(similarity_ac, 0.0, places=5)

        similarity_ad = calculate_similarity(embedding_a, embedding_d)
        self.assertAlmostEqual(similarity_ad, -1.0, places=5)
        
        # Test with non-unit vectors
        embedding_e = torch.tensor([[2.0, 2.0]])
        embedding_f = torch.tensor([[4.0, 4.0]]) # Same direction, different magnitude
        similarity_ef = calculate_similarity(embedding_e, embedding_f)
        self.assertAlmostEqual(similarity_ef, 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
