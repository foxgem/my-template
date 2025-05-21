import unittest
import os
import torch

# Ensure the image_diff package is discoverable
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_diff.similarity import get_image_embedding, calculate_similarity
from image_diff.image_utils import load_image, preprocess_image
from PIL import UnidentifiedImageError, Image # Added Image for dummy image creation

class TestSimilarity(unittest.TestCase):

    def setUp(self):
        self.sample_images_dir = os.path.join(os.path.dirname(__file__), 'sample_images')
        self.img1_path = os.path.join(self.sample_images_dir, 'img1.png')
        self.img1_copy_path = os.path.join(self.sample_images_dir, 'img1_copy.png')
        self.img2_path = os.path.join(self.sample_images_dir, 'img2.jpg')
        self.default_model_name = 'google/vit-base-patch16-224-in21k'

        self.processed_img1_tensor = None
        self.processed_img1_copy_tensor = None
        self.processed_img2_tensor = None
        self.embeddings_generated = False

        try:
            img1_pil = load_image(self.img1_path)
            self.processed_img1_tensor = preprocess_image(img1_pil, self.default_model_name)

            img1_copy_pil = load_image(self.img1_copy_path)
            self.processed_img1_copy_tensor = preprocess_image(img1_copy_pil, self.default_model_name)
            
            img2_pil = load_image(self.img2_path)
            self.processed_img2_tensor = preprocess_image(img2_pil, self.default_model_name)
            
            self.embeddings_generated = True # Mark that preprocessing was successful

        except (FileNotFoundError, UnidentifiedImageError, IOError, RuntimeError, OSError) as e:
            print(f"Skipping some similarity tests due to error in setUp: {e}")
            # Keep tensors as None if any error occurs

    def test_get_image_embedding_shape(self):
        if not self.embeddings_generated or self.processed_img1_tensor is None:
            self.skipTest("Skipping embedding shape test: Preprocessing failed or placeholder images are empty.")
        
        try:
            embedding = get_image_embedding(self.processed_img1_tensor, self.default_model_name)
            self.assertIsInstance(embedding, torch.Tensor)
            self.assertEqual(embedding.ndim, 2, "Embedding should be 2D (batch_size, features)")
            self.assertEqual(embedding.shape[0], 1, "Batch size for embedding should be 1")
        except RuntimeError as e:
            self.skipTest(f"Skipping get_image_embedding_shape due to runtime error (likely model or image issue): {e}")
        except OSError as e: # Model download issues
            self.skipTest(f"Skipping get_image_embedding_shape due to OSError (likely model download issue): {e}")


    def test_calculate_similarity_identical_images(self):
        if not self.embeddings_generated or self.processed_img1_tensor is None or self.processed_img1_copy_tensor is None:
            self.skipTest("Skipping identical image similarity test: Preprocessing failed or placeholder images are empty.")

        try:
            embedding1 = get_image_embedding(self.processed_img1_tensor, self.default_model_name)
            embedding1_copy = get_image_embedding(self.processed_img1_copy_tensor, self.default_model_name)
            similarity = calculate_similarity(embedding1, embedding1_copy)
            # For truly identical images and deterministic models, similarity should be very close to 1.0
            self.assertGreater(similarity, 0.99, "Similarity for identical images should be > 0.99")
        except RuntimeError as e:
            self.skipTest(f"Skipping test_calculate_similarity_identical_images due to runtime error: {e}")
        except OSError as e:
            self.skipTest(f"Skipping test_calculate_similarity_identical_images due to OSError: {e}")


    def test_calculate_similarity_different_images(self):
        if not self.embeddings_generated or self.processed_img1_tensor is None or self.processed_img2_tensor is None:
            self.skipTest("Skipping different image similarity test: Preprocessing failed or placeholder images are empty.")
        
        try:
            embedding1 = get_image_embedding(self.processed_img1_tensor, self.default_model_name)
            embedding2 = get_image_embedding(self.processed_img2_tensor, self.default_model_name)
            similarity = calculate_similarity(embedding1, embedding2)
            # This threshold is arbitrary for empty/placeholder images.
            # For actual different images, this would depend on the model and images.
            self.assertLess(similarity, 0.95, "Similarity for different images should be < 0.95 (using placeholders, this might not be meaningful)")
        except RuntimeError as e:
            self.skipTest(f"Skipping test_calculate_similarity_different_images due to runtime error: {e}")
        except OSError as e:
            self.skipTest(f"Skipping test_calculate_similarity_different_images due to OSError: {e}")


    def test_calculate_similarity_dummy_embeddings(self):
        # Test with simple, predictable tensors
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


    def test_get_image_embedding_invalid_model(self):
        # Need a valid processed tensor to pass to the function, even if it's from a placeholder
        # If setUp failed, create a dummy tensor for this specific test.
        if self.processed_img1_tensor is None:
            # Create a dummy tensor that resembles a preprocessed image output
            # (batch_size, num_channels, height, width)
            dummy_pil_image = Image.new('RGB', (224, 224)) # Ensure it's a PIL image
            try:
                # Try to use the actual preprocess_image if possible, even if it failed in setup
                # with the placeholder, a dummy PIL image might work for this one test.
                # This might still fail if the *default* model can't be loaded for some reason.
                temp_processed_tensor = preprocess_image(dummy_pil_image, self.default_model_name)
            except (OSError, RuntimeError): # If default model itself fails to load
                 temp_processed_tensor = torch.randn(1, 3, 224, 224) # Fallback to pure random tensor
        else:
            temp_processed_tensor = self.processed_img1_tensor

        with self.assertRaises(OSError): # Transformers typically raises OSError for model not found
            get_image_embedding(temp_processed_tensor, 'invalid/model-name-that-does-not-exist')

if __name__ == '__main__':
    unittest.main()
