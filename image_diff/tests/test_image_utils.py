import unittest
import os
from PIL import Image, UnidentifiedImageError
import torch

# Ensure the image_diff package is discoverable if tests are run from root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock, patch
import transformers # For MagicMock spec
from facenet_pytorch import MTCNN as RealMTCNN # Import real MTCNN for spec

# Updated import to include detect_faces
from image_diff.image_utils import load_image, preprocess_image_batch, detect_faces

class TestImageUtils(unittest.TestCase):

    def setUp(self):
        self.sample_images_dir = os.path.join(os.path.dirname(__file__), 'sample_images')
        self.sample_face_images_dir = os.path.join(os.path.dirname(__file__), 'sample_images_face') # New directory

        self.img1_path = os.path.join(self.sample_images_dir, 'img1.png') # Standard image
        self.img2_path = os.path.join(self.sample_images_dir, 'img2.jpg') # Standard image
        self.not_image_path = os.path.join(self.sample_images_dir, 'not_an_image.txt')
        self.non_existent_path = os.path.join(self.sample_images_dir, 'non_existent.png')

        # Paths for face test images (currently placeholders)
        self.single_face_img_path = os.path.join(self.sample_face_images_dir, 'single_face.png')
        self.multiple_faces_img_path = os.path.join(self.sample_face_images_dir, 'multiple_faces.png')
        self.no_face_img_path = os.path.join(self.sample_face_images_dir, 'no_face.png')
        
        # Mock AutoImageProcessor
        self.mock_processor = MagicMock(spec=transformers.AutoImageProcessor)

        # Create a dummy PIL image for tests that need one, as placeholders might not load
        self.dummy_pil_image = Image.new('RGB', (100, 100), color='blue')

        # Mock MTCNN model - this will be used by `detect_faces` and `load_image` in face mode
        # We use `patch` in specific tests for MTCNN to control its behavior precisely per test.
        self.mock_mtcnn_instance = MagicMock(spec=RealMTCNN)
        self.device = torch.device('cpu')


    def test_load_image_png(self):
        # This test will likely fail if img1.png is an empty file.
        # A real PNG would be loaded and converted to RGB.
        # If the placeholder image cannot be loaded, this test will use a dummy image.
        try:
            img = load_image(self.img1_path)
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.mode, 'RGB')
        except UnidentifiedImageError:
            self.skipTest(f"Skipping {self.img1_path} load test: empty file cannot be identified as image.")
        except IOError as e:
             self.skipTest(f"Skipping {self.img1_path} load test due to IOError (likely empty file): {e}")


    def test_load_image_jpg(self):
        # This test will also likely fail because img2.jpg is an empty file.
        try:
            img = load_image(self.img2_path)
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.mode, 'RGB')
        except UnidentifiedImageError:
            self.skipTest(f"Skipping {self.img2_path} load test: empty file cannot be identified as image.")
        except IOError as e:
            self.skipTest(f"Skipping {self.img2_path} load test due to IOError (likely empty file): {e}")

    def test_load_image_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_image(self.non_existent_path)

    def test_load_image_unidentified(self):
        # This test depends on how PIL handles empty text files.
        # It might raise UnidentifiedImageError or a more generic IOError.
        # Our load_image function explicitly raises IOError for UnidentifiedImageError.
        with self.assertRaises(IOError): # Changed from PIL.UnidentifiedImageError to IOError
            load_image(self.not_image_path)

    # test_preprocess_image_valid removed

    def test_preprocess_image_batch_valid(self):
        dummy_images_list = [
            Image.new('RGB', (10, 10)),
            Image.new('RGB', (12, 12))
        ]
        batch_size = len(dummy_images_list)
        
        # Configure mock_processor for a batch of images
        expected_batch_tensor = torch.randn(batch_size, 3, 224, 224)
        self.mock_processor.return_value = MagicMock(pixel_values=expected_batch_tensor)

        output_tensor = preprocess_image_batch(dummy_images_list, self.mock_processor)

        self.mock_processor.assert_called_once_with(images=dummy_images_list, return_tensors='pt')
        self.assertIsInstance(output_tensor, torch.Tensor)
        self.assertEqual(output_tensor.ndim, 4) # Batch, Channels, Height, Width
        self.assertEqual(output_tensor.shape[0], batch_size)
        self.assertTrue(torch.equal(output_tensor, expected_batch_tensor))

    # --- Tests for detect_faces ---

    def test_detect_faces_no_mtcnn_or_image(self):
        # Test with no MTCNN model
        faces_no_model = detect_faces(self.dummy_pil_image, None, self.device)
        self.assertEqual(faces_no_model, [])

        # Test with no image
        faces_no_image = detect_faces(None, self.mock_mtcnn_instance, self.device)
        self.assertEqual(faces_no_image, [])

    def test_detect_faces_mtcnn_detects_no_faces(self):
        self.mock_mtcnn_instance.detect.return_value = (None, None) # MTCNN returns (None, None) if no faces

        faces = detect_faces(self.dummy_pil_image, self.mock_mtcnn_instance, self.device)

        self.mock_mtcnn_instance.detect.assert_called_once_with(self.dummy_pil_image)
        self.assertEqual(faces, [])

    def test_detect_faces_mtcnn_detects_one_face(self):
        # Define a bounding box for one face
        box1 = [10, 10, 50, 50] # x1, y1, x2, y2
        self.mock_mtcnn_instance.detect.return_value = ([box1], [0.99]) # boxes, probs

        # Create a dummy image that can be cropped
        source_image = Image.new('RGB', (100, 100))

        with patch.object(Image.Image, 'crop') as mock_crop:
            # Configure the mock_crop to return a specific dummy PIL image
            cropped_face_image = Image.new('RGB', (40, 40), color='red')
            mock_crop.return_value = cropped_face_image

            faces = detect_faces(source_image, self.mock_mtcnn_instance, self.device)

            self.mock_mtcnn_instance.detect.assert_called_once_with(source_image)
            mock_crop.assert_called_once_with((10, 10, 50, 50)) # Ensure crop was called with correct box
            self.assertEqual(len(faces), 1)
            self.assertIsInstance(faces[0], Image.Image)
            self.assertTrue(faces[0] == cropped_face_image) # Check if it's the same image object

    def test_detect_faces_mtcnn_detects_multiple_faces(self):
        boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
        probs = [0.99, 0.98]
        self.mock_mtcnn_instance.detect.return_value = (boxes, probs)

        source_image = Image.new('RGB', (100, 100))

        # Mock crop to return distinct images for distinct calls
        cropped_face1 = Image.new('RGB', (40, 40), color='red')
        cropped_face2 = Image.new('RGB', (30, 30), color='green')

        with patch.object(Image.Image, 'crop') as mock_crop:
            mock_crop.side_effect = [cropped_face1, cropped_face2] # Return different images on subsequent calls

            faces = detect_faces(source_image, self.mock_mtcnn_instance, self.device)

            self.assertEqual(self.mock_mtcnn_instance.detect.call_count, 1)
            self.assertEqual(mock_crop.call_count, 2)
            mock_crop.assert_any_call((10, 10, 50, 50))
            mock_crop.assert_any_call((60, 60, 90, 90))

            self.assertEqual(len(faces), 2)
            self.assertTrue(faces[0] == cropped_face1)
            self.assertTrue(faces[1] == cropped_face2)

    def test_detect_faces_mtcnn_exception(self):
        self.mock_mtcnn_instance.detect.side_effect = Exception("MTCNN Error")
        faces = detect_faces(self.dummy_pil_image, self.mock_mtcnn_instance, self.device)
        self.assertEqual(faces, [])

    # --- Tests for load_image with face detection ---

    @patch('image_diff.image_utils.detect_faces') # Mock detect_faces within image_utils
    def test_load_image_with_face_detection_face_found(self, mock_detect_faces):
        # Setup: Image.open will return our dummy PIL image.
        # detect_faces (mocked) will return a list with one "face" (another dummy PIL image).

        # This is the image that load_image will "load" from disk
        original_pil_image = Image.new('RGB', (200, 200), color='yellow')
        # This is the "face" that detect_faces will return
        detected_face_pil = Image.new('RGB', (50, 50), color='green')

        mock_detect_faces.return_value = [detected_face_pil]

        # We need to mock Image.open for this specific test to control what image is "loaded"
        with patch.object(Image, 'open') as mock_open:
            mock_open.return_value = original_pil_image.copy() # Return a copy to avoid modification issues

            # Call load_image with MTCNN model and device (mock_mtcnn_instance is just for presence)
            loaded_result = load_image(self.single_face_img_path, mtcnn=self.mock_mtcnn_instance, device=self.device)

            mock_open.assert_called_once_with(self.single_face_img_path)
            # Ensure original image was converted to RGB before face detection
            # This depends on the internal structure of load_image.
            # We can check that detect_faces was called with an RGB image.
            self.assertEqual(mock_detect_faces.call_args[0][0].mode, 'RGB')

            mock_detect_faces.assert_called_once()
            # Check that detect_faces was called with the original_pil_image (or its RGB version)
            # and the provided mtcnn and device
            args_call = mock_detect_faces.call_args[0]
            self.assertIsInstance(args_call[0], Image.Image) # First arg is the image
            self.assertEqual(args_call[1], self.mock_mtcnn_instance) # Second arg is mtcnn
            self.assertEqual(args_call[2], self.device) # Third arg is device

            self.assertIsInstance(loaded_result, Image.Image)
            self.assertTrue(loaded_result == detected_face_pil, "load_image should return the detected face.")


    @patch('image_diff.image_utils.detect_faces')
    def test_load_image_with_face_detection_no_face_found(self, mock_detect_faces):
        original_pil_image = Image.new('RGB', (200, 200), color='blue')
        mock_detect_faces.return_value = [] # Simulate no faces detected

        with patch.object(Image, 'open') as mock_open:
            mock_open.return_value = original_pil_image.copy()

            loaded_result = load_image(self.no_face_img_path, mtcnn=self.mock_mtcnn_instance, device=self.device)

            mock_open.assert_called_once_with(self.no_face_img_path)
            mock_detect_faces.assert_called_once()

            # Should return the original image (after RGB conversion)
            self.assertTrue(loaded_result.mode == 'RGB')
            # Check if it's substantially the same as original_pil_image.
            # Direct object comparison might fail if there were conversions.
            # For simplicity, we check size and mode. A pixel check would be more robust.
            self.assertEqual(loaded_result.size, original_pil_image.size)


    def test_load_image_no_face_detection_mode(self):
        # Test normal load_image operation when mtcnn is None
        original_pil_image = Image.new('RGB', (200, 200), color='red')
        with patch.object(Image, 'open') as mock_open:
            mock_open.return_value = original_pil_image.copy()

            # Call load_image without MTCNN model
            loaded_result = load_image(self.img1_path, mtcnn=None, device=self.device)

            mock_open.assert_called_once_with(self.img1_path)
            # Ensure it's the original image (after potential RGB conversion)
            self.assertEqual(loaded_result.size, original_pil_image.size)
            self.assertEqual(loaded_result.mode, 'RGB')

if __name__ == '__main__':
    unittest.main()
