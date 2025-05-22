import unittest
from unittest.mock import patch, MagicMock
import argparse
import os # For os.path.exists and os.path.isdir mocks
import sys

# Ensure the image_diff package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main function from cli.py
from image_diff.cli import main as cli_main

class TestCLI(unittest.TestCase):

import pathlib # For mocking pathlib.Path
import torch # For creating dummy tensors

# Import the main function from cli.py
from image_diff.cli import main as cli_main, CACHE_DIR # Import CACHE_DIR for tests

class TestCLI(unittest.TestCase):

    # Common mocks for most CLI tests
    MOCK_DECORATORS = [
        patch('image_diff.cli.os.path.exists'),
        patch('image_diff.cli.os.path.isdir'),
        patch('image_diff.cli.os.listdir'),
        patch('image_diff.cli.AutoImageProcessor.from_pretrained'),
        patch('image_diff.cli.AutoModel.from_pretrained'),
        patch('image_diff.cli.load_image'),
        patch('image_diff.cli.preprocess_image_batch'),
        patch('image_diff.cli.get_image_embeddings'),
        patch('image_diff.cli.calculate_similarity'),
        patch('image_diff.cli.pathlib.Path.exists'), # Cache related
        patch('image_diff.cli.torch.load'),         # Cache related
        patch('image_diff.cli.torch.save'),          # Cache related
        patch('image_diff.cli.os.path.getmtime')     # Cache related
    ]

    def _apply_mocks(self, func, *args):
        # Helper to apply multiple decorators
        for decorator in reversed(self.MOCK_DECORATORS):
            func = decorator(func)
        return func(*args)

    def run_cli_with_args(self, args_list, mock_getmtime, mock_torch_save, mock_torch_load, 
                          mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                          mock_preprocess_batch, mock_load_img, mock_auto_model, 
                          mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists):
        """Helper function to run cli_main with mocked sys.argv and other mocks."""
        
        # Filesystem & Model Loading Defaults
        mock_os_path_exists.return_value = True # folder_path exists
        mock_isdir.return_value = True      # folder_path is a directory
        mock_listdir.return_value = ['img1.png', 'img2.png'] # Found images
        mock_auto_proc.return_value = MagicMock() # Processor loaded
        mock_auto_model.return_value = MagicMock() # Model loaded
        
        # Image Loading & Processing Defaults
        # Simulate load_image returning a basic PIL-like mock
        mock_pil_image = MagicMock()
        mock_pil_image.convert.return_value = mock_pil_image # If .convert('RGB') is used
        mock_load_img.return_value = mock_pil_image
        
        # Simulate preprocess_image_batch returning a dummy tensor for 2 images
        mock_preprocess_batch.return_value = torch.randn(2, 3, 224, 224) 
        # Simulate get_image_embeddings returning 2 dummy embeddings
        mock_get_embeddings.return_value = torch.randn(2, 768) 
        mock_calc_sim.return_value = 0.5 # Default similarity

        # Cache Defaults (important for controlling test scenarios)
        mock_path_exists.return_value = False # Default: cache does not exist
        mock_torch_load.return_value = {}     # Default: empty cache
        mock_getmtime.return_value = 12345.0  # Default mtime

        with patch('sys.argv', ['image_diff/cli.py'] + args_list):
            try:
                cli_main()
            except SystemExit:
                pass # Argparse calls sys.exit on error or --help

    # Test folder_path is required
    def test_folder_path_required(self):
        # This test doesn't need the full run_cli_with_args setup
        with patch('sys.argv', ['image_diff/cli.py']):
            with self.assertRaises(SystemExit):
                cli_main()

    def test_default_arguments(self):
        def test_logic(mock_getmtime, mock_torch_save, mock_torch_load, 
                       mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                       mock_preprocess_batch, mock_load_img, mock_auto_model, 
                       mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists):
            
            # Prevent the main comparison loop for this arg parsing test
            with patch('image_diff.cli.itertools.combinations', return_value=[]):
                # Patch the ArgumentParser.parse_args method to capture args
                original_parse_args = argparse.ArgumentParser.parse_args
                parsed_args_value = None
                def mock_parse_args_capture(self_parser_instance, args_to_parse=None): # Renamed 'args' to 'args_to_parse'
                    nonlocal parsed_args_value
                    # Use sys.argv[1:] if args_to_parse is None (standard behavior)
                    actual_args_to_parse = args_to_parse if args_to_parse is not None else sys.argv[1:]
                    parsed_args_value = original_parse_args(self_parser_instance, actual_args_to_parse)
                    return parsed_args_value

                with patch('argparse.ArgumentParser.parse_args', side_effect=mock_parse_args_capture):
                    self.run_cli_with_args(['dummy_folder'], mock_getmtime, mock_torch_save, mock_torch_load, 
                                           mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                                           mock_preprocess_batch, mock_load_img, mock_auto_model, 
                                           mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists)
            
            self.assertIsNotNone(parsed_args_value)
            self.assertEqual(parsed_args_value.folder_path, 'dummy_folder')
            self.assertEqual(parsed_args_value.threshold, 0.9)
            self.assertEqual(parsed_args_value.model_name, 'google/vit-base-patch16-224-in21k')
        
        self._apply_mocks(test_logic)


    def test_custom_arguments(self):
        def test_logic(mock_getmtime, mock_torch_save, mock_torch_load, 
                       mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                       mock_preprocess_batch, mock_load_img, mock_auto_model, 
                       mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists):

            with patch('image_diff.cli.itertools.combinations', return_value=[]):
                original_parse_args = argparse.ArgumentParser.parse_args
                parsed_args_value = None
                def mock_parse_args_capture(self_parser_instance, args_to_parse=None): # Renamed 'args' to 'args_to_parse'
                    nonlocal parsed_args_value
                    actual_args_to_parse = args_to_parse if args_to_parse is not None else sys.argv[1:]
                    parsed_args_value = original_parse_args(self_parser_instance, actual_args_to_parse)
                    return parsed_args_value

                with patch('argparse.ArgumentParser.parse_args', side_effect=mock_parse_args_capture):
                    custom_cli_args = ['custom_folder', '--threshold', '0.75', '--model_name', 'custom/model']
                    self.run_cli_with_args(custom_cli_args, mock_getmtime, mock_torch_save, mock_torch_load, 
                                           mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                                           mock_preprocess_batch, mock_load_img, mock_auto_model, 
                                           mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists)

            self.assertIsNotNone(parsed_args_value)
            self.assertEqual(parsed_args_value.folder_path, 'custom_folder')
            self.assertEqual(parsed_args_value.threshold, 0.75)
            self.assertEqual(parsed_args_value.model_name, 'custom/model')

        self._apply_mocks(test_logic)

    @patch('builtins.print')
    def test_folder_not_exists(self, mock_print):
        # Only mock os.path.exists for this specific test, others can be default
        def test_logic(mock_getmtime, mock_torch_save, mock_torch_load, 
                       mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                       mock_preprocess_batch, mock_load_img, mock_auto_model, 
                       mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists):
            mock_os_path_exists.return_value = False # Specific mock for this test
            self.run_cli_with_args(['non_existent_folder'], mock_getmtime, mock_torch_save, mock_torch_load, 
                                   mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                                   mock_preprocess_batch, mock_load_img, mock_auto_model, 
                                   mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists)
            mock_print.assert_any_call("Error: Folder not found: non_existent_folder")
        self._apply_mocks(test_logic)


    @patch('builtins.print')
    def test_path_is_not_directory(self, mock_print):
        def test_logic(mock_getmtime, mock_torch_save, mock_torch_load, 
                       mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                       mock_preprocess_batch, mock_load_img, mock_auto_model, 
                       mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists):
            mock_os_path_exists.return_value = True 
            mock_isdir.return_value = False # Specific mock for this test
            self.run_cli_with_args(['file_path_not_folder'], mock_getmtime, mock_torch_save, mock_torch_load, 
                                   mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                                   mock_preprocess_batch, mock_load_img, mock_auto_model, 
                                   mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists)
            mock_print.assert_any_call("Error: Provided path is not a directory: file_path_not_folder")
        self._apply_mocks(test_logic)

    @patch('builtins.print')
    def test_too_few_images(self, mock_print):
        def test_logic(mock_getmtime, mock_torch_save, mock_torch_load, 
                       mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                       mock_preprocess_batch, mock_load_img, mock_auto_model, 
                       mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists):
            mock_listdir.return_value = ['img1.png'] # Specific mock for this test
            self.run_cli_with_args(['folder_with_one_image'], mock_getmtime, mock_torch_save, mock_torch_load, 
                                   mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                                   mock_preprocess_batch, mock_load_img, mock_auto_model, 
                                   mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists)
            mock_print.assert_any_call("Found 1 image(s). Need at least two images to compare.")
        self._apply_mocks(test_logic)

    def test_cli_with_empty_cache(self):
        # Test that with an empty cache, computation functions are called and save is called
        def test_logic(mock_getmtime, mock_torch_save, mock_torch_load, 
                       mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                       mock_preprocess_batch, mock_load_img, mock_auto_model, 
                       mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists):
            
            mock_path_exists.return_value = False # Simulate cache file not existing
            mock_listdir.return_value = ['imgA.png', 'imgB.png'] # Two images to process
            
            # Mock load_image to return distinct mocks if needed, or just one type
            mock_pil_A = MagicMock(name="PIL_A"); mock_pil_A.convert.return_value = mock_pil_A
            mock_pil_B = MagicMock(name="PIL_B"); mock_pil_B.convert.return_value = mock_pil_B
            mock_load_img.side_effect = [mock_pil_A, mock_pil_B]

            # Mock mtime for these files
            mock_getmtime.side_effect = [111.0, 222.0] # Different mtimes

            # Simulate preprocess_image_batch output for 2 images
            processed_batch_tensor = torch.randn(2, 3, 224, 224)
            mock_preprocess_batch.return_value = processed_batch_tensor
            
            # Simulate get_image_embeddings output for 2 images
            embeddings_batch_tensor = torch.randn(2, 768)
            mock_get_embeddings.return_value = embeddings_batch_tensor

            self.run_cli_with_args(['dummy_folder_empty_cache'], mock_getmtime, mock_torch_save, mock_torch_load, 
                                   mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                                   mock_preprocess_batch, mock_load_img, mock_auto_model, 
                                   mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists)

            mock_preprocess_batch.assert_called_once_with([mock_pil_A, mock_pil_B], mock_auto_proc.return_value)
            mock_get_embeddings.assert_called_once_with(processed_batch_tensor, mock_auto_model.return_value, unittest.mock.ANY) # device
            self.assertTrue(mock_torch_save.called)
            # Check that the saved cache contains the new embeddings
            saved_cache_data = mock_torch_save.call_args[0][1]
            self.assertIn('dummy_folder_empty_cache/imgA.png', saved_cache_data)
            self.assertIn('dummy_folder_empty_cache/imgB.png', saved_cache_data)
            self.assertTrue(torch.equal(saved_cache_data['dummy_folder_empty_cache/imgA.png']['embedding'], embeddings_batch_tensor[0]))
            self.assertEqual(saved_cache_data['dummy_folder_empty_cache/imgA.png']['mtime'], 111.0)

        self._apply_mocks(test_logic)

    def test_cli_with_full_cache(self):
        # Test that with a full and valid cache, computation functions are not called for those images
        def test_logic(mock_getmtime, mock_torch_save, mock_torch_load, 
                       mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                       mock_preprocess_batch, mock_load_img, mock_auto_model, 
                       mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists):

            img_names = ['imgC.png', 'imgD.png']
            folder_name = 'dummy_folder_full_cache'
            full_paths = [os.path.join(folder_name, name) for name in img_names]
            
            mock_listdir.return_value = img_names
            mock_path_exists.return_value = True # Cache file exists

            # Simulate mtimes that will match the cache
            mtimes = {'imgC.png': 333.0, 'imgD.png': 444.0}
            mock_getmtime.side_effect = lambda path: mtimes[os.path.basename(path)]
            
            # Simulate load_image returning distinct mocks
            mock_pil_C = MagicMock(name="PIL_C"); mock_pil_C.convert.return_value = mock_pil_C
            mock_pil_D = MagicMock(name="PIL_D"); mock_pil_D.convert.return_value = mock_pil_D
            mock_load_img.side_effect = [mock_pil_C, mock_pil_D]


            # Pre-populate cache data to be returned by torch.load
            cached_embedding_C = torch.randn(1, 768).squeeze(0) # Squeeze to match storage format
            cached_embedding_D = torch.randn(1, 768).squeeze(0)
            mock_torch_load.return_value = {
                full_paths[0]: {'mtime': mtimes['imgC.png'], 'embedding': cached_embedding_C},
                full_paths[1]: {'mtime': mtimes['imgD.png'], 'embedding': cached_embedding_D}
            }
            
            self.run_cli_with_args([folder_name], mock_getmtime, mock_torch_save, mock_torch_load, 
                                   mock_path_exists, mock_calc_sim, mock_get_embeddings, 
                                   mock_preprocess_batch, mock_load_img, mock_auto_model, 
                                   mock_auto_proc, mock_listdir, mock_isdir, mock_os_path_exists)

            mock_preprocess_batch.assert_not_called() # Should not be called as all are cached
            mock_get_embeddings.assert_not_called()   # Should not be called
            
            # Assert similarity was called with the cached embeddings
            # calculate_similarity expects (1, dim) tensors
            mock_calc_sim.assert_called_once_with(cached_embedding_C.unsqueeze(0), cached_embedding_D.unsqueeze(0))
            
            # Check that torch.save was called to potentially update/resave the cache
            # (even if no new items, it resaves validated items)
            self.assertTrue(mock_torch_save.called)
            saved_cache_data = mock_torch_save.call_args[0][1]
            self.assertTrue(torch.equal(saved_cache_data[full_paths[0]]['embedding'], cached_embedding_C))


        self._apply_mocks(test_logic)


if __name__ == '__main__':
    unittest.main()
