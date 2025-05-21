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

    @patch('image_diff.cli.os.path.exists')
    @patch('image_diff.cli.os.path.isdir')
    @patch('image_diff.cli.os.listdir')
    @patch('image_diff.cli.AutoImageProcessor.from_pretrained') # Mock model/processor loading
    @patch('image_diff.cli.AutoModel.from_pretrained')
    @patch('image_diff.cli.load_image') # Further mock image processing functions
    @patch('image_diff.cli.preprocess_image')
    @patch('image_diff.cli.get_image_embedding')
    @patch('image_diff.cli.calculate_similarity')
    def run_cli_with_args(self, args_list, mock_calc_sim, mock_get_emb, mock_preproc, mock_load_img, 
                          mock_auto_model, mock_auto_proc, mock_listdir, mock_isdir, mock_exists):
        """Helper function to run cli_main with mocked sys.argv and other mocks."""
        # Default mocks for filesystem and model loading to prevent errors
        mock_exists.return_value = True # Assume folder_path exists
        mock_isdir.return_value = True  # Assume folder_path is a directory
        mock_listdir.return_value = ['img1.png', 'img2.png'] # Simulate finding images
        mock_auto_proc.return_value = MagicMock() # Simulate successful processor load
        mock_auto_model.return_value = MagicMock() # Simulate successful model load
        mock_load_img.return_value = MagicMock() # Simulate successful image load
        mock_preproc.return_value = MagicMock() # Simulate successful preprocessing
        mock_get_emb.return_value = MagicMock() # Simulate successful embedding
        mock_calc_sim.return_value = 0.5 # Simulate a similarity score

        with patch('sys.argv', ['image_diff/cli.py'] + args_list):
            try:
                cli_main()
            except SystemExit: # Argparse calls sys.exit on error
                pass # Expected for missing required args, or --help

    # 1. Test folder_path is required
    # We need to capture the actual ArgumentParser instance to check its behavior,
    # or check for SystemExit, which is what argparse does.
    @patch('argparse.ArgumentParser.parse_args')
    def test_folder_path_required(self, mock_parse_args):
        # Simulate calling with no arguments
        with patch('sys.argv', ['image_diff/cli.py']):
            # Argparse by default prints to stderr and exits.
            # We expect SystemExit if a required argument is missing.
            with self.assertRaises(SystemExit):
                cli_main()
    
    # 2. Test default arguments
    # This requires letting parse_args run and then checking the args object.
    # We'll mock the rest of the main function to prevent actual processing.
    @patch('image_diff.cli.os.path.exists', return_value=True)
    @patch('image_diff.cli.os.path.isdir', return_value=True)
    @patch('image_diff.cli.os.listdir', return_value=['img1.png', 'img2.png']) # Need at least 2 images
    @patch('image_diff.cli.AutoImageProcessor.from_pretrained', return_value=MagicMock())
    @patch('image_diff.cli.AutoModel.from_pretrained', return_value=MagicMock())
    @patch('image_diff.cli.itertools.combinations', return_value=[]) # Prevent actual comparison loop
    def test_default_arguments(self, mock_combinations, mock_model, mock_processor, mock_listdir, mock_isdir, mock_exists):
        # To check the parsed args, we need to intercept them.
        # One way is to patch the ArgumentParser.parse_args method.
        
        original_parse_args = argparse.ArgumentParser.parse_args
        parsed_args_value = None

        def mock_parse_args_capture(self_parser_instance, args=None):
            nonlocal parsed_args_value
            # If args are passed directly (e.g. from a test), use them.
            # Otherwise, parse_args will use sys.argv internally.
            parsed_args_value = original_parse_args(self_parser_instance, args=sys.argv[1:])
            return parsed_args_value

        with patch('argparse.ArgumentParser.parse_args', side_effect=mock_parse_args_capture) as mp_parse_args:
            with patch('sys.argv', ['image_diff/cli.py', 'dummy_folder']):
                try:
                    cli_main()
                except SystemExit: # Should not exit if args are okay
                    self.fail("CLI exited unexpectedly with valid default args setup.")
            
            self.assertIsNotNone(parsed_args_value)
            self.assertEqual(parsed_args_value.folder_path, 'dummy_folder')
            self.assertEqual(parsed_args_value.threshold, 0.9) # Default value
            self.assertEqual(parsed_args_value.model_name, 'google/vit-base-patch16-224-in21k') # Default


    @patch('image_diff.cli.os.path.exists', return_value=True)
    @patch('image_diff.cli.os.path.isdir', return_value=True)
    @patch('image_diff.cli.os.listdir', return_value=['img1.png', 'img2.png'])
    @patch('image_diff.cli.AutoImageProcessor.from_pretrained', return_value=MagicMock())
    @patch('image_diff.cli.AutoModel.from_pretrained', return_value=MagicMock())
    @patch('image_diff.cli.itertools.combinations', return_value=[]) # Prevent actual comparison loop
    def test_custom_arguments(self, mock_combinations, mock_model, mock_processor, mock_listdir, mock_isdir, mock_exists):
        
        original_parse_args = argparse.ArgumentParser.parse_args
        parsed_args_value = None

        def mock_parse_args_capture(self_parser_instance, args=None):
            nonlocal parsed_args_value
            parsed_args_value = original_parse_args(self_parser_instance, args=sys.argv[1:])
            return parsed_args_value

        with patch('argparse.ArgumentParser.parse_args', side_effect=mock_parse_args_capture) as mp_parse_args:
            custom_args = [
                'custom_folder',
                '--threshold', '0.75',
                '--model_name', 'custom/model'
            ]
            with patch('sys.argv', ['image_diff/cli.py'] + custom_args):
                try:
                    cli_main()
                except SystemExit:
                     self.fail("CLI exited unexpectedly with custom args setup.")

            self.assertIsNotNone(parsed_args_value)
            self.assertEqual(parsed_args_value.folder_path, 'custom_folder')
            self.assertEqual(parsed_args_value.threshold, 0.75)
            self.assertEqual(parsed_args_value.model_name, 'custom/model')

    @patch('builtins.print') # Mock print to check output
    @patch('image_diff.cli.os.path.exists', return_value=False) # Folder does not exist
    def test_folder_not_exists(self, mock_exists, mock_print):
        self.run_cli_with_args(['non_existent_folder'])
        mock_print.assert_any_call("Error: Folder not found: non_existent_folder")

    @patch('builtins.print')
    @patch('image_diff.cli.os.path.exists', return_value=True)
    @patch('image_diff.cli.os.path.isdir', return_value=False) # Path is not a directory
    def test_path_is_not_directory(self, mock_isdir, mock_exists, mock_print):
        self.run_cli_with_args(['file_path_not_folder'])
        mock_print.assert_any_call("Error: Provided path is not a directory: file_path_not_folder")
    
    @patch('builtins.print')
    @patch('image_diff.cli.os.path.exists', return_value=True)
    @patch('image_diff.cli.os.path.isdir', return_value=True)
    @patch('image_diff.cli.os.listdir', return_value=['img1.png']) # Only one image
    def test_too_few_images(self, mock_listdir, mock_isdir, mock_exists, mock_print):
        self.run_cli_with_args(['folder_with_one_image'])
        mock_print.assert_any_call("Found 1 image(s). Need at least two images to compare.")


if __name__ == '__main__':
    unittest.main()
