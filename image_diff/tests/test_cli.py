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

import torch # For creating dummy tensors
import lancedb # For spec if needed
import pyarrow as pa # For spec if needed
import json
import csv
from unittest.mock import mock_open # For mocking file open

# Import the main function from cli.py
from image_diff.cli import main as cli_main

class TestCLI(unittest.TestCase):

    # Updated MOCK_DECORATORS for LanceDB and output files
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
        patch('image_diff.cli.lancedb.connect'),    # LanceDB related
        patch('image_diff.cli.os.path.getmtime'),   # LanceDB mtime check
        patch('builtins.open', new_callable=mock_open) # For output file
    ]

    def _apply_mocks(self, func, *args):
        for decorator in reversed(self.MOCK_DECORATORS):
            func = decorator(func)
        return func(*args)

    def run_cli_with_args(self, args_list, mock_open_file, mock_getmtime, mock_lancedb_connect,
                          mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                          mock_load_img, mock_auto_model, mock_auto_proc, 
                          mock_listdir, mock_isdir, mock_os_path_exists):
        """Helper function to run cli_main with updated mocks."""
        
        # Filesystem & Model Loading Defaults
        mock_os_path_exists.return_value = True 
        mock_isdir.return_value = True      
        mock_listdir.return_value = ['img1.png', 'img2.png'] 
        mock_auto_proc_instance = MagicMock()
        mock_auto_proc.return_value = mock_auto_proc_instance
        
        mock_auto_model_instance = MagicMock()
        # Set a default embedding dimension for the model config
        mock_auto_model_instance.config.hidden_size = 768 
        mock_auto_model.return_value = mock_auto_model_instance
        
        # Image Loading & Processing Defaults
        mock_pil_image = MagicMock()
        mock_pil_image.convert.return_value = mock_pil_image
        mock_load_img.return_value = mock_pil_image
        
        mock_preprocess_batch.return_value = torch.randn(len(mock_listdir.return_value), 3, 224, 224) 
        mock_get_embeddings.return_value = torch.randn(len(mock_listdir.return_value), 768) 
        mock_calc_sim.return_value = 0.5 

        # LanceDB Defaults
        self.mock_db_conn = MagicMock()
        mock_lancedb_connect.return_value = self.mock_db_conn
        self.mock_table = MagicMock()
        # Default: table does not exist, so open_table raises FileNotFoundError
        self.mock_db_conn.open_table.side_effect = FileNotFoundError 
        self.mock_db_conn.create_table.return_value = self.mock_table # create_table returns the mock table
        # Default: table search returns no results
        self.mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = [] 
        
        mock_getmtime.return_value = 12345.0  # Default mtime

        with patch('sys.argv', ['image_diff/cli.py'] + args_list):
            try:
                cli_main()
            except SystemExit:
                pass 

    def test_folder_path_required(self):
        with patch('sys.argv', ['image_diff/cli.py']):
            with self.assertRaises(SystemExit):
                cli_main()

    def test_default_arguments(self):
        def test_logic(mock_open_file, mock_getmtime, mock_lancedb_connect,
                       mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                       mock_load_img, mock_auto_model, mock_auto_proc, 
                       mock_listdir, mock_isdir, mock_os_path_exists):
            
            with patch('image_diff.cli.itertools.combinations', return_value=[]):
                original_parse_args = argparse.ArgumentParser.parse_args
                parsed_args_value = None
                def mock_parse_args_capture(self_parser_instance, args_to_parse=None):
                    nonlocal parsed_args_value
                    actual_args_to_parse = args_to_parse if args_to_parse is not None else sys.argv[1:]
                    parsed_args_value = original_parse_args(self_parser_instance, actual_args_to_parse)
                    return parsed_args_value

                with patch('argparse.ArgumentParser.parse_args', side_effect=mock_parse_args_capture):
                    self.run_cli_with_args(['dummy_folder'], mock_open_file, mock_getmtime, mock_lancedb_connect,
                                           mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                                           mock_load_img, mock_auto_model, mock_auto_proc, 
                                           mock_listdir, mock_isdir, mock_os_path_exists)
            
            self.assertIsNotNone(parsed_args_value)
            self.assertEqual(parsed_args_value.folder_path, 'dummy_folder')
            self.assertEqual(parsed_args_value.threshold, 0.9)
            self.assertEqual(parsed_args_value.model_name, 'google/vit-base-patch16-224-in21k')
            self.assertEqual(parsed_args_value.output_file, 'similar_pairs.csv') # Check new default
        
        self._apply_mocks(test_logic)

    def test_custom_arguments(self):
        def test_logic(mock_open_file, mock_getmtime, mock_lancedb_connect,
                       mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                       mock_load_img, mock_auto_model, mock_auto_proc, 
                       mock_listdir, mock_isdir, mock_os_path_exists):

            with patch('image_diff.cli.itertools.combinations', return_value=[]):
                original_parse_args = argparse.ArgumentParser.parse_args
                parsed_args_value = None
                def mock_parse_args_capture(self_parser_instance, args_to_parse=None):
                    nonlocal parsed_args_value
                    actual_args_to_parse = args_to_parse if args_to_parse is not None else sys.argv[1:]
                    parsed_args_value = original_parse_args(self_parser_instance, actual_args_to_parse)
                    return parsed_args_value

                with patch('argparse.ArgumentParser.parse_args', side_effect=mock_parse_args_capture):
                    custom_cli_args = ['custom_folder', '--threshold', '0.75', 
                                       '--model_name', 'custom/model', '--output_file', 'out.json']
                    self.run_cli_with_args(custom_cli_args, mock_open_file, mock_getmtime, mock_lancedb_connect,
                                           mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                                           mock_load_img, mock_auto_model, mock_auto_proc, 
                                           mock_listdir, mock_isdir, mock_os_path_exists)

            self.assertIsNotNone(parsed_args_value)
            self.assertEqual(parsed_args_value.folder_path, 'custom_folder')
            self.assertEqual(parsed_args_value.threshold, 0.75)
            self.assertEqual(parsed_args_value.model_name, 'custom/model')
            self.assertEqual(parsed_args_value.output_file, 'out.json')

        self._apply_mocks(test_logic)

    @patch('builtins.print')
    def test_folder_not_exists(self, mock_print):
        def test_logic(*mocks): # All mocks passed via _apply_mocks
            mock_os_path_exists = mocks[-1] # Last one in MOCK_DECORATORS is os.path.exists
            mock_os_path_exists.return_value = False 
            self.run_cli_with_args(['non_existent_folder'], *mocks)
            mock_print.assert_any_call("Error: Folder not found: non_existent_folder")
        self._apply_mocks(test_logic)

    @patch('builtins.print')
    def test_path_is_not_directory(self, mock_print):
        def test_logic(*mocks):
            mock_isdir = mocks[-2] # Second to last is os.path.isdir
            mock_os_path_exists = mocks[-1]
            mock_os_path_exists.return_value = True 
            mock_isdir.return_value = False 
            self.run_cli_with_args(['file_path_not_folder'], *mocks)
            mock_print.assert_any_call("Error: Provided path is not a directory: file_path_not_folder")
        self._apply_mocks(test_logic)

    @patch('builtins.print')
    def test_too_few_images(self, mock_print):
        def test_logic(*mocks):
            mock_listdir = mocks[-3] # Third to last is os.listdir
            mock_listdir.return_value = ['img1.png'] 
            self.run_cli_with_args(['folder_with_one_image'], *mocks)
            mock_print.assert_any_call("Found 1 image(s). Need at least two images to compare.")
        self._apply_mocks(test_logic)

    # LanceDB Integration Tests
    def test_cli_lancedb_new_images_create_table(self):
        def test_logic(mock_open_file, mock_getmtime, mock_lancedb_connect,
                       mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                       mock_load_img, mock_auto_model, mock_auto_proc, 
                       mock_listdir, mock_isdir, mock_os_path_exists):
            
            # Setup: Table does not exist, then gets created
            self.mock_db_conn.open_table.side_effect = FileNotFoundError 
            mock_listdir.return_value = ['new1.png', 'new2.png']
            mock_getmtime.return_value = 123.0

            # Expected embeddings to be added
            expected_embeddings = torch.randn(2, 768)
            mock_get_embeddings.return_value = expected_embeddings

            with patch('image_diff.cli.itertools.combinations', return_value=[]): # No comparisons for this test
                self.run_cli_with_args(['dummy_folder'], mock_open_file, mock_getmtime, mock_lancedb_connect,
                                   mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                                   mock_load_img, mock_auto_model, mock_auto_proc, 
                                   mock_listdir, mock_isdir, mock_os_path_exists)

            self.mock_db_conn.create_table.assert_called_once()
            self.mock_table.add.assert_called_once()
            # Check data added to table
            added_data = self.mock_table.add.call_args[0][0]
            self.assertEqual(len(added_data), 2)
            self.assertEqual(added_data[0]['image_path'], 'new1.png')
            self.assertEqual(added_data[0]['mtime'], 123.0)
            self.assertEqual(added_data[0]['embedding'], expected_embeddings[0].tolist())
            mock_preprocess_batch.assert_called_once()
            mock_get_embeddings.assert_called_once()
        self._apply_mocks(test_logic)

    def test_cli_lancedb_existing_images_no_change(self):
        def test_logic(mock_open_file, mock_getmtime, mock_lancedb_connect,
                       mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                       mock_load_img, mock_auto_model, mock_auto_proc, 
                       mock_listdir, mock_isdir, mock_os_path_exists):
            
            self.mock_db_conn.open_table.side_effect = None # Table exists
            self.mock_db_conn.open_table.return_value = self.mock_table
            
            img_names = ['cached1.png', 'cached2.png']
            mock_listdir.return_value = img_names
            mtime_val = 456.0
            mock_getmtime.return_value = mtime_val # All files have this mtime
            
            cached_embeddings_list = [
                {'image_path': 'cached1.png', 'mtime': mtime_val, 'embedding': torch.randn(768).tolist()},
                {'image_path': 'cached2.png', 'mtime': mtime_val, 'embedding': torch.randn(768).tolist()}
            ]
            # Configure search to return these for each image
            self.mock_table.search.return_value.where.return_value.limit.return_value.to_list.side_effect = [
                [cached_embeddings_list[0]], # Search for cached1.png
                [cached_embeddings_list[1]]  # Search for cached2.png
            ]
            
            with patch('image_diff.cli.itertools.combinations', return_value=[(0,1)]): # One comparison
                 self.run_cli_with_args(['dummy_folder'], mock_open_file, mock_getmtime, mock_lancedb_connect,
                                   mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                                   mock_load_img, mock_auto_model, mock_auto_proc, 
                                   mock_listdir, mock_isdir, mock_os_path_exists)

            mock_preprocess_batch.assert_not_called()
            mock_get_embeddings.assert_not_called()
            self.mock_table.add.assert_not_called()
            self.mock_table.delete.assert_not_called()
            mock_calc_sim.assert_called_once() # Ensure comparison happened with cached embeddings
        self._apply_mocks(test_logic)

    def test_cli_lancedb_existing_images_changed_mtime(self):
        def test_logic(mock_open_file, mock_getmtime, mock_lancedb_connect,
                       mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                       mock_load_img, mock_auto_model, mock_auto_proc, 
                       mock_listdir, mock_isdir, mock_os_path_exists):
            
            self.mock_db_conn.open_table.side_effect = None
            self.mock_db_conn.open_table.return_value = self.mock_table
            
            mock_listdir.return_value = ['changed.png']
            
            # mtime in DB is old, current mtime is new
            old_mtime = 789.0
            new_mtime = 999.0
            mock_getmtime.return_value = new_mtime # Current file mtime
            
            self.mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = [
                {'image_path': 'changed.png', 'mtime': old_mtime, 'embedding': torch.randn(768).tolist()}
            ]
            
            new_embedding = torch.randn(1, 768)
            mock_get_embeddings.return_value = new_embedding # This will be the newly computed embedding

            with patch('image_diff.cli.itertools.combinations', return_value=[]):
                self.run_cli_with_args(['dummy_folder'], mock_open_file, mock_getmtime, mock_lancedb_connect,
                                   mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                                   mock_load_img, mock_auto_model, mock_auto_proc, 
                                   mock_listdir, mock_isdir, mock_os_path_exists)

            mock_preprocess_batch.assert_called_once()
            mock_get_embeddings.assert_called_once()
            self.mock_table.delete.assert_called_once_with("image_path = 'changed.png'")
            self.mock_table.add.assert_called_once()
            added_data = self.mock_table.add.call_args[0][0]
            self.assertEqual(added_data[0]['mtime'], new_mtime)
            self.assertEqual(added_data[0]['embedding'], new_embedding[0].tolist())
        self._apply_mocks(test_logic)

    # Output File Tests
    @patch('image_diff.cli.csv.DictWriter') # Mock csv.DictWriter specifically
    def test_cli_output_csv(self, mock_csv_writer_class):
        def test_logic(mock_open_file, mock_getmtime, mock_lancedb_connect,
                       mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                       mock_load_img, mock_auto_model, mock_auto_proc, 
                       mock_listdir, mock_isdir, mock_os_path_exists, # These are from _apply_mocks
                       mock_csv_writer_class_arg): # This is from @patch
            
            mock_calc_sim.return_value = 0.95 # Ensure one pair is found
            # Mock listdir to provide two images for one comparison
            mock_listdir.return_value = ['imgS1.png', 'imgS2.png']
            mock_get_embeddings.return_value = torch.randn(2, 768) # Match listdir length

            self.run_cli_with_args(['dummy_folder', '--output_file', 'test.csv'], 
                                   mock_open_file, mock_getmtime, mock_lancedb_connect,
                                   mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                                   mock_load_img, mock_auto_model, mock_auto_proc, 
                                   mock_listdir, mock_isdir, mock_os_path_exists)
            
            mock_open_file.assert_called_with('test.csv', 'w', newline='')
            mock_csv_writer_instance = mock_csv_writer_class_arg.return_value
            mock_csv_writer_instance.writeheader.assert_called_once()
            mock_csv_writer_instance.writerows.assert_called_once()
            # Check data passed to writerows
            written_data = mock_csv_writer_instance.writerows.call_args[0][0]
            self.assertEqual(len(written_data), 1)
            self.assertEqual(written_data[0]['image1'], 'imgS1.png')
            self.assertEqual(written_data[0]['image2'], 'imgS2.png')
            self.assertEqual(written_data[0]['similarity'], 0.95)

        # Need to pass the mock_csv_writer_class from the decorator into test_logic
        self._apply_mocks(lambda *args: test_logic(*args, mock_csv_writer_class))


    @patch('image_diff.cli.json.dump')
    def test_cli_output_json(self, mock_json_dump):
        def test_logic(mock_open_file, mock_getmtime, mock_lancedb_connect,
                       mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                       mock_load_img, mock_auto_model, mock_auto_proc, 
                       mock_listdir, mock_isdir, mock_os_path_exists, # from _apply_mocks
                       mock_json_dump_arg): # from @patch
            
            mock_calc_sim.return_value = 0.92
            mock_listdir.return_value = ['imgJ1.png', 'imgJ2.png']
            mock_get_embeddings.return_value = torch.randn(2, 768)

            self.run_cli_with_args(['dummy_folder', '--output_file', 'test.json'],
                                   mock_open_file, mock_getmtime, mock_lancedb_connect,
                                   mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                                   mock_load_img, mock_auto_model, mock_auto_proc, 
                                   mock_listdir, mock_isdir, mock_os_path_exists)
            
            mock_open_file.assert_called_with('test.json', 'w')
            mock_json_dump_arg.assert_called_once()
            dumped_data = mock_json_dump_arg.call_args[0][0]
            self.assertEqual(len(dumped_data), 1)
            self.assertEqual(dumped_data[0]['image1'], 'imgJ1.png')
            self.assertEqual(dumped_data[0]['similarity'], 0.92)

        self._apply_mocks(lambda *args: test_logic(*args, mock_json_dump))

    def test_cli_output_txt(self):
        def test_logic(mock_open_file, mock_getmtime, mock_lancedb_connect,
                       mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                       mock_load_img, mock_auto_model, mock_auto_proc, 
                       mock_listdir, mock_isdir, mock_os_path_exists):
            
            mock_calc_sim.return_value = 0.98
            mock_listdir.return_value = ['imgT1.png', 'imgT2.png']
            mock_get_embeddings.return_value = torch.randn(2, 768)

            self.run_cli_with_args(['dummy_folder', '--output_file', 'test.txt'],
                                   mock_open_file, mock_getmtime, mock_lancedb_connect,
                                   mock_calc_sim, mock_get_embeddings, mock_preprocess_batch, 
                                   mock_load_img, mock_auto_model, mock_auto_proc, 
                                   mock_listdir, mock_isdir, mock_os_path_exists)
            
            mock_open_file.assert_called_with('test.txt', 'w')
            mock_file_handle = mock_open_file.return_value
            self.assertTrue(mock_file_handle.write.called)
            # Example check for content (might be more specific)
            args, kwargs = mock_file_handle.write.call_args
            self.assertIn("Image 1: imgT1.png", args[0])
            self.assertIn("Similarity: 0.98", args[0])

        self._apply_mocks(test_logic)

if __name__ == '__main__':
    unittest.main()
