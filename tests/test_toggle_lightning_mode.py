import sys
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import os

from finetuning_scheduler.dynamic_versioning.toggle_lightning_mode import parse_args, main


@pytest.mark.parametrize("cli_args,expected_mode,expected_debug", [
    ([], "unified", False),
    (["--mode", "standalone"], "standalone", False),
    (["--debug"], "unified", True),
    (["--mode", "standalone", "--debug"], "standalone", True),
])
def test_parse_args(cli_args, expected_mode, expected_debug):
    """Test argument parsing for toggle_lightning_mode with various argument combinations."""
    with patch('sys.argv', ['toggle_lightning_mode.py'] + cli_args):
        args = parse_args()
        assert args.mode == expected_mode
        assert args.debug == expected_debug


@pytest.mark.parametrize("mode,debug", [("unified", False),("standalone", True) ,])
def test_main_success(mode, debug):
    """Test successful execution of the main function with different arguments."""
    with patch('finetuning_scheduler.dynamic_versioning.toggle_lightning_mode.parse_args') as mock_parse_args, \
         patch('finetuning_scheduler.dynamic_versioning.toggle_lightning_mode.toggle_lightning_imports') as mock_toggle:

        # Setup mock arguments
        mock_args = MagicMock()
        mock_args.mode = mode
        mock_args.debug = debug
        mock_parse_args.return_value = mock_args

        # Call the main function
        result = main()

        # Verify imports were toggled with correct parameters
        mock_toggle.assert_called_once_with(mode, debug=debug)

        # Verify successful return code
        assert result == 0


def test_main_error_handling():
    """Test error handling in the main function."""
    # Setup mock arguments
    mock_args = MagicMock()
    mock_args.mode = "standalone"
    mock_args.debug = True

    with patch('finetuning_scheduler.dynamic_versioning.toggle_lightning_mode.parse_args', return_value=mock_args), \
         patch('finetuning_scheduler.dynamic_versioning.toggle_lightning_mode.toggle_lightning_imports') as mock_tgl, \
         patch('builtins.print') as mock_print:

        # Make toggle_imports raise an exception
        error_message = "Test error"
        mock_tgl.side_effect = Exception(error_message)

        # Call the main function
        result = main()

        # Verify imports were attempted with correct parameters
        mock_tgl.assert_called_once_with("standalone", debug=True)

        # Verify error message was printed
        mock_print.assert_called_once_with(f"Error toggling imports: {error_message}")

        # Verify error return code
        assert result == 1


def test_script_execution_as_main():
    """Test that the script runs successfully when executed as the main module."""
    # Get the path to the script
    script_path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'finetuning_scheduler',
        'dynamic_versioning', 'toggle_lightning_mode.py'
    )

    # Execute the script with --help to avoid making actual changes
    result = subprocess.run([sys.executable, script_path, '--help'], capture_output=True, text=True)

    # Check return code
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"

    # Check that help information was printed
    assert 'usage:' in result.stdout.lower()
    assert 'toggle between standalone and unified lightning imports' in result.stdout.lower()


if __name__ == "__main__":
    sys.exit(pytest.main(["-xvs", __file__]))
