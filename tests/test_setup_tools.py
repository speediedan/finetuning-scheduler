import os
import sys
import tempfile
from unittest.mock import patch
import pytest

# Import setup_tools functions for testing
from finetuning_scheduler.setup_tools import (_load_readme_description, _load_requirements, disable_always_warnings)

def test_load_requirements():
    """Test loading requirements from a file."""
    # Create a temporary requirements file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
        temp_file.write(b'package1>=1.0.0\n')
        temp_file.write(b'# Comment line\n')
        temp_file.write(b'package2==2.0.0 # Another comment\n')
        temp_file.write(b'git+https://github.com/user/repo.git@branch#egg=package3\n')
        temp_file.write(b'http://example.com/package4.tar.gz\n')
        temp_path = temp_file.name

    try:
        # Test loading requirements
        with patch('builtins.print'):  # Suppress print statements without needing mock_print
            reqs = _load_requirements(os.path.dirname(temp_path), os.path.basename(temp_path))

        # Check result
        assert len(reqs) == 2  # Only normal package requirements should be included
        assert 'package1>=1.0.0' in reqs
        assert 'package2==2.0.0' in reqs

        # Check that direct dependencies were skipped
        assert not any(r.startswith('git+') for r in reqs)
        assert not any(r.startswith('http') for r in reqs)

    finally:
        # Clean up
        os.unlink(temp_path)


def test_load_readme_description():
    """Test loading readme as description."""
    # Create a temporary README file
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
        temp_file.write(b'<div align="center">Test README</div>\n')
        temp_file.write(b'docs/source/_static/image.png\n')
        temp_file.write(b'badge/?version=stable\n')
        temp_file.write(b'finetuning-scheduler.readthedocs.io/en/latest/\n')
        temp_file.write(b'/branch/main/graph/badge.svg\n')
        temp_file.write(b'badge.svg?branch=main&event=push\n')
        temp_file.write(b'?branchName=main\n')
        temp_file.write(b'?definitionId=2&branchName=main\n')
        temp_file.write(b'<!-- following section will be skipped from PyPI description -->\n')
        temp_file.write(b'Skip this content\n')
        temp_file.write(b'<!-- end skipping PyPI description -->\n')
        temp_path = temp_file.name

    try:
        # Test loading README
        with patch('os.path.join', return_value=temp_path):
            description = _load_readme_description("/fake/path", "https://github.com/user/repo", "1.0.0")

            # Check replacements
            assert '<div align="center">Test README</div>' in description
            assert 'https://github.com/user/repo/raw/v1.0.0/docs/source/_static/' in description
            assert 'badge/?version=1.0.0' in description
            assert 'finetuning-scheduler.readthedocs.io/en/1.0.0' in description
            assert '/release/1.0.0/graph/badge.svg' in description
            assert 'badge.svg?tag=1.0.0' in description
            assert '?branchName=refs%2Ftags%2F1.0.0' in description
            assert 'Skip this content' not in description
    finally:
        # Clean up
        os.unlink(temp_path)


def test_disable_always_warnings():
    """Test that disable_always_warnings context manager works."""
    import warnings

    # Create a test warning filter
    with patch('warnings.simplefilter') as mock_simplefilter:
        # Test with 'always' which should be blocked
        with disable_always_warnings():
            warnings.simplefilter('always')
            # The call with 'always' should not reach the real simplefilter
            mock_simplefilter.assert_not_called()

            # But other calls should work
            warnings.simplefilter('ignore')
            mock_simplefilter.assert_called_with('ignore')


if __name__ == "__main__":
    sys.exit(pytest.main(["-xvs", __file__]))
