import sys
from unittest.mock import patch
from pathlib import Path
import pytest

from finetuning_scheduler.dynamic_versioning.utils import (
    get_lightning_requirement, _retrieve_files,
    _replace_imports, _check_import_format, use_standalone_pl, use_unified_pl,
    get_project_paths, toggle_lightning_imports,
    LIGHTNING_PACKAGE_MAPPING, EXCLUDE_FILES_FROM_CONVERSION, get_requirement_files,
    _is_package_installed, get_base_dependencies, BASE_DEPENDENCIES
)


def test_get_lightning_requirement():
    """Test getting the Lightning requirement string.

    Note: Commit pinning is now handled at install time via UV_OVERRIDE,
    so get_lightning_requirement just returns version-constrained requirements.
    """
    # Test unified package
    req = get_lightning_requirement("unified")
    assert "lightning>=" in req
    assert "@" not in req

    # Test standalone package
    req = get_lightning_requirement("standalone")
    assert "pytorch-lightning>=" in req
    assert "@" not in req


def test_lightning_package_mapping():
    """Test Lightning package mapping constants."""
    assert "lightning.pytorch" in LIGHTNING_PACKAGE_MAPPING
    assert LIGHTNING_PACKAGE_MAPPING["lightning.pytorch"] == "pytorch_lightning"
    assert "lightning.fabric" in LIGHTNING_PACKAGE_MAPPING
    assert LIGHTNING_PACKAGE_MAPPING["lightning.fabric"] == "lightning_fabric"

def test_retrieve_files(tmp_path):
    """Test retrieving files with extensions and exclusions."""
    # Create directory structure
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    # Create test files
    py_file1 = test_dir / "test1.py"
    py_file1.write_text("# Test file 1")

    py_file2 = test_dir / "test2.py"
    py_file2.write_text("# Test file 2")

    txt_file = test_dir / "test.txt"
    txt_file.write_text("Test text file")

    exclude_file = test_dir / "utils.py"
    exclude_file.write_text("# Should be excluded")

    # Create subdirectory with a file to test path-based exclusion
    subdir = test_dir / "dynamic_versioning"
    subdir.mkdir()
    path_exclude_file = subdir / "utils.py"
    path_exclude_file.write_text("# Should be excluded by path")

    other_file_in_subdir = subdir / "other.py"
    other_file_in_subdir.write_text("# Should not be excluded")

    # Test retrieving all files without extension filter
    all_files = _retrieve_files(str(test_dir))
    assert len(all_files) == 6

    # Test retrieving only python files
    py_files = _retrieve_files(str(test_dir), ".py")
    assert len(py_files) == 5
    assert str(py_file1) in py_files
    assert str(py_file2) in py_files

    # Test with filename exclusions
    py_files_with_exclusion = _retrieve_files(str(test_dir), ".py", exclude_files=["utils.py"])
    assert len(py_files_with_exclusion) == 3  # Expects 3 files (excluding both utils.py files)
    assert str(exclude_file) not in py_files_with_exclusion
    assert str(path_exclude_file) not in py_files_with_exclusion

    # Test with path-based exclusions
    py_files_with_path_exclusion = _retrieve_files(str(test_dir), ".py",
                                                   exclude_files=["dynamic_versioning/utils.py"])
    assert len(py_files_with_path_exclusion) == 4
    assert str(exclude_file) in py_files_with_path_exclusion  # Regular utils.py should be included
    assert str(path_exclude_file) not in py_files_with_path_exclusion  # dynamic_versioning/utils.py excluded
    assert str(other_file_in_subdir) in py_files_with_path_exclusion  # Other files should be included


def test_replace_imports():
    """Test replacing imports in code blocks."""
    # Test "from" import replacements
    lines = [
        "from lightning.pytorch import Trainer",
        "from lightning.pytorch.callbacks import Callback",
        "import os",
        "import lightning.pytorch as pl",
        "from lightning.fabric import Fabric"
    ]

    mapping = [("lightning.pytorch", "pytorch_lightning"), ("lightning.fabric", "lightning_fabric")]

    # Replace imports
    replaced_lines = _replace_imports(lines, mapping)

    assert replaced_lines[0] == "from pytorch_lightning import Trainer"
    assert replaced_lines[1] == "from pytorch_lightning.callbacks import Callback"
    assert replaced_lines[2] == "import os"
    assert replaced_lines[3] == "import pytorch_lightning as pl"
    assert replaced_lines[4] == "from lightning_fabric import Fabric"

    # Test with lightning_by argument
    replaced_lines = _replace_imports(lines, mapping, lightning_by="pytorch_lightning")
    assert "from pytorch_lightning import " in replaced_lines[0]


def test_check_import_format():
    """Test checking import formats."""
    # Test code with unified imports
    unified_code = """
    from lightning.pytorch import Trainer
    import lightning.fabric as fabric
    """

    # Test code with standalone imports
    standalone_code = """
    from pytorch_lightning import Trainer
    import lightning_fabric as fabric
    """
    source_imports = ["lightning.pytorch", "lightning.fabric"]
    # Check if the code needs conversion to standalone
    assert not _check_import_format(unified_code, source_imports)
    assert _check_import_format(standalone_code, source_imports)


def test_toggle_lightning_imports():
    """Test toggling Lightning imports with different parameters."""
    # Test with mock implementations to verify behavior
    with patch('finetuning_scheduler.dynamic_versioning.utils._is_package_installed', return_value=True), \
         patch('finetuning_scheduler.dynamic_versioning.utils.get_project_paths',
               return_value=(Path("/fake/path"), {"source": Path("/fake/path/src")})), \
         patch('finetuning_scheduler.dynamic_versioning.utils.use_standalone_pl') as mock_standalone, \
         patch('finetuning_scheduler.dynamic_versioning.utils.use_unified_pl') as mock_unified:

        # Test standalone mode
        toggle_lightning_imports("standalone", debug=True)
        mock_standalone.assert_called_once()
        mock_unified.assert_not_called()

        # Reset mocks
        mock_standalone.reset_mock()
        mock_unified.reset_mock()

        # Test unified mode
        toggle_lightning_imports("unified")
        mock_unified.assert_called_once()
        mock_standalone.assert_not_called()


def test_get_requirement_files():
    """Test the behavior of get_requirement_files.

    Note: Commit pinning is now handled at install time via UV_OVERRIDE,
    so get_requirement_files just returns version-constrained requirements.
    """
    # Test standalone mode
    reqs = get_requirement_files(standalone=True)
    assert any(r.startswith("pytorch-lightning>=") for r in reqs)
    # Should include base dependencies
    assert any(r.startswith("torch>=") for r in reqs)

    # Test unified mode
    reqs = get_requirement_files(standalone=False)
    assert any(r.startswith("lightning>=") for r in reqs)
    assert any(r.startswith("torch>=") for r in reqs)


def test_get_base_dependencies():
    """Test that get_base_dependencies returns expected deps."""
    deps = get_base_dependencies()
    assert isinstance(deps, list)
    assert any(d.startswith("torch>=") for d in deps)
    # Ensure it returns a copy, not the original
    deps.append("test-package")
    assert "test-package" not in BASE_DEPENDENCIES


def test_get_requirement_files_comment_handling():
    """Test that get_requirement_files correctly handles base dependencies."""
    # Dependencies come from BASE_DEPENDENCIES constant
    reqs = get_requirement_files(standalone=False)
    # Should have torch and Lightning
    assert any(r.startswith("torch>=") for r in reqs)
    assert any(r.startswith("lightning>=") for r in reqs)
    # Lightning should not appear twice
    lightning_count = sum(1 for r in reqs if r.startswith("lightning>=") or r.startswith("lightning "))
    assert lightning_count == 1


@pytest.mark.parametrize("use_function,source_format,target_format,sample_content", [
    (use_standalone_pl, "unified", "standalone", """
from lightning.pytorch import Trainer
import lightning.fabric as fabric
"""),
    (use_unified_pl, "standalone", "unified", """
from pytorch_lightning import Trainer
import lightning_fabric as fabric
""")
])
def test_lightning_import_conversion(tmp_path, use_function, source_format, target_format, sample_content):
    """Test Lightning import conversion with various file scenarios."""
    # Set up test directory
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create test Python files
    py_file1 = src_dir / "module.py"
    py_file1.write_text(sample_content)

    # Create file that should be excluded
    exclude_dir = src_dir / "dynamic_versioning"
    exclude_dir.mkdir()
    exclude_file = exclude_dir / "utils.py"  # Should be excluded
    exclude_file.write_text(f"""
# This file should be excluded
from {"lightning.pytorch" if source_format == "unified" else "pytorch_lightning"} import Trainer
""")

    # Create a binary file to test UnicodeDecodeError handling
    binary_file = src_dir / "binary.py"
    with open(binary_file, 'wb') as f:
        f.write(b'\x80\x81\x82')

    # Test with debug=True to check debug output
    with patch('builtins.print') as mock_print:
        # Use normalized forward slash path for exclusion consistently across platforms
        with patch('finetuning_scheduler.dynamic_versioning.utils.EXCLUDE_FILES_FROM_CONVERSION',
                   ["dynamic_versioning/utils.py"]):
            use_function({src_dir}, debug=True)

            # Verify the conversion happened correctly for non-excluded files
            with open(py_file1) as f:
                content = f.read()
                if target_format == "standalone":
                    assert "from pytorch_lightning import Trainer" in content
                else:
                    assert "from lightning.pytorch import Trainer" in content

            # Excluded file should remain unchanged
            with open(exclude_file) as f:
                content = f.read()
                # Check that file still has the original import format
                if source_format == "unified":
                    assert "from lightning.pytorch import Trainer" in content
                else:
                    assert "from pytorch_lightning import Trainer" in content

            # Check that prints happened correctly
            assert any("Updated imports in" in call_args[0][0] for call_args in mock_print.call_args_list)
            assert any("Skipping" in call_args[0][0] and "prevent self-modification" in call_args[0][0]
                       for call_args in mock_print.call_args_list)


@pytest.mark.parametrize("use_function,target_format,already_converted_content", [
    (use_standalone_pl, "standalone", """
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from lightning_fabric import Fabric
"""),
    (use_unified_pl, "unified", """
from lightning.pytorch import Trainer
import lightning.pytorch as pl
""")
])
def test_lightning_imports_already_converted(tmp_path, use_function, target_format, already_converted_content):
    """Test behavior with files that are already in the target format."""
    # Set up test directory
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create a Python file that's already in the target format
    py_file = src_dir / f"already_{target_format}.py"
    py_file.write_text(already_converted_content)

    # Execute with debug=True to test the debug output
    with patch('builtins.print') as mock_print:
        use_function({src_dir}, debug=True)

        # Check debug output was printed
        debug_prints = [args[0][0] for args in mock_print.call_args_list
                        if f"No imports needed conversion to {target_format} format" in args[0][0]]
        assert len(debug_prints) > 0
        assert f"already_{target_format}.py" in debug_prints[0]

        # Verify file content is unchanged
        with open(py_file) as f:
            content = f.read()
            assert content.strip() == already_converted_content.strip()


@pytest.mark.parametrize("use_function", [use_standalone_pl, use_unified_pl])
def test_lightning_imports_no_files(tmp_path, use_function):
    """Test behavior with no files to process."""
    # Set up test directory
    src_dir = tmp_path / "empty_dir"
    src_dir.mkdir()

    # Case 1: Empty directory (no files)
    with patch('builtins.print'):
        use_function({src_dir}, debug=True)
        # No files to process, so no operations should happen

    # Case 2: Directory with only non-python files
    txt_file = src_dir / "test.txt"
    txt_file.write_text("This is not a Python file")

    with patch('builtins.print'):
        use_function({src_dir}, debug=True)
        # No Python files to process


def test_get_project_paths():
    """Test get_project_paths in different installation scenarios."""
    # Test scenario: Regular development environment
    with patch('os.path.dirname', side_effect=['dir1', 'dir2', 'dir3', 'dir4', 'project_root']), \
         patch('os.path.abspath', return_value='/path/to/file'), \
         patch.object(Path, 'exists', return_value=True):

        project_root, install_paths = get_project_paths()

        assert str(project_root).endswith('project_root')
        assert 'source' in install_paths
        assert 'tests' in install_paths
        assert 'require' in install_paths

    # Test scenario: Installed package (site-packages)
    with patch.object(Path, '__str__', return_value='/path/to/site-packages/pkg'), \
         patch.object(Path, 'exists', return_value=True):

        project_root, install_paths = get_project_paths()

        assert 'source' in install_paths
        assert 'examples' in install_paths


def test_is_package_installed():
    """Test _is_package_installed with successful and failed imports."""
    # Test with real packages instead of mocking to avoid conflicts with debugpy

    # Test with package that definitely exists
    assert _is_package_installed('os') is True

    # Test with package that definitely doesn't exist
    assert _is_package_installed('non_existent_random_package_name_12345') is False

    # Test with monkeypatch for more controlled testing
    original_import = __builtins__['__import__']

    try:
        # Create a custom import function that controls which packages "exist"
        def mock_import(name, *args, **kwargs):
            if name == 'test_package_exists':
                return object()  # Return a dummy object
            if name == 'test_package_missing':
                raise ImportError("Package not found")
            return original_import(name, *args, **kwargs)

        # Replace the built-in import function temporarily
        __builtins__['__import__'] = mock_import

        # Now test with our controlled mock
        assert _is_package_installed('test_package_exists') is True
        assert _is_package_installed('test_package_missing') is False

    finally:
        # Always restore the original import function
        __builtins__['__import__'] = original_import


def test_toggle_lightning_imports_package_checks():
    """Test toggle_lightning_imports handling when packages are not installed."""
    # Test when unified package is not installed
    with patch('finetuning_scheduler.dynamic_versioning.utils._is_package_installed', return_value=False), \
         patch('builtins.print') as mock_print:

        toggle_lightning_imports(mode="unified")

        warning_calls = [call for call in mock_print.call_args_list
                        if "Cannot toggle to unified imports" in call[0][0]]
        assert len(warning_calls) > 0

    # Test when standalone package is not installed
    with patch('finetuning_scheduler.dynamic_versioning.utils._is_package_installed',
              side_effect=lambda pkg: False if pkg == "pytorch_lightning" else True), \
         patch('builtins.print') as mock_print:

        toggle_lightning_imports(mode="standalone")

        warning_calls = [call for call in mock_print.call_args_list
                        if "Cannot toggle to standalone imports" in call[0][0]]
        assert len(warning_calls) > 0


def test_toggle_lightning_imports_exception_handling():
    """Test exception handling in toggle_lightning_imports."""
    with patch('finetuning_scheduler.dynamic_versioning.utils._is_package_installed', return_value=True), \
         patch('finetuning_scheduler.dynamic_versioning.utils.get_project_paths',
               side_effect=Exception("Test exception")), \
         patch('traceback.print_exc') as mock_traceback, \
         pytest.raises(RuntimeError, match="Failed to toggle Lightning imports"):

        toggle_lightning_imports(mode="unified")

        # Verify traceback was printed
        mock_traceback.assert_called_once()


def test_exclude_files_from_conversion():
    """Test excluding specific files from conversion process."""
    # Verify the EXCLUDE_FILES_FROM_CONVERSION list has updated paths
    assert "dynamic_versioning/utils.py" in EXCLUDE_FILES_FROM_CONVERSION
    assert "dynamic_versioning/toggle_lightning_mode.py" in EXCLUDE_FILES_FROM_CONVERSION
    assert "test_dynamic_versioning_utils.py" in EXCLUDE_FILES_FROM_CONVERSION


if __name__ == "__main__":
    sys.exit(pytest.main(["-xvs", __file__]))
