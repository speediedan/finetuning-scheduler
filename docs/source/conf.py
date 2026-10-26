#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import glob
import os
import shutil
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import lightning.pytorch  # noqa: F401 workaround for TypeError on sphinx import

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..")
sys.path.insert(0, os.path.abspath(os.path.join(PATH_ROOT, "src")))


FOLDER_GENERATED = "generated"
SPHINX_MOCK_REQUIREMENTS = int(os.environ.get("SPHINX_MOCK_REQUIREMENTS", True))

spec = spec_from_file_location(
    "src/finetuning_scheduler/__about__.py", os.path.join(PATH_ROOT, "src", "finetuning_scheduler", "__about__.py")
)
about = module_from_spec(spec)
spec.loader.exec_module(about)


def _transform_changelog(path_in: str, path_out: str) -> None:
    with open(path_in) as fp:
        chlog_lines = fp.readlines()
    # enrich short subsub-titles to be unique
    chlog_ver = ""
    for i, ln in enumerate(chlog_lines):
        if ln.startswith("## "):
            chlog_ver = ln[2:].split("-")[0].strip()
        elif ln.startswith("### "):
            ln = ln.replace("###", f"### {chlog_ver} -")
            chlog_lines[i] = ln
    with open(path_out, "w") as fp:
        fp.writelines(chlog_lines)


os.makedirs(os.path.join(PATH_HERE, FOLDER_GENERATED), exist_ok=True)
# copy all documents from GH templates like contribution guide (exclude internal development docs)
for md in glob.glob(os.path.join(PATH_ROOT, ".github", "*.md")):
    # Skip copilot-instructions.md as it's an internal development document
    if os.path.basename(md) == "copilot-instructions.md":
        continue
    shutil.copy(md, os.path.join(PATH_HERE, FOLDER_GENERATED, os.path.basename(md)))
# copy also the changelog
_transform_changelog(os.path.join(PATH_ROOT, "CHANGELOG.md"), os.path.join(PATH_HERE, FOLDER_GENERATED, "CHANGELOG.md"))

# The tutorial notebooks, staged into `examples/notebooks/` (gitignored): the executed docs artifact where one exists, the
# published (output-free) copy otherwise, with the publisher's install cell removed (noise on a static page). The
# helpers are the docs-artifact renderer vendored from interpretune, loaded by file path; see pyproject's
# [tool.notebook-publishing] and scripts/render_notebook_docs.py.
NOTEBOOK_STAGE_DIR = os.path.join(PATH_HERE, "examples", "notebooks")
NOTEBOOK_SOURCES = [
    os.path.join(PATH_ROOT, "docs", "notebook_artifacts"),
    os.path.join(PATH_ROOT, "src", "fts_examples", "notebooks", "publish"),
]
NOTEBOOK_ASSET_SUFFIXES = {".ipynb", ".png", ".gif", ".jpg", ".jpeg", ".svg"}


def _stage_notebooks() -> None:
    engine = os.path.join(PATH_ROOT, "scripts", "vendored", "notebook_tooling", "notebook_docs.py")
    nb_spec = spec_from_file_location("_notebook_docs", engine)
    helpers = module_from_spec(nb_spec)
    sys.modules[nb_spec.name] = helpers  # dataclasses resolve `from __future__` annotations through sys.modules
    nb_spec.loader.exec_module(helpers)
    shutil.rmtree(NOTEBOOK_STAGE_DIR, ignore_errors=True)
    staged = set()
    for source_dir in NOTEBOOK_SOURCES:
        for dirpath, _, filenames in os.walk(source_dir):
            for name in sorted(filenames):
                rel = os.path.relpath(os.path.join(dirpath, name), source_dir)
                # notebooks and their images only: a local run leaves logs and checkpoints beside the notebook
                if name.startswith(".") or os.path.splitext(name)[1] not in NOTEBOOK_ASSET_SUFFIXES:
                    continue
                if rel in staged:  # an executed artifact wins over the published copy
                    continue
                target = os.path.join(NOTEBOOK_STAGE_DIR, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if name.endswith(".ipynb"):
                    notebook, _ = helpers.strip_docs_excluded_cells(helpers.load_notebook(Path(dirpath, name)))
                    helpers.save_notebook(notebook, Path(target))
                else:
                    shutil.copy2(os.path.join(dirpath, name), target)
                staged.add(rel)


_stage_notebooks()

# -- Project information -----------------------------------------------------

project = "Fine-Tuning Scheduler"
copyright = about.__copyright__
author = about.__author__

# The short X.Y version
version = about.__version__
# The full version, including alpha/beta/rc tags
release = about.__version__

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.

needs_sphinx = "6.2"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",  # keep intersphinx early so external+<proj> roles are registered before autodoc runs
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.imgmath",
    "sphinx.ext.autosectionlabel",
    "myst_nb",  # MyST markdown and the tutorial notebooks; loads myst_parser itself
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_paramlinks",
    "sphinx_togglebutton",
    "sphinx_design",
]

autodoc_typehints = "none"

# Suppress warnings about duplicate labels (needed for PL tutorials)
suppress_warnings = [
    "autosectionlabel.*",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Notebooks are never executed by the docs build: the executed copies are rendered offline, under the GPU lease, into
# docs/notebook_artifacts (scripts/render_notebook_docs.py).
nb_execution_mode = "off"
nb_merge_streams = True

# myst-parser, forcing to parse all html pages with mathjax
# https://github.com/executablebooks/MyST-Parser/issues/394
myst_update_mathjax = False

# MyST enables NO extensions by default, so `$x^2$` in a .md page renders as the literal characters
# until dollarmath is on. The mathjax handling above already assumes math is wanted; this is what
# actually delivers it.
# attrs_inline: the tutorial notebooks size their images with `![alt](img.png){height="55px" width="401px"}`.
myst_enable_extensions = ["dollarmath", "attrs_inline"]


# Jupyter-style heading anchors, so a notebook's in-page links (`[Basic Usage](#Basic-Usage)`) resolve in the docs as
# they do in Jupyter. An import path rather than a function, so Sphinx can cache the configuration.
sys.path.insert(0, os.path.join(PATH_HERE, "_ext"))
myst_heading_anchors = 3
myst_heading_slug_func = "notebook_anchors.jupyter_heading_slug"

# Harden against a prose `$` opening an equation that swallows the rest of the line. These stop a
# SPACED pair (`$HOME and $PATH`); they do not stop an ADJACENT one (`$TMPDIR/$SLURM_JOB_ID`), so they
# reduce the blast radius rather than closing it.
#
# The pages this governs are NOT the ones under docs/: every .md there is build-generated and
# gitignored. MyST only ever sees the copies made above from `.github/*.md` and `CHANGELOG.md`, and the
# markdown cells of the tutorial notebooks staged above, so those are the files to re-scan before adding
# math-adjacent prose; the .rst pages are not MyST-parsed at all. The first two scanned clean before enablement:
#   python .claude/skills/cross-platform-latex/scripts/check_math.py --collision-scan .github --strict
myst_dmath_allow_space = False
myst_dmath_allow_digits = False

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    f"{FOLDER_GENERATED}/PULL_REQUEST_TEMPLATE.md",
    # "notebooks/sample-template*",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

rst_prolog = """
.. include:: /_static/fts_cust_roles.rst
"""

# -- Options for HTML output -------------------------------------------------

# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/index.html
html_theme = "pydata_sphinx_theme"

# Read the Docs supplies the canonical URL of the version being built; local builds fall back to ``latest``.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", about.__docs_url__)

html_theme_options = {
    "github_url": "https://github.com/speediedan/finetuning-scheduler",
    # every page is a direct child of ``index``, so the header carries no section links; the full captioned toctree
    # is rendered in the primary sidebar instead (``_templates/sidebar-nav-full.html``)
    "navbar_center": [],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_nav_level": 1,
    "navigation_depth": 2,
    "collapse_navigation": False,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_sidebars = {"**": ["sidebar-nav-full"]}

html_logo = "_static/images/logos/logo_fts.svg"

html_favicon = "_static/images/logos/icon_fts.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# sphinx-copybutton: copy the code, not the ``>>>``/``...`` prompts or doctest output
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = project + "-doc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    # Latex figure (float) alignment
    "figure_align": "htbp"
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, project + ".tex", project + " Documentation", author, "manual")]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, project, project + " Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        project,
        project + " Documentation",
        author,
        project,
        "One line description of project.",
        "Miscellaneous",
    )
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("http://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pl": ("https://lightning.ai/docs/pytorch/stable/", None),
}

# -- Options for linkcheck builder -------------------------------------------

# PyTorch's documentation site renders API anchors client-side, so ``linkcheck`` cannot resolve
# fragments like ``#torch.nn.Module`` even though the pages themselves are reachable. Check that those
# URLs resolve, but skip their anchors -- otherwise every intersphinx-style torch reference reports as
# broken and drowns out real failures.
linkcheck_anchors_ignore_for_url = [
    r"https://pytorch\.org/docs/.*",
    r"https://docs\.pytorch\.org/docs/.*",
    # The Lightning docs are now a client-rendered app: every path, including a nonexistent one, returns the same
    # HTML shell, so no fragment on it can be resolved by ``linkcheck``.
    r"https://lightning\.ai/docs/.*",
]

# Stack Overflow answers ``linkcheck``'s requests with ``403 Forbidden`` (bot filtering) while the same URL redirects
# normally for a browser, so the result says nothing about whether the link is valid.
linkcheck_ignore = [
    r"https://stackoverflow\.com/.*",
]

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


def setup(app):
    app.add_css_file("main.css")
    app.add_css_file("fts_cust_roles.css")


# Ignoring Third-party packages
# https://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule
def package_list_from_file(file):
    """List up package name (not containing version and extras) from a package list file."""
    mocked_packages = []
    with open(file) as fp:
        for ln in fp.readlines():
            # Example: `tqdm>=4.41.0` => `tqdm`
            # `[` is for package with extras
            found = [ln.index(ch) for ch in list(",=<>#[") if ch in ln]
            pkg = ln[: min(found)] if found else ln
            if pkg.rstrip():
                mocked_packages.append(pkg.rstrip())
    return mocked_packages


# define mapping from PyPI names to python imports
PACKAGE_MAPPING = {
    "PyYAML": "yaml",
    "pyDeprecate": "deprecate",
}
# Packages to mock for autodoc (RTD doesn't install ML dependencies)
# These are the packages from pyproject.toml dependencies and optional-dependencies
MOCK_PACKAGES = []
if SPHINX_MOCK_REQUIREMENTS:
    # Base dependencies (torch + Lightning are imported directly)
    MOCK_PACKAGES += ["torch", "torchmetrics"]
    # Examples optional dependencies
    MOCK_PACKAGES += [
        "datasets",
        "evaluate",
        "transformers",
        "sklearn",
        "sentencepiece",
        "tensorboardX",
        "tabulate",
        "psutil",
    ]
    # Extra optional dependencies
    MOCK_PACKAGES += ["rich", "jsonargparse", "omegaconf", "hydra"]
MOCK_PACKAGES = [PACKAGE_MAPPING.get(pkg, pkg) for pkg in MOCK_PACKAGES]
autodoc_mock_imports = MOCK_PACKAGES

autosummary_generate = True
# document the names a module lists in ``__all__``, so a module re-exporting its implementation from elsewhere
# (``finetuning_scheduler.fts``, ``finetuning_scheduler.fts_supporters``) keeps its classes, summaries and anchors
autosummary_ignore_module_all = False

autodoc_member_order = "groupwise"

autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
}

# Sphinx will add “permalinks” for each heading and description environment as paragraph signs that
#  become visible when the mouse hovers over them.
# This value determines the text for the permalink; it defaults to "¶". Set it to None or the empty
#  string to disable permalinks.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_add_permalinks
html_permalinks = True
html_permalinks_icon = "¶"

# True to prefix each section label with the name of the document it is in, followed by a colon.
#  For example, index:Introduction for a section called Introduction that appears in document index.rst.
#  Useful for avoiding ambiguity when the same section heading appears in different documents.
# http://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
autosectionlabel_prefix_document = True

# only run doctests marked with a ".. doctest::" directive
doctest_test_doctest_blocks = ""
doctest_global_setup = """
import importlib
import os
import sys
from typing import Optional

import torch
import lightning.pytorch as pl
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import _JSONARGPARSE_SIGNATURES_AVAILABLE as _JSONARGPARSE_AVAILABLE
from lightning.pytorch.utilities import (
    _TORCHVISION_AVAILABLE,
)
"""
coverage_skip_undoc_in_source = True
