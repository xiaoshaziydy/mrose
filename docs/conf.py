"""Sphinx configuration for the mROSE Read the Docs site."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "mROSE"
author = "mROSE contributors"
copyright = "2026, mROSE contributors"
release = "0.1.0"

extensions = [
    "myst_nb",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_heading_anchors = 3
nb_execution_mode = "off"

html_theme = "sphinx_rtd_theme"
html_title = "mROSE documentation"
html_logo = "assets/mrose-icon.png"
html_static_path = []
html_theme_options = {
    "logo_only": False,
    "collapse_navigation": False,
}
