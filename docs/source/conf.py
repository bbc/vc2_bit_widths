# Configuration file for the Sphinx documentation builder.

import os
import sys


# -- Path setup --------------------------------------------------------------

# To find the vc2_bit_widths module
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "SMPTE ST 2042-2 (VC-2) Bit Widths"
copyright = "2019, SMPTE"
author = "SMPTE"

from vc2_bit_widths import __version__ as version
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinxcontrib.programoutput",
    "sphinxcontrib.inkscapeconverter",
]

# -- Options for numpydoc/autodoc --------------------------------------------

# Fixes autosummary errors
numpydoc_show_class_members = False

autodoc_member_order = "bysource"

add_module_names = False

autodoc_default_flags = [
    "members",
    "undoc-members",
]

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
}


# -- Options for HTML output -------------------------------------------------

html_theme = "nature"

html_static_path = ["_static"]


# -- Options for PDF output --------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
}
