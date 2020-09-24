# Configuration file for the Sphinx documentation builder.

import os
import sys


# -- Path setup --------------------------------------------------------------

# To find the vc2_bit_widths module
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "SMPTE ST 2042-1 (VC-2) Bit Widths"
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
    "sphinxcontrib.intertex",
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
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}


# -- Options for intertex ----------------------------------------------------

intertex_mapping = {
    "vc2_data_tables": "{vc2_data_tables}/../docs/build/latex/*.aux",
}

# While the other modules' documentation is not published publicly online,
# we'll use Intersphinx in the HTML too.
intertex_formats = ["html", "latex"]


# -- Options for HTML output -------------------------------------------------

html_theme = "nature"

html_static_path = ["_static"]


# -- Options for PDF output --------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
    # Add an 'Preface' chapter heading to the content which appears before all
    # of the main chapters.
    "tableofcontents": r"""
        \sphinxtableofcontents
        \chapter{Preface}
    """,
    # Make index entries smaller since some are quite long
    "printindex": r"\footnotesize\raggedright\printindex",
    # Override ToC depth to include sections
    "preamble": r"\setcounter{tocdepth}{1}",
}

# Show page numbers in references
latex_show_pagerefs = True

# Show hyperlink URLs in footnotes
latex_show_urls = "footnote"

# Divide the document into parts, then chapters, then sections
latex_toplevel_sectioning = "part"

# Don't include a module index (the main index should be sufficient)
latex_domain_indices = False
