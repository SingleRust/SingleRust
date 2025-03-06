import os
import sys
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------
project = 'SingleRust'
copyright = '2025, Ian F. Diks'
author = 'Ian F. Diks and contributors'
release = '0.2.1-alpha.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/singlerust_logo.png'  # Add a logo if you have one
html_favicon = '_static/favicon.ico'  # Add a favicon if you have one

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

# -- Options for MyST parser -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# -- Options for autodoc extension -------------------------------------------
autodoc_member_order = 'bysource'