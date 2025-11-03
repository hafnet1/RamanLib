import os, sys
# Make your package importable for autodoc:
# Adjust this if your package lives at top level or under src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx.ext.intersphinx",
]

# --- basics ---
project = "RamanLib"
author = "Thomas Hafner"
html_title = "RamanLib Documentation"   # title in the browser/tab
html_short_title = "RamanLib"           # left nav top-left title

# GitHub buttons in the header & footer
html_theme_options = {
    "source_repository": "https://github.com/hafnet1/ramanlib/",
    "source_branch": "main",          # or your default branch
    "source_directory": "docs/",
    "navigation_depth": 4,
    "collapse_navigation": False,
    "style_nav_header_background": "#2980B9",  # nice blue header (optional)
}

# Use the classic Read the Docs theme
html_theme = "sphinx_rtd_theme"
html_title = "RamanLib Documentation"
html_short_title = "RamanLib"

# NumPy docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Autosummary: generate pages for functions/classes automatically
autosummary_generate = True

# nbsphinx settings
nbsphinx_execute = "never"

# Sensible autodoc defaults
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "exclude-members": "__weakref__",
}

# Intersphinx: link to other projects' documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "ramanspy": ("https://ramanspy.readthedocs.io/en/latest/", None),
}