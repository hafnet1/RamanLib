import os, sys
# Make your package importable for autodoc:
# Adjust this if your package lives at top level or under src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
]

# --- basics ---
project = "RamanLib"
author = "Thomas Hafner"
html_title = "RamanLib Documentation"   # title in the browser/tab
html_short_title = "RamanLib"           # left nav top-left title

html_theme = "sphinx_rtd_theme"

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
