# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Preprocessing of information for Project information --------------------

with open("../../README.md", "r") as readme:
    long_description = readme.read()

version_string_start = long_description.find("**Version ") + 10
version_string_end = long_description.find("**", version_string_start)
version_string = long_description[version_string_start:version_string_end]

year = datetime.datetime.now().year

# -- Project information -----------------------------------------------------

master_doc = 'index'
project = 'Tensorflow-Neuroevolution Framework'
copyright = f'{year}, Paul Pauls'
author = 'Paul Pauls'

# The full version, including alpha/beta/rc tags
release = version_string

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Make version number variable available to documentation rst files
rst_epilog = """
.. |version_string_bold| replace:: **{version_string}**
""".format(version_string=version_string)
