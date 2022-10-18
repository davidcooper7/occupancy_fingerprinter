"""A tool to generate grid-based binding site shapes."""

# Add imports here
from .occupancy_fingerprinter import *


from ._version import __version__

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
