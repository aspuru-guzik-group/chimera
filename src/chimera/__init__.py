#!/usr/bin/env python

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .chimera import Chimera
