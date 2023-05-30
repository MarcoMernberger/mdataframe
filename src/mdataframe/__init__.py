# -*- coding: utf-8 -*-
"""
Created on Aug 28, 2015

@author: mernberger
"""
__author__ = "MarcoMernberger"
__copyright__ = "MarcoMernberger"
__license__ = "mit"

from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


from .mdataframe import *
from .strategies import *
from .transformations import TMM, VST, _Transformer
from .filter import Filter
from .differential import *
