# -*- coding: utf-8 -*-

"""Top-level package for BERA Tools."""

__author__ = """AppliedGRG"""
__email__ = 'appliedgrg@gmail.com'
__version__ = '0.1'

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# from .BTools import *


