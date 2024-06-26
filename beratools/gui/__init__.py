# -*- coding: utf-8 -*-

"""Top-level package for BERA Tools."""
import os
import sys
import inspect
from .bt_gui_main import *

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

__author__ = """AppliedGRG"""
__email__ = 'appliedgrg@gmail.com'
__version__ = '0.1'

name = 'gui'


def gui_main():
    runner()

