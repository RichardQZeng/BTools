# -*- coding: utf-8 -*-

"""Top-level package for BERA Tools."""
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from bt_gui_main import *

__author__ = """AppliedGRG"""
__email__ = 'appliedgrg@gmail.com'
__version__ = '0.1'

name = 'gui'


def gui_main():
    runner()

