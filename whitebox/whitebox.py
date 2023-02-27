# -*- coding: utf-8 -*-

"""Main module."""

from .whitebox_tools import WhiteboxTools


def Runner():
    from .wb_runner import Runner
    Runner()


if __name__ == '__main__':
    Runner()
