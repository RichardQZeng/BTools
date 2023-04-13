#!/usr/bin/env python3
""" This file is intended to be hosting common functions for BERA Tools.
"""

# This script is part of the BERA Tools geospatial library.
# Author: Richard Zeng
# Created: 12/04/2023
# License: MIT

# constants

import os
import json

USE_MULTI_PROCESSING = True
USE_SCIPY_DISTANCE = True


def get_settings(setting_file):
    settings = {}
    if os.path.isfile(setting_file):
        # read the settings.json file if it exists
        with open(setting_file, 'r') as settings_file:
            gui_settings = json.load(settings_file)
    else:
        print("Settings.json not exist, creat one.")
        return None

    return settings

