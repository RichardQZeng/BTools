#!/usr/bin/env python3
""" This file is intended to be a helper for running BERA tools from a Python script.
"""

# This script is part of the BERA Tools geospatial library.
# Original Authors: Dr. John Lindsay
# Created: 28/11/2017
# Modified: 23/03/2023
# Author: Richard Zeng
# License: MIT

from __future__ import print_function
import os
from os import path
from pathlib import Path
import sys
import platform
import re
import json
import multiprocessing
from subprocess import CalledProcessError, Popen, PIPE, STDOUT

from common import *

running_windows = platform.system() == 'Windows'
if running_windows:
    from subprocess import STARTUPINFO, STARTF_USESHOWWINDOW


def default_callback(value):
    """ 
    A simple default callback that outputs using the print function. When
    tools are called without providing a custom callback, this function
    will be used to print to standard output.
    """
    print(value)


def to_camelcase(name):
    """
    Convert snake_case name to CamelCase name 
    """
    return ''.join(x.title() for x in name.split('_'))


def to_snakecase(name):
    """
    Convert CamelCase name to snake_case name
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class BeraTools(object):
    """ 
    An object for interfacing with the BERA Tools executable.
    """

    def __init__(self):
        if running_windows:
            self.ext = '.exe'
        else:
            self.ext = ''
        self.exe_name = "BERA_tools{}".format(self.ext)
        # self.exe_path = os.path.dirname(shutil.which(
        #     self.exe_name) or path.dirname(path.abspath(__file__)))
        # self.exe_path = os.path.dirname(os.path.join(os.path.realpath(__file__)))
        self.exe_path = path.dirname(path.abspath(__file__))

        self.work_dir = ""
        self.verbose = False
        self.show_advanced = False
        self.__compress_rasters = False
        self.__max_procs = -1
        self.recent_tool = None
        self.ascii_art = None

        # set maximum available cpu core for tools
        self.__max_cpu_cores = min(BT_MAXIMUM_CPU_CORES, multiprocessing.cpu_count())

        # load bera tools
        self.tools_list = []
        self.sorted_tools = []
        self.get_bera_tools()
        self.get_bera_tool_list()

        self.setting_file = os.path.join(self.exe_path, '..\..\.data\saved_tool_parameters.json')
        if os.path.isfile(self.setting_file):
            # read the settings.json file if it exists
            with open(self.setting_file, 'r') as settings_file:
                settings = json.load(settings_file)

            # parse file
            if 'gui_parameters' in settings.keys():
                gui_settings = settings['gui_parameters']
                if 'working_directory' in gui_settings.keys():
                    self.work_dir = str(gui_settings['working_directory'])
                if 'verbose_mode' in gui_settings.keys():
                    self.verbose = str(gui_settings['verbose_mode'])
                if 'compress_rasters' in gui_settings.keys():
                    self.__compress_rasters = gui_settings['compress_rasters']
                if 'max_procs' in gui_settings.keys():
                    self.__max_procs = gui_settings['max_procs']
                if 'recent_tool' in gui_settings.keys():
                    self.recent_tool = gui_settings['recent_tool']
        else:
            print("Settings.json not exist.")

        self.gui_setting_file = os.path.join(self.exe_path, '..\gui\gui.json')
        if os.path.isfile(self.gui_setting_file):
            # read the settings.json file if it exists
            with open(self.gui_setting_file, 'r') as gui_setting_file:
                gui_settings = json.load(gui_setting_file)

            # parse file
            if 'ascii_art' in gui_settings.keys():
                bera_art = ''
                for line_of_art in gui_settings['ascii_art']:
                    bera_art += line_of_art
                self.ascii_art = bera_art
        else:
            print("Settings.json not exist.")

        self.cancel_op = False
        self.default_callback = default_callback
        self.start_minimized = False

    def set_bera_dir(self, path_str):
        """ 
        Sets the directory to the BERA Tools executable file.
        """
        self.exe_path = path_str

    def save_setting(self, key, value):
        settings = {}

        if os.path.isfile(self.setting_file):
            # read the settings.json file if it exists
            with open(self.setting_file, 'r') as read_settings_file:
                settings = json.load(read_settings_file)

            if not settings:
                settings = {}
        else:
            print("Settings file not exist, creating one.")

        if value:
            if 'gui_parameters' not in settings.keys():
                settings['gui_parameters'] = {}

            settings['gui_parameters'][key] = value

            with open(self.setting_file, 'w') as write_settings_file:
                json.dump(settings, write_settings_file, indent=4)

    def set_working_dir(self, path_str):
        """ 
        Sets the working directory, i.e. the directory in which
        the data files are located. By setting the working 
        directory, tool input parameters that are files need only
        specify the file name rather than the complete file path.
        """
        self.work_dir = path.normpath(path_str)
        self.save_setting('working_directory', self.work_dir)

    def get_working_dir(self):
        return self.work_dir

    def get_verbose_mode(self):
        return self.verbose

    def set_verbose_mode(self, val=True):
        """ 
        Sets verbose mode. If verbose mode is False, tools will not
        print output messages. Tools will frequently provide substantial
        feedback while they are operating, e.g. updating progress for 
        various sub-routines. When the user has scripted a workflow
        that ties many tools in sequence, this level of tool output
        can be problematic. By setting verbose mode to False, these
        messages are suppressed and tools run as background processes.
        """
        self.verbose = val
        self.save_setting('verbose_mode', val)


    def set_max_procs(self, val=-1):
        """ 
        Sets the flag used by BERA Tools to determine whether to use compression for output rasters.
        """
        self.__max_procs = val

        self.save_setting('max_procs', val)

    def get_max_procs(self):
        return self.__max_procs

    def get_max_cpu_cores(self):
        return self.__max_cpu_cores

    def save_recent_tool(self):
        gui_settings = {}
        if os.path.isfile(self.setting_file):
            # read the settings.json file if it exists
            with open(self.setting_file, 'r') as settings_file:
                gui_settings = json.load(settings_file)
        else:
            print("Settings.json not exist, creat one.")

        if self.recent_tool and len(self.recent_tool) > 0:
            if 'gui_parameters' not in gui_settings.keys():
                gui_settings['gui_parameters'] = {}

            gui_settings['gui_parameters']['recent_tool'] = self.recent_tool
            with open(self.setting_file, 'w') as settings_file:
                json.dump(gui_settings, settings_file, indent=4)

    def run_tool(self, tool_name, args, callback=None):
        """ 
        Runs a tool and specifies tool arguments.
        Returns 0 if completes without error.
        Returns 1 if error encountered (details are sent to callback).
        Returns 2 if process is cancelled by user.
        """

        try:
            if callback is None:
                callback = self.default_callback

            work_dir = os.getcwd()
            os.chdir(self.exe_path)

        except (OSError, ValueError, CalledProcessError) as err:
            callback(str(err))
            return 1
        finally:
            os.chdir(work_dir)

        # Call script using new process to make GUI responsive
        try:
            proc = None
            args_string = str(args).replace("'", '"')
            args_tool = ['python', os.path.join(r'..\tools', tool_name+'.py'),
                         '-i', args_string, '-p', str(self.get_max_procs()), '-v', str(self.verbose)]

            if running_windows and self.start_minimized:
                si = STARTUPINFO()
                si.dwFlags = STARTF_USESHOWWINDOW
                si.wShowWindow = 7  # Set window minimized and not activated
                proc = Popen(args_tool, shell=False, stdout=PIPE,
                             stderr=STDOUT, bufsize=1, universal_newlines=True,
                             startupinfo=si)
            else:
                proc = Popen(args_tool, shell=False, stdout=PIPE,
                             stderr=STDOUT, bufsize=1, universal_newlines=True)

            while proc is not None:
                line = proc.stdout.readline()
                sys.stdout.flush()
                if line != '':
                    if not self.cancel_op:
                        callback(line.strip())
                    else:
                        self.cancel_op = False
                        proc.terminate()
                        callback('Tool operation terminated.')
                        return 2

                else:
                    break

            callback('---------------------------')
            callback('{} tool finished'.format(tool_name))
            callback('---------------------------')

            return 0
        except (OSError, ValueError, CalledProcessError) as err:
            callback(str(err))
            return 1

    def about(self):
        """ 
        Retrieves the help description for BERA Tools.
        """
        try:
            work_dir = os.getcwd()
            os.chdir(self.exe_path)

            about_text = 'BERA Tools provide a series of tools developed by AppliedGRG lab.\n\n'
            about_text += self.ascii_art

            return about_text

        except (OSError, ValueError, CalledProcessError) as err:
            return err
        finally:
            os.chdir(work_dir)

    def license(self, tool_name=None):
        """ 
        Retrieves the license information for BERA Tools.
        """

        work_dir = os.getcwd()
        os.chdir(self.exe_path)

        try:
            with open(os.path.join(self.exe_path, r'..\..\LICENSE.txt'), 'r') as f:
                ret = f.read()

            return ret
        except (OSError, ValueError, CalledProcessError) as err:
            return err
        finally:
            os.chdir(work_dir)

    def list_tools(self, keywords=[]):
        """ 
        Lists all available tools in BERA Tools.
        """
        try:
            work_dir = os.getcwd()
            os.chdir(self.exe_path)
            args = []
            args.append("." + os.path.sep + self.exe_name)
            args.append("--listtools")
            if len(keywords) > 0:
                for kw in keywords:
                    args.append(kw)

            proc = Popen(args, shell=False, stdout=PIPE,
                         stderr=STDOUT, bufsize=1, universal_newlines=True)
            ret = {}
            line = proc.stdout.readline()  # skip number of available tools header
            while True:
                line = proc.stdout.readline()
                if line != '':
                    if line.strip() != '':
                        name, descr = line.split(':')
                        ret[to_snakecase(name.strip())] = descr.strip()
                else:
                    break

            return ret
        except (OSError, ValueError, CalledProcessError) as err:
            return err
        finally:
            os.chdir(work_dir)

    def get_saved_tool_parameter(self, tool, variable):
        data_path = Path(__file__).resolve().cwd().parent.parent.joinpath(r'.data')
        if not data_path.exists():
            data_path.mkdir()

        json_file = data_path.joinpath(data_path, 'saved_tool_parameters.json')
        if json_file.exists():
            with open(json_file) as open_file:
                saved_parameters = json.load(open_file)
                if tool in list(saved_parameters.keys()):
                    tool_params = saved_parameters[tool]
                    if variable in tool_params.keys():
                        saved_value = tool_params[variable]
                        return saved_value

        return None

    def get_bera_tools(self):
        tool_json = os.path.join(self.exe_path, r'beratools.json')
        if os.path.exists(tool_json):
            tool_json = open(os.path.join(self.exe_path, r'beratools.json'))
            self.bera_tools = json.load(tool_json)
        else:
            print('Tool configuration file not exists')

    def get_bera_tool_list(self):
        self.tools_list = []
        self.sorted_tools = []
        selected_item = -1
        for toolbox in self.bera_tools['toolbox']:
            category = []
            for item in toolbox['tools']:
                if item['name']:
                    category.append(item['name'])
                    self.tools_list.append(item['name'])  # add tool to list

            self.sorted_tools.append(category)
