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
import sys
import platform
import re
from subprocess import CalledProcessError, Popen, PIPE, STDOUT

from centerline import *

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
        self.verbose = True
        self.show_advanced = False
        self.__compress_rasters = False
        self.__max_procs = -1
        self.recent_tool = None

        self.setting_file = os.path.join(self.exe_path, '..\gui\settings.json')
        if os.path.isfile(self.setting_file):
            # read the settings.json file if it exists
            with open(self.setting_file, 'r') as settings_file:
                settings = json.load(settings_file)

            # parse file
            if 'working_directory' in settings.keys():
                self.work_dir = str(settings['working_directory'])
            if 'verbose_mode' in settings.keys():
                self.verbose = str(settings['verbose_mode'])
            if 'compress_rasters' in settings.keys():
                self.__compress_rasters = settings['compress_rasters']
            if 'max_procs' in settings.keys():
                self.__max_procs = settings['max_procs']
            if 'recent_tool' in settings.keys():
                self.recent_tool = settings['recent_tool']
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

    def set_working_dir(self, path_str):
        """ 
        Sets the working directory, i.e. the directory in which
        the data files are located. By setting the working 
        directory, tool input parameters that are files need only
        specify the file name rather than the complete file path.
        """
        self.work_dir = path.normpath(path_str)
        settings = {}

        if os.path.isfile(self.setting_file):
            # read the settings.json file if it exists
            with open(self.setting_file, 'r') as read_settings_file:
                settings = json.load(read_settings_file)
                if not settings:
                    settings = {}
        else:
            print("Settings.json not exist, creat one.")

        if self.work_dir:
            settings['working_directory'] = self.work_dir
        with open(self.setting_file, 'w') as write_settings_file:
            json.dump(settings, write_settings_file, indent=4)

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
        settings = {}

        if os.path.isfile(self.setting_file):
            # read the settings.json file if it exists
            with open(self.setting_file, 'r') as read_settings_file:
                settings = json.load(read_settings_file)

            if not settings:
                settings = {}
        else:
            print("Settings.json not exist, create one.")

        settings['verbose_mode'] = self.verbose
        with open(self.setting_file, 'w') as write_settings_file:
            json.dump(settings, write_settings_file, indent=4)

    def set_default_callback(self, callback_func):
        """
        Sets the default callback used for handling tool text outputs.
        """
        self.default_callback = callback_func


    def set_max_procs(self, val=-1):
        """ 
        Sets the flag used by BERA Tools to determine whether to use compression for output rasters.
        """
        self.__max_procs = val
        settings = {}
        if os.path.isfile(self.setting_file):
            # read the settings.json file if it exists
            with open(self.setting_file, 'r') as read_settings_file:
                settings = json.load(read_settings_file)

            if not settings:
                settings = {}
        else:
            print("Settings.json not exist, creat one.")

        if self.__max_procs:
            settings['max_procs'] = self.__max_procs
        with open(self.setting_file, 'w') as write_settings_file:
            json.dump(settings, write_settings_file, indent=4)

    def get_max_procs(self):
        return self.__max_procs

    def save_recent_tool(self):
        settings = {}
        if os.path.isfile(self.setting_file):
            # read the settings.json file if it exists
            with open(self.setting_file, 'r') as settings_file:
                settings = json.load(settings_file)
        else:
            print("Settings.json not exist, creat one.")

        if self.recent_tool and len(self.recent_tool) > 0:
            settings['recent_tool'] = self.recent_tool
            with open(self.setting_file, 'w') as settings_file:
                json.dump(settings, settings_file, indent=4)

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

            if self.verbose:
                cl = tool_name + " ".join(args)
                callback(cl.strip() + "\n")

            # current_tool = globals()[tool_name]
            # current_tool(callback, **args)

        except (OSError, ValueError, CalledProcessError) as err:
            callback(str(err))
            return 1
        finally:
            os.chdir(work_dir)

        # Call script using new process to make GUI responsive
        try:
            proc = None
            args_string = str(args).replace("'", '"')
            args_tool = ['python' , os.path.join(r'..\tools', tool_name+'.py'), '--input', args_string]

            if running_windows and self.start_minimized == True:
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

            return 'BERA Tools provide a series of tools developed by AppliedGRG lab.'

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

    def version(self):
        """ 
        Retrieves the version information for BERA Tools.
        """
        try:
            work_dir = os.getcwd()
            os.chdir(self.exe_path)
            args = []
            args.append("." + os.path.sep + self.exe_name)
            args.append("--version")

            proc = Popen(args, shell=False, stdout=PIPE,
                         stderr=STDOUT, bufsize=1, universal_newlines=True)
            ret = ""
            while True:
                line = proc.stdout.readline()
                if line != '':
                    ret += line
                else:
                    break

            return ret
        except (OSError, ValueError, CalledProcessError) as err:
            return err
        finally:
            os.chdir(work_dir)

    def tool_help(self, tool_name=''):
        """ 
        Retrieves the help description for a specific tool.
        """
        try:
            work_dir = os.getcwd()
            os.chdir(self.exe_path)
            args = []
            args.append("." + os.path.sep + self.exe_name)
            args.append("--toolhelp={}".format(to_camelcase(tool_name)))

            proc = Popen(args, shell=False, stdout=PIPE,
                         stderr=STDOUT, bufsize=1, universal_newlines=True)
            ret = ""
            while True:
                line = proc.stdout.readline()
                if line != '':
                    ret += line
                else:
                    break

            return ret
        except (OSError, ValueError, CalledProcessError) as err:
            return err
        finally:
            os.chdir(work_dir)

    def tool_parameters(self, tool_name):
        """ 
        Retrieves the tool parameter descriptions for a specific tool.
        """
        try:
            work_dir = os.getcwd()
            os.chdir(self.exe_path)
            args = []
            args.append("." + os.path.sep + self.exe_name)
            args.append("--toolparameters={}".format(to_camelcase(tool_name)))

            proc = Popen(args, shell=False, stdout=PIPE,
                         stderr=STDOUT, bufsize=1, universal_newlines=True)
            ret = ""
            while True:
                line = proc.stdout.readline()
                if line != '':
                    ret += line
                else:
                    break

            return ret
        except (OSError, ValueError, CalledProcessError) as err:
            return err
        finally:
            os.chdir(work_dir)

    def toolbox(self, tool_name=''):
        """ 
        Retrieve the toolbox for a specific tool.
        """
        try:
            work_dir = os.getcwd()
            os.chdir(self.exe_path)
            args = []
            args.append("." + os.path.sep + self.exe_name)
            args.append("--toolbox={}".format(to_camelcase(tool_name)))

            proc = Popen(args, shell=False, stdout=PIPE,
                         stderr=STDOUT, bufsize=1, universal_newlines=True)
            ret = ""
            while True:
                line = proc.stdout.readline()
                if line != '':
                    ret += line
                else:
                    break

            return ret
        except (OSError, ValueError, CalledProcessError) as err:
            return err
        finally:
            os.chdir(work_dir)

    def view_code(self, tool_name):
        """ 
        Opens a web browser to view the source code for a specific tool
        on the projects source code repository.
        """
        try:
            work_dir = os.getcwd()
            os.chdir(self.exe_path)
            args = []
            args.append("." + os.path.sep + self.exe_name)
            args.append("--viewcode={}".format(to_camelcase(tool_name)))

            proc = Popen(args, shell=False, stdout=PIPE,
                         stderr=STDOUT, bufsize=1, universal_newlines=True)
            ret = ""
            while True:
                line = proc.stdout.readline()
                if line != '':
                    ret += line
                else:
                    break

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

    ########################################################################
    # The following methods are convenience methods for each available tool.
    # This needs updating whenever new tools are added to the library.
    ########################################################################
    def convert_raster_format(self, i, output, callback=None):
        """Converts raster data from one format to another.

        Keyword arguments:

        i -- Input raster file.
        output -- Output raster file.
        callback -- Custom function for handling tool text outputs.
        """
        args = []
        args.append("--input='{}'".format(i))
        args.append("--output='{}'".format(output))
        return self.run_tool('convert_raster_format', args, callback)  # returns 1 if error

    def export_table_to_csv(self, i, output, headers=True, callback=None):
        """Exports an attribute table to a CSV text file.

        Keyword arguments:

        i -- Input vector file.
        output -- Output csv file.
        headers -- Export field names as file header?.
        callback -- Custom function for handling tool text outputs.
        """
        args = []
        args.append("--input='{}'".format(i))
        args.append("--output='{}'".format(output))
        if headers: args.append("--headers")
        return self.run_tool('export_table_to_csv', args, callback)  # returns 1 if error
