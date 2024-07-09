#!/usr/bin/env python3
""" This file is intended to be a helper for running BERA tools from a Python script.
"""

# This script is part of the BERA Tools geospatial library.
# Created: 23/03/2023
# Author: Richard Zeng
# License: MIT

import os
from os import path
from pathlib import Path

import platform
import json
from json.decoder import JSONDecodeError
import multiprocessing
from subprocess import CalledProcessError
from collections import OrderedDict

from beratools.tools.common import *
from beratools.core.constants import *

running_windows = platform.system() == 'Windows'


def default_callback(value):
    """ 
    A simple default callback that outputs using the print function. When
    tools are called without providing a custom callback, this function
    will be used to print to standard output.
    """
    print(value)


class BTData(object):
    """ 
    An object for interfacing with the BERA Tools executable.
    """

    def __init__(self):
        if running_windows:
            self.ext = '.exe'
        else:
            self.ext = ''
        self.exe_name = "BERA_tools{}".format(self.ext)
        self.exe_path = path.dirname(path.abspath(__file__))

        self.work_dir = ""
        self.verbose = False
        self.show_advanced = BT_SHOW_ADVANCED_OPTIONS
        self.max_procs = -1
        self.recent_tool = None
        self.ascii_art = None

        # set maximum available cpu core for tools
        self.max_cpu_cores = min(BT_MAXIMUM_CPU_CORES, multiprocessing.cpu_count())

        # load bera tools
        self.tool_history = []
        self.settings = {}
        self.bera_tools = None
        self.tools_list = []
        self.sorted_tools = []
        self.upper_toolboxes = []
        self.lower_toolboxes = []
        self.get_bera_tools()
        self.get_bera_tool_list()
        self.get_bera_toolboxes()
        self.toolbox_list = self.get_bera_toolboxes()
        self.sort_toolboxes()

        self.setting_file = Path(self.exe_path).joinpath(r'.data\saved_tool_parameters.json')
        self.gui_setting_file = Path(self.exe_path).joinpath(r'gui.json')

        self.load_saved_tool_info()
        self.load_gui_data()
        self.get_tool_history()

        self.default_callback = default_callback
        self.start_minimized = False

    def set_bera_dir(self, path_str):
        """ 
        Sets the directory to the BERA Tools executable file.
        """
        self.exe_path = path_str

    def add_tool_history(self, tool, params):
        if 'tool_history' not in self.settings:
            self.settings['tool_history'] = OrderedDict()

        self.settings['tool_history'][tool] = params
        self.settings['tool_history'].move_to_end(tool, last=False)

    def save_tool_info(self):
        if self.recent_tool:
            if 'gui_parameters' not in self.settings.keys():
                self.settings['gui_parameters'] = {}

            self.settings['gui_parameters']['recent_tool'] = self.recent_tool

        with open(self.setting_file, 'w') as file_setting:
            try:
                json.dump(self.settings, file_setting, indent=4)
            except JSONDecodeError:
                pass

    def save_setting(self, key, value):
        # check setting directory existence
        data_path = Path(self.setting_file).resolve().parent
        if not data_path.exists():
            data_path.mkdir()

        self.load_saved_tool_info()

        if value is not None:
            if 'gui_parameters' not in self.settings.keys():
                self.settings['gui_parameters'] = {}

            self.settings['gui_parameters'][key] = value

            with open(self.setting_file, 'w') as write_settings_file:
                json.dump(self.settings, write_settings_file, indent=4)

    def set_working_dir(self, path_str):
        self.work_dir = path.normpath(path_str)
        self.save_setting('working_directory', self.work_dir)

    def get_working_dir(self):
        return self.work_dir

    def get_data_folder(self):
        data_folder = Path(self.setting_file).parent
        if not data_folder.exists():
            data_folder.mkdir()

        return data_folder.as_posix()

    def get_verbose_mode(self):
        return self.verbose

    def set_verbose_mode(self, val=True):
        self.verbose = val
        self.save_setting('verbose_mode', val)

    def set_max_procs(self, val=-1):
        """ 
        Sets the flag used by BERA Tools to determine whether to use compression for output rasters.
        """
        self.max_procs = val
        self.save_setting('max_procs', val)

    def get_max_procs(self):
        return self.max_procs

    def get_max_cpu_cores(self):
        return self.max_cpu_cores

    def run_tool(self, tool_api, args, callback=None, verbose=True):
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
        except Exception as err:
            callback(str(err))
            return 1
        finally:
            os.chdir(work_dir)

        # Call script using new process to make GUI responsive
        try:
            proc = None

            # convert to valid json string
            args_string = str(args).replace("'", '"')
            args_string = args_string.replace('True', 'true')
            args_string = args_string.replace('False', 'false')

            tool_name = self.get_bera_tool_name(tool_api)
            tool_type = self.get_bera_tool_type(tool_name)
            tool_args = None

            if tool_type == 'python':
                tool_args = [Path(work_dir).joinpath(f'beratools/tools/{tool_api}.py').as_posix(),
                             '-i', args_string, '-p', str(self.get_max_procs()),
                             '-v', str(self.verbose)]
            elif tool_type == 'executable':
                print(globals().get(tool_api))
                tool_args = globals()[tool_api](args_string)
                work_dir = os.getcwd()
                lapis_path = Path(work_dir).parent.joinpath('./third_party/Lapis_0_8')
                os.chdir(lapis_path.as_posix())
        except Exception as err:
            callback(str(err))
            return 1

        return tool_type, tool_args

    def about(self):
        """ 
        Retrieves the help description for BERA Tools.
        """
        work_dir = None
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

    def license(self):
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

    def load_saved_tool_info(self):
        data_path = Path(self.setting_file).parent
        if not data_path.exists():
            data_path.mkdir()

        saved_parameters = {}
        json_file = Path(self.setting_file)
        if json_file.exists():
            with open(json_file) as open_file:
                try:
                    saved_parameters = json.load(open_file, object_pairs_hook=OrderedDict)
                except json.decoder.JSONDecodeError:
                    pass

        self.settings = saved_parameters

        # parse file
        if 'gui_parameters' in self.settings.keys():
            gui_settings = self.settings['gui_parameters']

            if 'working_directory' in gui_settings.keys():
                self.work_dir = str(gui_settings['working_directory'])
            if 'verbose_mode' in gui_settings.keys():
                self.verbose = str(gui_settings['verbose_mode'])
            else:
                self.verbose = False

            if 'max_procs' in gui_settings.keys():
                self.max_procs = gui_settings['max_procs']

            if 'recent_tool' in gui_settings.keys():
                self.recent_tool = gui_settings['recent_tool']
                if not self.get_bera_tool_api(self.recent_tool):
                    self.recent_tool = None

    def load_gui_data(self):
        gui_settings = {}
        if not self.gui_setting_file.exists():
            print("gui.json not exist.")
        else:
            # read the settings.json file if it exists
            with open(self.gui_setting_file, 'r') as file_gui:
                try:
                    gui_settings = json.load(file_gui)
                except json.decoder.JSONDecodeError:
                    pass

            # parse file
            if 'ascii_art' in gui_settings.keys():
                bera_art = ''
                for line_of_art in gui_settings['ascii_art']:
                    bera_art += line_of_art
                self.ascii_art = bera_art

    def get_tool_history(self):
        tool_history = []
        self.load_saved_tool_info()
        if self.settings:
            if 'tool_history' in self.settings:
                tool_history = self.settings['tool_history']

        if tool_history:
            self.tool_history = []
            for item in tool_history:
                item = self.get_bera_tool_name(item)
                self.tool_history.append(item)

    def get_saved_tool_params(self, tool_api, variable=None):
        self.load_saved_tool_info()

        if 'tool_history' in self.settings:
            if tool_api in list(self.settings['tool_history']):
                tool_params = self.settings['tool_history'][tool_api]
                if tool_params:
                    if variable:
                        if variable in tool_params.keys():
                            saved_value = tool_params[variable]
                            return saved_value
                    else:  # return all params
                        return tool_params

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

        for toolbox in self.bera_tools['toolbox']:
            category = []
            for item in toolbox['tools']:
                if item['name']:
                    category.append(item['name'])
                    self.tools_list.append(item['name'])  # add tool to list

            self.sorted_tools.append(category)

    def sort_toolboxes(self):
        for toolbox in self.toolbox_list:
            if toolbox.find('/') == (-1):  # Does not contain a sub toolbox, i.e. does not contain '/'
                self.upper_toolboxes.append(toolbox)  # add to both upper toolbox list and lower toolbox list
                self.lower_toolboxes.append(toolbox)
            else:  # Contains a sub toolbox
                self.lower_toolboxes.append(toolbox)  # add to only the lower toolbox list

    def get_bera_toolboxes(self):
        toolboxes = []
        for toolbox in self.bera_tools['toolbox']:
            tb = toolbox['category']
            toolboxes.append(tb)
        return toolboxes

    def get_bera_tool_params(self, tool_name):
        new_params = {'parameters': []}
        tool = {}
        batch_tool_list = []
        for toolbox in self.bera_tools['toolbox']:
            for single_tool in toolbox['tools']:
                if single_tool['batch_processing']:
                    batch_tool_list.append(single_tool['name'])

                if tool_name == single_tool['name']:
                    tool = single_tool

        for key, value in tool.items():
            if key != 'parameters':
                new_params[key] = value

        # convert json format for parameters
        if 'parameters' not in tool.keys():
            print('issue')

        for param in tool['parameters']:
            new_param = {'name': param['parameter']}
            if 'variable' in param.keys():
                new_param['flag'] = param['variable']
                # restore saved parameters
                saved_value = self.get_saved_tool_params(tool['tool_api'], param['variable'])
                if saved_value is not None:
                    new_param['saved_value'] = saved_value
            else:
                new_param['flag'] = 'FIXME'

            if not param['output']:
                if param['type'] == 'list':
                    if tool_name == 'Batch Processing':
                        new_param['parameter_type'] = {'OptionList': batch_tool_list}
                        new_param['data_type'] = 'String'
                    else:
                        new_param['parameter_type'] = {'OptionList': param['data']}
                        new_param['data_type'] = 'String'
                        if param['typelab'] == 'text':
                            new_param['data_type'] = 'String'
                        elif param['typelab'] == 'int':
                            new_param['data_type'] = 'Integer'
                        elif param['typelab'] == 'float':
                            new_param['data_type'] = 'Float'
                        elif param['typelab'] == 'bool':
                            new_param['data_type'] = 'Boolean'
                elif param['type'] == 'text':
                    new_param['parameter_type'] = 'String'
                elif param['type'] == 'number':
                    if param['typelab'] == 'int':
                        new_param['parameter_type'] = 'Integer'
                    else:
                        new_param['parameter_type'] = 'Float'
                elif param['type'] == 'file':
                    new_param['parameter_type'] = {'ExistingFile': [param['typelab']]}
                else:
                    new_param['parameter_type'] = {'ExistingFile': ''}
            else:
                new_param["parameter_type"] = {'NewFile': [param['typelab']]}

            new_param['description'] = param['description']

            if param['type'] == 'raster':
                for i in new_param["parameter_type"].keys():
                    new_param['parameter_type'][i] = 'Raster'
            elif param['type'] == 'lidar':
                for i in new_param["parameter_type"].keys():
                    new_param['parameter_type'][i] = 'Lidar'
            elif param['type'] == 'vector':
                for i in new_param["parameter_type"].keys():
                    new_param['parameter_type'][i] = 'Vector'
            elif param['type'] == 'Directory':
                new_param['parameter_type'] = {'Directory': [param['typelab']]}

            new_param['default_value'] = param['default']
            if "optional" in param.keys():
                new_param['optional'] = param['optional']
            else:
                new_param['optional'] = False

            new_params['parameters'].append(new_param)

        return new_params

    def get_bera_tool_parameters_list(self, tool_name):
        params = self.get_bera_tool_params(tool_name)
        param_list = {}
        for item in params['parameters']:
            param_list[item['flag']] = item['default_value']

        return param_list

    def get_bera_tool_args(self, tool_name):
        params = self.get_bera_tool_params(tool_name)
        tool_args = params['parameters']

        return tool_args

    def get_bera_tool_name(self, tool_api):
        tool_name = None
        for toolbox in self.bera_tools['toolbox']:
            for tool in toolbox['tools']:
                if tool_api == tool['tool_api']:
                    tool_name = tool['name']

        return tool_name

    def get_bera_tool_api(self, tool_name):
        tool_api = None
        for toolbox in self.bera_tools['toolbox']:
            for tool in toolbox['tools']:
                if tool_name == tool['name']:
                    tool_api = tool['tool_api']

        return tool_api

    def get_bera_tool_type(self, tool_name):
        tool_type = None
        for toolbox in self.bera_tools['toolbox']:
            for tool in toolbox['tools']:
                if tool_name == tool['name']:
                    tool_type = tool['tool_type']

        return tool_type
