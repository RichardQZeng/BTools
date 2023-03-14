#!/usr/bin/env python3

# This script is part of the BERA Tools.
# Authors: Richard Zeng
# Created: 01/03/2023
# License: MIT

import __future__
import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import json
import os
from os import path
import platform
import re  # Added by Rachel for snake_to_camel function
import glob
from sys import platform as _platform

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
from tkinter.simpledialog import askinteger
from tkinter import messagebox
from tkinter import PhotoImage
import webbrowser
from PIL import Image, ImageTk
import multiprocessing

from ..tools.beratools import BeraTools, to_camelcase

bt = BeraTools()


class FileSelector(tk.Frame):
    def __init__(self, json_str, runner, master=None):
        # first make sure that the json data has the correct fields
        j = json.loads(json_str)
        self.name = j['name']
        self.description = j['description']
        self.flag = j['flags']
        self.parameter_type = j['parameter_type']
        self.file_type = ""
        if "ExistingFile" in self.parameter_type:
            self.file_type = j['parameter_type']['ExistingFile']
        elif "NewFile" in self.parameter_type:
            self.file_type = j['parameter_type']['NewFile']
        self.optional = j['optional']
        default_value = j['default_value']

        self.runner = runner

        ttk.Frame.__init__(self, master, padding='0.02i')
        self.grid()

        self.label = ttk.Label(self, text=self.name, justify=tk.LEFT)
        self.label.grid(row=0, column=0, sticky=tk.W)
        self.label.columnconfigure(0, weight=1)

        if not self.optional:
            self.label['text'] = self.label['text'] + "*"

        fs_frame = ttk.Frame(self, padding='0.0i')
        self.value = tk.StringVar()
        self.entry = ttk.Entry(
            fs_frame, width=45, justify=tk.LEFT, textvariable=self.value)
        self.entry.grid(row=0, column=0, sticky=tk.NSEW)
        self.entry.columnconfigure(0, weight=1)
        if default_value:
            self.value.set(default_value)

        self.open_button = ttk.Button(fs_frame, width=4, text="...", command=self.select_file, padding='0.02i')
        self.open_button.grid(row=0, column=1, sticky=tk.E)
        self.open_button.columnconfigure(0, weight=1)
        fs_frame.grid(row=1, column=0, sticky=tk.NSEW)
        fs_frame.columnconfigure(0, weight=10)
        fs_frame.columnconfigure(1, weight=1)
        # self.pack(fill=tk.BOTH, expand=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Add the bindings
        if _platform == "darwin":
            self.entry.bind("<Command-Key-a>", self.select_all)
        else:
            self.entry.bind("<Control-Key-a>", self.select_all)

    def select_file(self):
        try:
            result = self.value.get()
            if self.parameter_type == "Directory":
                result = filedialog.askdirectory(initialdir=self.runner.working_dir, title="Select directory")
            elif "ExistingFile" in self.parameter_type:
                file_types = [('All files', '*.*')]
                if 'RasterAndVector' in self.file_type:
                    file_types = [("Shapefiles", "*.shp"), ('Raster files', ('*.dep', '*.tif', '*.tiff',
                                                                             '*.bil', '*.flt', '*.sdat',
                                                                             '*.rdc', '*.asc', '*grd'))]
                elif 'Raster' in self.file_type:
                    file_types = [('Raster files', ('*.dep', '*.tif', '*.tiff', '*.bil', '*.flt',
                                                    '*.sdat', '*.rdc', '*.asc', '*.grd'))]
                elif 'Lidar' in self.file_type:
                    file_types = [("LiDAR files", ('*.las', '*.zlidar', '*.laz', '*.zip'))]
                elif 'Vector' in self.file_type:
                    file_types = [("Shapefiles", "*.shp")]
                elif 'Text' in self.file_type:
                    file_types = [("Text files", "*.txt"), ("all files", "*.*")]
                elif 'Csv' in self.file_type:
                    file_types = [("CSC files", "*.csv"), ("all files", "*.*")]
                elif 'Dat' in self.file_type:
                    file_types = [("Binary data files", "*.dat"), ("all files", "*.*")]
                elif 'Html' in self.file_type:
                    file_types = [("HTML files", "*.html")]

                result = filedialog.askopenfilename(
                    initialdir=self.runner.working_dir, title="Select file", filetypes=file_types)

            elif "NewFile" in self.parameter_type:
                result = filedialog.asksaveasfilename()

            self.value.set(result)
            # update the working directory
            self.runner.working_dir = os.path.dirname(result)

        except:
            t = "file"
            if self.parameter_type == "Directory":
                t = "directory"
            messagebox.showinfo("Warning", "Could not find {}".format(t))

    def get_value(self):
        if self.value.get():
            v = self.value.get()
            # Do some quality assurance here.
            # Is there a directory included?
            if not path.dirname(v):
                v = path.join(self.runner.working_dir, v)

            # What about a file extension?
            ext = os.path.splitext(v)[-1].lower().strip()
            if not ext:
                ext = ""
                if 'RasterAndVector' in self.file_type:
                    ext = '.tif'
                elif 'Raster' in self.file_type:
                    ext = '.tif'
                elif 'Lidar' in self.file_type:
                    ext = '.las'
                elif 'Vector' in self.file_type:
                    ext = '.shp'
                elif 'Text' in self.file_type:
                    ext = '.txt'
                elif 'Csv' in self.file_type:
                    ext = '.csv'
                elif 'Html' in self.file_type:
                    ext = '.html'

                v += ext

            v = path.normpath(v)

            # return "{}='{}'".format(self.flag, v)
            return self.flag, v
        else:
            t = "file"
            if self.parameter_type == "Directory":
                t = "directory"
            if not self.optional:
                messagebox.showinfo(
                    "Error", "Unspecified {} parameter {}.".format(t, self.flag))

        return None

    def select_all(self, event):
        self.entry.select_range(0, tk.END)
        return 'break'


class FileOrFloat(tk.Frame):
    def __init__(self, json_str, runner, master=None):
        # first make sure that the json data has the correct fields
        j = json.loads(json_str)
        self.name = j['name']
        self.description = j['description']
        self.flag = j['flags']
        self.parameter_type = j['parameter_type']
        self.file_type = j['parameter_type']['ExistingFileOrFloat']
        self.optional = j['optional']
        default_value = j['default_value']

        self.runner = runner

        ttk.Frame.__init__(self, master)
        self.grid()
        self['padding'] = '0.02i'

        self.label = ttk.Label(self, text=self.name, justify=tk.LEFT)
        self.label.grid(row=0, column=0, sticky=tk.W)
        self.label.columnconfigure(0, weight=1)

        if not self.optional:
            self.label['text'] = self.label['text'] + "*"

        fs_frame = ttk.Frame(self, padding='0.0i')
        self.value = tk.StringVar()
        self.entry = ttk.Entry(
            fs_frame, width=35, justify=tk.LEFT, textvariable=self.value)
        self.entry.grid(row=0, column=0, sticky=tk.NSEW)
        self.entry.columnconfigure(0, weight=1)
        if default_value:
            self.value.set(default_value)

        self.open_button = ttk.Button(
            fs_frame, width=4, text="...", command=self.select_file)
        self.open_button.grid(row=0, column=1, sticky=tk.E)

        self.label = ttk.Label(fs_frame, text='OR', justify=tk.LEFT)
        self.label.grid(row=0, column=2, sticky=tk.W)

        self.value2 = tk.StringVar()
        self.entry2 = ttk.Entry(
            fs_frame, width=10, justify=tk.LEFT, textvariable=self.value2)
        self.entry2.grid(row=0, column=3, sticky=tk.NSEW)
        self.entry2.columnconfigure(0, weight=1)
        self.entry2['justify'] = 'right'

        fs_frame.grid(row=1, column=0, sticky=tk.NSEW)
        fs_frame.columnconfigure(0, weight=10)
        fs_frame.columnconfigure(1, weight=1)
        # self.pack(fill=tk.BOTH, expand=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Add the bindings
        if _platform == "darwin":
            self.entry.bind("<Command-Key-a>", self.select_all)
        else:
            self.entry.bind("<Control-Key-a>", self.select_all)

    def select_file(self):
        try:
            result = self.value.get()
            file_types = [('All files', '*.*')]
            if 'RasterAndVector' in self.file_type:
                file_types = [("Shapefiles", "*.shp"), ('Raster files', ('*.dep', '*.tif', '*.tiff', '*.bil',
                                                                         '*.flt', '*.sdat', '*.rdc', '*.asc'))]
            elif 'Raster' in self.file_type:
                file_types = [('Raster files', ('*.dep', '*.tif', '*.tiff', '*.bil',
                                                '*.flt', '*.sdat', '*.rdc', '*.asc'))]
            elif 'Lidar' in self.file_type:
                file_types = [("LiDAR files", ('*.las', '*.zlidar', '*.laz', '*.zip'))]
            elif 'Vector' in self.file_type:
                file_types = [("Shapefiles", "*.shp")]
            elif 'Text' in self.file_type:
                file_types = [("Text files", "*.txt"), ("all files", "*.*")]
            elif 'Csv' in self.file_type:
                file_types = [("CSC files", "*.csv"), ("all files", "*.*")]
            elif 'Html' in self.file_type:
                file_types = [("HTML files", "*.html")]

            result = filedialog.askopenfilename(
                initialdir=self.runner.working_dir, title="Select file", filetypes=file_types)

            self.value.set(result)

            # update the working directory
            self.runner.working_dir = os.path.dirname(result)

        except:
            t = "file"
            if self.parameter_type == "Directory":
                t = "directory"
            messagebox.showinfo("Warning", "Could not find {}".format(t))

    def RepresentsFloat(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def get_value(self):
        if self.value.get():
            v = self.value.get()
            # Do some quality assurance here.
            # Is there a directory included?
            if not path.dirname(v):
                v = path.join(self.runner.working_dir, v)

            # What about a file extension?
            ext = os.path.splitext(v)[-1].lower()
            if not ext:
                ext = ""
                if 'RasterAndVector' in self.file_type:
                    ext = '.tif'
                elif 'Raster' in self.file_type:
                    ext = '.tif'
                elif 'Lidar' in self.file_type:
                    ext = '.las'
                elif 'Vector' in self.file_type:
                    ext = '.shp'
                elif 'Text' in self.file_type:
                    ext = '.txt'
                elif 'Csv' in self.file_type:
                    ext = '.csv'
                elif 'Html' in self.file_type:
                    ext = '.html'

                v = v + ext

            v = path.normpath(v)

            # return "{}='{}'".format(self.flag, v)
            return self.flag, v
        elif self.value2.get():
            v = self.value2.get()
            if self.RepresentsFloat(v):
                return "{}={}".format(self.flag, v)
            else:
                messagebox.showinfo(
                    "Error", "Error converting parameter {} to type Float.".format(self.flag))
        else:
            if not self.optional:
                messagebox.showinfo(
                    "Error", "Unspecified file/numeric parameter {}.".format(self.flag))

        return None

    def select_all(self, event):
        self.entry.select_range(0, tk.END)
        return 'break'


class MultifileSelector(tk.Frame):
    def __init__(self, json_str, runner, master=None):
        # first make sure that the json data has the correct fields
        j = json.loads(json_str)
        self.name = j['name']
        self.description = j['description']
        self.flag = j['flags']
        self.parameter_type = j['parameter_type']
        self.file_type = ""
        self.file_type = j['parameter_type']['FileList']
        self.optional = j['optional']
        default_value = j['default_value']

        self.runner = runner

        ttk.Frame.__init__(self, master)
        self.grid()
        self['padding'] = '0.05i'

        self.label = ttk.Label(self, text=self.name, justify=tk.LEFT)
        self.label.grid(row=0, column=0, sticky=tk.W)
        self.label.columnconfigure(0, weight=1)

        if not self.optional:
            self.label['text'] = self.label['text'] + "*"

        fs_frame = ttk.Frame(self, padding='0.0i')
        # , variable=self.value)
        self.opt = tk.Listbox(fs_frame, width=44, height=4)
        self.opt.grid(row=0, column=0, sticky=tk.NSEW)
        s = ttk.Scrollbar(fs_frame, orient=tk.VERTICAL, command=self.opt.yview)
        s.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.opt['yscrollcommand'] = s.set

        btn_frame = ttk.Frame(fs_frame, padding='0.0i')
        self.open_button = ttk.Button(
            btn_frame, width=4, text="...", command=self.select_file)
        self.open_button.grid(row=0, column=0, sticky=tk.NE)
        self.open_button.columnconfigure(0, weight=1)
        self.open_button.rowconfigure(0, weight=1)

        self.delete_button = ttk.Button(
            btn_frame, width=4, text="del", command=self.delete_entry)
        self.delete_button.grid(row=1, column=0, sticky=tk.NE)
        self.delete_button.columnconfigure(0, weight=1)
        self.delete_button.rowconfigure(1, weight=1)

        btn_frame.grid(row=0, column=2, sticky=tk.NE)

        fs_frame.grid(row=1, column=0, sticky=tk.NSEW)
        fs_frame.columnconfigure(0, weight=10)
        fs_frame.columnconfigure(1, weight=1)
        fs_frame.columnconfigure(2, weight=1)
        # self.pack(fill=tk.BOTH, expand=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

    def select_file(self):
        try:
            # result = self.value.get()
            init_dir = self.runner.working_dir
            file_types = [('All files', '*.*')]
            if 'RasterAndVector' in self.file_type:
                file_types = [("Shapefiles", "*.shp"), ('Raster files', ('*.dep', '*.tif', '*.tiff', '*.bil',
                                                                         '*.flt', '*.sdat', '*.rdc', '*.asc'))]
            elif 'Raster' in self.file_type:
                file_types = [('Raster files', ('*.dep', '*.tif', '*.tiff', '*.bil',
                                                '*.flt', '*.sdat', '*.rdc', '*.asc'))]
            elif 'Lidar' in self.file_type:
                file_types = [("LiDAR files", ('*.las', '*.zlidar', '*.laz', '*.zip'))]
            elif 'Vector' in self.file_type:
                file_types = [("Shapefiles", "*.shp")]
            elif 'Text' in self.file_type:
                file_types = [("Text files", "*.txt"), ("all files", "*.*")]
            elif 'Csv' in self.file_type:
                file_types = [("CSC files", "*.csv"), ("all files", "*.*")]
            elif 'Html' in self.file_type:
                file_types = [("HTML files", "*.html")]

            result = filedialog.askopenfilenames(
                initialdir=init_dir, title="Select files", filetypes=file_types)
            if result:
                for v in result:
                    self.opt.insert(tk.END, v)

                # update the working directory
                self.runner.working_dir = os.path.dirname(result[0])

        except:
            messagebox.showinfo("Warning", "Could not find file")

    def delete_entry(self):
        self.opt.delete(tk.ANCHOR)

    def get_value(self):
        try:
            l = self.opt.get(0, tk.END)
            if l:
                s = ""
                for i in range(0, len(l)):
                    v = l[i]
                    if not path.dirname(v):
                        v = path.join(self.runner.working_dir, v)
                    v = path.normpath(v)
                    if i < len(l) - 1:
                        s += "{};".format(v)
                    else:
                        s += "{}".format(v)

                # return "{}='{}'".format(self.flag, s)
                return self.flag, v
            else:
                if not self.optional:
                    messagebox.showinfo(
                        "Error", "Unspecified non-optional parameter {}.".format(self.flag))

        except:
            messagebox.showinfo(
                "Error", "Error formatting files for parameter {}".format(self.flag))

        return None


class BooleanInput(tk.Frame):
    def __init__(self, json_str, master=None):
        # first make sure that the json data has the correct fields
        j = json.loads(json_str)
        self.name = j['name']
        self.description = j['description']
        self.flag = j['flags']
        self.parameter_type = j['parameter_type']
        # just for quality control. BooleanInputs are always optional.
        self.optional = True
        default_value = j['default_value']

        ttk.Frame.__init__(self, master)
        self.grid()
        self['padding'] = '0.05i'

        frame = ttk.Frame(self, padding='0.0i')

        self.value = tk.IntVar()
        c = ttk.Checkbutton(frame, text=self.name,
                            width=55, variable=self.value)
        c.grid(row=0, column=0, sticky=tk.W)

        # set the default value
        if j['default_value'] is not None and j['default_value'] != 'false':
            self.value.set(1)
        else:
            self.value.set(0)

        frame.grid(row=1, column=0, sticky=tk.W)
        frame.columnconfigure(0, weight=1)

        # self.pack(fill=tk.BOTH, expand=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def get_value(self):
        if self.value.get() == 1:
            return self.flag  # FIXME: return tuple
        else:
            return None


class OptionsInput(tk.Frame):
    def __init__(self, json_str, master=None):

        # first make sure that the json data has the correct fields
        j = json.loads(json_str)
        self.name = j['name']
        self.description = j['description']
        self.flag = j['flags']
        self.parameter_type = j['parameter_type']
        self.optional = j['optional']
        default_value = j['default_value']

        ttk.Frame.__init__(self, master)
        self.grid()
        self['padding'] = '0.02i'

        frame = ttk.Frame(self, padding='0.0i')

        self.label = ttk.Label(self, text=self.name, justify=tk.LEFT)
        self.label.grid(row=0, column=0, sticky=tk.W)
        self.label.columnconfigure(0, weight=1)

        frame2 = ttk.Frame(frame, padding='0.0i')
        opt = ttk.Combobox(frame2, width=40)
        opt.grid(row=0, column=0, sticky=tk.NSEW)

        self.value = None  # initialize in event of no default and no selection
        i = 1
        default_index = -1
        option_list = j['parameter_type']['OptionList']
        values = ()
        for v in option_list:
            values += (v,)
            # opt.insert(tk.END, v)
            if v == default_value:
                default_index = i - 1
            i = i + 1

        opt['values'] = values

        opt.bind("<<ComboboxSelected>>", self.select)
        if default_index >= 0:
            opt.current(default_index)
            opt.event_generate("<<ComboboxSelected>>")

        frame2.grid(row=0, column=0, sticky=tk.W)
        frame.grid(row=1, column=0, sticky=tk.W)
        frame.columnconfigure(0, weight=1)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def get_value(self):
        if self.value:
            # return "{}='{}'".format(self.flag, self.value)
            return self.flag, self.value
        else:
            if not self.optional:
                messagebox.showinfo(
                    "Error", "Unspecified non-optional parameter {}.".format(self.flag))

        return None

    def select(self, event):
        widget = event.widget
        self.value = widget.get()  # selection[0])


class DataInput(tk.Frame):
    def __init__(self, json_str, master=None):

        # first make sure that the json data has the correct fields
        j = json.loads(json_str)
        self.name = j['name']
        self.description = j['description']
        self.flag = j['flags']
        self.parameter_type = j['parameter_type']
        self.optional = j['optional']
        default_value = j['default_value']

        ttk.Frame.__init__(self, master)
        self.grid()
        self['padding'] = '0.1i'

        self.label = ttk.Label(self, text=self.name, justify=tk.LEFT)
        self.label.grid(row=0, column=0, sticky=tk.W)
        self.label.columnconfigure(0, weight=1)

        self.value = tk.StringVar()
        if default_value:
            self.value.set(default_value)
        else:
            self.value.set("")

        self.entry = ttk.Entry(self, justify=tk.LEFT, textvariable=self.value)
        self.entry.grid(row=0, column=1, sticky=tk.NSEW)
        self.entry.columnconfigure(1, weight=10)

        if not self.optional:
            self.label['text'] = self.label['text'] + "*"

        if ("Integer" in self.parameter_type or
                "Float" in self.parameter_type or
                "Double" in self.parameter_type):
            self.entry['justify'] = 'right'

        # Add the bindings
        if _platform == "darwin":
            self.entry.bind("<Command-Key-a>", self.select_all)
        else:
            self.entry.bind("<Control-Key-a>", self.select_all)

        # self.pack(fill=tk.BOTH, expand=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=10)
        self.rowconfigure(0, weight=1)

    def RepresentsInt(self, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def RepresentsFloat(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def get_value(self):
        v = self.value.get()
        if v:
            if "Integer" in self.parameter_type:
                if self.RepresentsInt(self.value.get()):
                    # return "{}={}".format(self.flag, self.value.get())
                    return self.flag, self.value.get()
                else:
                    messagebox.showinfo("Error", "Error converting parameter {} to type Integer.".format(self.flag))
            elif "Float" in self.parameter_type:
                if self.RepresentsFloat(self.value.get()):
                    # return "{}={}".format(self.flag, self.value.get())
                    return self.flag, self.value.get()
                else:
                    messagebox.showinfo("Error", "Error converting parameter {} to type Float.".format(self.flag))
            elif "Double" in self.parameter_type:
                if self.RepresentsFloat(self.value.get()):
                    # return "{}={}".format(self.flag, self.value.get())
                    return self.flag, self.value.get()
                else:
                    messagebox.showinfo("Error", "Error converting parameter {} to type Double.".format(self.flag))
            else:  # String or StringOrNumber types
                # return "{}='{}'".format(self.flag, self.value.get())
                return self.flag, self.value.get()
        else:
            if not self.optional:
                messagebox.showinfo("Error", "Unspecified non-optional parameter {}.".format(self.flag))

        return None

    def select_all(self, event):
        self.entry.select_range(0, tk.END)
        return 'break'


class MainGui(tk.Frame):
    def __init__(self, tool_name=None, master=None):
        self.descriptionList = None
        self.search_string = None
        self.toolbox_open = None
        self.toolbox_name = None
        self.tools_list = None
        self.sorted_tools = None
        self.lower_toolboxes = None
        self.upper_toolboxes = None
        self.filemenu = None
        self.closed_toolbox_icon = None
        self.open_toolbox_icon = None
        self.tool_icon = None
        self.tool_tree = None
        self.progress = None
        self.progress_var = None
        self.progress_label = None
        self.help_button = None
        self.quit_button = None
        self.run_button = None
        # self.arg_scroll_frame = None
        self.view_code_button = None
        self.current_tool_lbl = None
        self.current_tool_frame = None
        self.search_scroll = None
        self.search_results_listbox = None
        self.search_bar = None
        self.search_label = None
        self.search_frame = None
        self.search_text = None
        self.search_list = None
        self.tools_frame = None
        self.toolbox_list = None
        self.tools_and_toolboxes = None
        self.out_text = None
        self.current_tool_api = None

        if platform.system() == 'Windows':
            self.ext = '.exe'
        else:
            self.ext = ''

        exe_name = "BERA_tools{}".format(self.ext)

        # Load BERA Tools from json file
        tools = open(r'beratools\tools\beratools.json')
        self.get_bera_tools = json.load(tools)

        self.exe_path = path.dirname(path.abspath(__file__))
        os.chdir(self.exe_path)
        for filename in glob.iglob('**/*', recursive=True):
            if filename.endswith(exe_name):
                self.exe_path = path.dirname(path.abspath(filename))
                break

        bt.set_bera_dir(self.exe_path)

        ttk.Frame.__init__(self, master)
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.grid()
        self.tool_name = tool_name
        self.master.title("BERA Tools")
        if _platform == "darwin":
            os.system('''/usr/bin/osascript -e 'tell app "Finder" to set front-most of process "Python" to true' ''')
        self.create_widgets()
        self.working_dir = bt.get_working_dir()  # str(Path.home())

    def create_widgets(self):
        #########################################################
        #              Overall/Top level Frame
        #
        # define left-side frame (toplevel_frame) and right-side frame (overall_frame)
        toplevel_frame = ttk.Frame(self, padding='0.1i')
        overall_frame = ttk.Frame(self, padding='0.1i')

        # set-up layout
        overall_frame.grid(row=0, column=1, sticky=tk.NSEW)
        toplevel_frame.grid(row=0, column=0, sticky=tk.NSEW)

        # BERA tool list
        self.get_bera_tool_list()
        self.toolbox_list = self.get_bera_toolboxes()
        self.sort_toolboxes()

        # Icons to be used in tool treeview
        self.tool_icon = tk.PhotoImage(file=self.script_dir + '//img//tool.gif')
        self.open_toolbox_icon = tk.PhotoImage(file=self.script_dir + '//img//open.gif')
        self.closed_toolbox_icon = tk.PhotoImage(file=self.script_dir + '//img//closed.gif')

        #########################################################
        #                  Toolboxes Frame
        # FIXME: change width or make horizontally scrollable
        #
        # define tools_frame and tool_tree
        self.tools_frame = ttk.LabelFrame(toplevel_frame, text="{} Available Tools".format(len(self.tools_list)),
                                          padding='0.1i')
        self.tool_tree = ttk.Treeview(self.tools_frame, height=21)

        # Set up layout
        self.tool_tree.grid(row=0, column=0, sticky=tk.NSEW)
        self.tool_tree.column("#0", width=280)  # Set width so all tools are readable within the frame
        self.tools_frame.grid(row=0, column=0, sticky=tk.NSEW)
        self.tools_frame.columnconfigure(0, weight=10)
        self.tools_frame.columnconfigure(1, weight=1)
        self.tools_frame.rowconfigure(0, weight=10)
        self.tools_frame.rowconfigure(1, weight=1)

        # Add toolboxes and tools to treeview
        self.add_tools_to_treeview()

        # bind tools in treeview to self.tree_update_tool_help function
        # and toolboxes to self.update_toolbox_icon function
        # TODO: BERA tool help
        self.tool_tree.tag_bind('tool', "<<TreeviewSelect>>", self.tree_update_tool_help)
        self.tool_tree.tag_bind('toolbox', "<<TreeviewSelect>>", self.update_toolbox_icon)

        # Add vertical scrollbar to treeview frame
        s = ttk.Scrollbar(self.tools_frame, orient=tk.VERTICAL, command=self.tool_tree.yview)
        s.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tool_tree['yscrollcommand'] = s.set

        #########################################################
        #                     Search Bar
        # create variables for search results and search input
        self.search_list = []
        self.search_text = tk.StringVar()

        # Create the elements of the search frame
        self.search_frame = ttk.LabelFrame(toplevel_frame, padding='0.1i',
                                           text="{} Tools Found".format(len(self.search_list)))
        self.search_label = ttk.Label(self.search_frame, text="Search: ")
        self.search_bar = ttk.Entry(self.search_frame, width=30, textvariable=self.search_text)
        self.search_results_listbox = tk.Listbox(self.search_frame, height=11)
        self.search_scroll = ttk.Scrollbar(self.search_frame, orient=tk.VERTICAL,
                                           command=self.search_results_listbox.yview)
        self.search_results_listbox['yscrollcommand'] = self.search_scroll.set

        # Add bindings
        self.search_results_listbox.bind("<<ListboxSelect>>", self.search_update_tool_help)
        self.search_bar.bind('<Return>', self.update_search)

        # Define layout of the frame
        self.search_frame.grid(row=1, column=0, sticky=tk.NSEW)
        self.search_label.grid(row=0, column=0, sticky=tk.NW)
        self.search_bar.grid(row=0, column=1, sticky=tk.NE)
        self.search_results_listbox.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW, pady=5)
        self.search_scroll.grid(row=1, column=2, sticky=(tk.N, tk.S))

        # Configure rows and columns of the frame
        self.search_frame.columnconfigure(0, weight=1)
        self.search_frame.columnconfigure(1, weight=10)
        self.search_frame.columnconfigure(1, weight=1)
        self.search_frame.rowconfigure(0, weight=1)
        self.search_frame.rowconfigure(1, weight=10)

        #########################################################
        #                 Current Tool Frame
        # Create the elements of the current tool frame
        self.current_tool_frame = ttk.Frame(overall_frame, padding='0.01i')
        self.current_tool_lbl = ttk.Label(self.current_tool_frame, text="Current Tool: {}".format(self.tool_name),
                                          justify=tk.LEFT)  # , font=("Helvetica", 12, "bold")
        self.view_code_button = ttk.Button(self.current_tool_frame, text="View Code", width=12, command=self.view_code)

        # Define layout of the frame
        self.view_code_button.grid(row=0, column=1, sticky=tk.E)
        self.current_tool_lbl.grid(row=0, column=0, sticky=tk.W)
        self.current_tool_frame.grid(row=0, column=0, columnspan=2, sticky=tk.NSEW)

        # Configure rows and columns of the frame
        self.current_tool_frame.columnconfigure(0, weight=1)
        self.current_tool_frame.columnconfigure(1, weight=1)

        self.arg_scroll_frame = ttk.Frame(overall_frame, padding='0.0i')
        self.arg_scroll_frame.grid(row=1, column=0, sticky=tk.NSEW)
        self.arg_scroll_frame.columnconfigure(0, weight=1)

        #########################################################
        #                   Buttons Frame
        #
        # Create the elements of the buttons frame
        buttons_frame = ttk.Frame(overall_frame, padding='0.1i')
        self.run_button = ttk.Button(buttons_frame, text="Run", width=8, command=self.run_tool)
        self.quit_button = ttk.Button(buttons_frame, text="Cancel", width=8, command=self.cancel_operation)
        self.help_button = ttk.Button(buttons_frame, text="Help", width=8, command=self.tool_help_button)

        # Define layout of the frame
        self.run_button.grid(row=0, column=0)
        self.quit_button.grid(row=0, column=1)
        self.help_button.grid(row=0, column=2)
        buttons_frame.grid(row=2, column=0, columnspan=2, sticky=tk.E)

        #########################################################
        #                  Output Frame
        # Create the elements of the output frame
        output_frame = ttk.Frame(overall_frame)
        out_label = ttk.Label(output_frame, text="Output:", justify=tk.LEFT)
        self.out_text = ScrolledText(output_frame, width=63, height=15, wrap=tk.NONE, padx=7, pady=7, exportselection=0)
        output_scrollbar = ttk.Scrollbar(output_frame, orient=tk.HORIZONTAL, command=self.out_text.xview)
        self.out_text['xscrollcommand'] = output_scrollbar.set
        # Retrieve and insert the text for the current tool

        # BERA Tools help text
        k = self.get_bera_tool_help()

        self.out_text.insert(tk.END, k)
        # Define layout of the frame
        out_label.grid(row=0, column=0, sticky=tk.NW)
        self.out_text.grid(row=1, column=0, sticky=tk.NSEW)
        output_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.NS, tk.E))
        output_scrollbar.grid(row=2, column=0, sticky=(tk.W, tk.E))

        # Configure rows and columns of the frame
        self.out_text.columnconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        # Add the binding
        if _platform == "darwin":
            self.out_text.bind("<Command-Key-a>", self.select_all)
        else:
            self.out_text.bind("<Control-Key-a>", self.select_all)

        #########################################################
        #                  Progress Frame
        #
        # Create the elements of the progress frame
        progress_frame = ttk.Frame(overall_frame, padding='0.1i')
        self.progress_label = ttk.Label(progress_frame, text="Progress:", justify=tk.LEFT)
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", variable=self.progress_var, length=200,
                                        maximum=100)

        # Define layout of the frame
        self.progress_label.grid(row=0, column=0, sticky=tk.E, padx=5)
        self.progress.grid(row=0, column=1, sticky=tk.E)
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=tk.SE)

        #########################################################
        #                  Tool Selection
        #
        # Select the appropriate tool, if specified, otherwise the first tool
        self.tool_tree.focus(self.tool_name)
        self.tool_tree.selection_set(self.tool_name)
        self.tool_tree.event_generate("<<TreeviewSelect>>")

        #########################################################
        #                       Menus
        menubar = tk.Menu(self)

        self.filemenu = tk.Menu(menubar, tearoff=0)
        self.filemenu.add_command(label="Set Working Directory", command=self.set_directory)
        self.filemenu.add_command(label="Refresh Tools", command=self.refresh_tools)

        if bt.get_verbose_mode():
            self.filemenu.add_command(label="Do Not Print Tool Output", command=self.update_verbose)
        else:
            self.filemenu.add_command(label="Print Tool Output", command=self.update_verbose)

        self.filemenu.add_command(label="Set Num. Processors", command=self.set_procs)

        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=self.filemenu)

        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Cut", command=lambda: self.focus_get().event_generate("<<Cut>>"))
        editmenu.add_command(label="Copy", command=lambda: self.focus_get().event_generate("<<Copy>>"))
        editmenu.add_command(label="Paste", command=lambda: self.focus_get().event_generate("<<Paste>>"))
        menubar.add_cascade(label="Edit ", menu=editmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.help)
        helpmenu.add_command(label="License", command=self.license)
        menubar.add_cascade(label="Help ", menu=helpmenu)

        self.master.config(menu=menubar)

    def update_verbose(self):
        if bt.get_verbose_mode():
            bt.set_verbose_mode(False)
            self.filemenu.entryconfig(2, label="Print Tool Output")
        else:
            bt.set_verbose_mode(True)
            self.filemenu.entryconfig(2, label="Do Not Print Tool Output")

    def update_compress(self):
        if bt.get_compress_rasters():
            bt.set_compress_rasters(False)
            self.filemenu.entryconfig(3, label="Compress Output TIFFs")
        else:
            bt.set_compress_rasters(True)
            self.filemenu.entryconfig(3, label="Do Not Compress Output TIFFs")

    def install_extension(self):

        self.refresh_tools()


    def sort_toolboxes(self):
        self.upper_toolboxes = []
        self.lower_toolboxes = []
        for toolbox in self.toolbox_list:
            if toolbox.find('/') == (-1):  # Does not contain a sub toolbox, i.e. does not contain '/'
                self.upper_toolboxes.append(toolbox)  # add to both upper toolbox list and lower toolbox list
                self.lower_toolboxes.append(toolbox)
            else:  # Contains a sub toolbox
                self.lower_toolboxes.append(toolbox)  # add to only the lower toolbox list


    def get_tools_list(self):
        self.tools_list = []
        selected_item = -1
        for item in bt.list_tools().keys():
            if item:
                value = to_camelcase(item).replace("TIN", "Tin").replace("KS", "Ks").replace("FD",
                                                                                             "Fd")  # format tool name
                self.tools_list.append(value)  # add tool to list
                if item == self.tool_name:  # update selected_item it tool found
                    selected_item = len(self.tools_list) - 1
        if selected_item == -1:  # set self.tool_name as default tool
            selected_item = 0
            self.tool_name = self.tools_list[0]

    def get_bera_toolboxes(self):
        toolboxes = list()
        for toolbox in self.get_bera_tools['toolbox']:
            tb = toolbox['category']
            toolboxes.append(tb)
        return toolboxes

    def get_bera_tool_list(self):
        self.tools_list = []
        self.sorted_tools = []
        selected_item = -1
        for toolbox in self.get_bera_tools['toolbox']:
            category = []
            for item in toolbox['tools']:
                if item['name']:
                    category.append(item['name'])
                    self.tools_list.append(item['name'])  # add tool to list
                    if item == self.tool_name:  # update selected_item it tool found
                        selected_item = len(self.tools_list) - 1

            self.sorted_tools.append(category)

        if selected_item == -1:  # set self.tool_name as default tool
            selected_item = 0
            self.tool_name = self.tools_list[0]

    def get_bera_tool_help(self):
        for toolbox in self.get_bera_tools['toolbox']:
            for tool in toolbox['tools']:
                if tool['name'] == self.tool_name:
                    return tool['info']

    def get_bera_tool_parameters(self, tool_name):
        new_params = {'parameters': []}

        for toolbox in self.get_bera_tools['toolbox']:
            for tool in toolbox['tools']:
                if tool_name == tool['name']:
                    self.current_tool_api = tool['scriptFile']
                    new_params['tech_link'] = tool['tech_link']

                    # convert json format for parameters
                    for param in tool['parameters']:
                        new_param = {'name': param['parameter']}
                        if 'variable' in param.keys():
                            new_param['flags'] = param['variable']
                        else:
                            new_param['flags'] = 'FIXME'

                        if not param['output']:
                            if param['typelab'] == 'text':
                                if param['type'] == 'string':
                                    new_param["parameter_type"] = 'StringOrNumber'
                                elif 'list' in param['type']:
                                    a = param['type'].lstrip('list:').split(',')
                                    new_param["parameter_type"] = {"OptionList": a}
                            elif param['typelab'] == 'number':
                                if param['type'] == 'number':
                                    new_param["parameter_type"] = 'Float'
                                elif 'list' in param['type']:
                                    a = param['type'].lstrip('list:').split(',')
                                    new_param["parameter_type"] = {"OptionList": a}
                            elif param['typelab'] == r'True/False':
                                new_param["parameter_type"] = {"OptionList": ['True', 'False']}
                            else:
                                new_param['parameter_type'] = {'ExistingFile': ''}
                        else:
                            new_param["parameter_type"] = {'NewFile': ''}
                        new_param['description'] = param['description']

                        if param['typelab'] == 'TIF':
                            for i in new_param["parameter_type"].keys():
                                new_param['parameter_type'][i] = 'Raster'
                        elif param['typelab'] == 'SHP':
                            for i in new_param["parameter_type"].keys():
                                new_param['parameter_type'][i] = 'Vector'

                        new_param['default_value'] = param['default']
                        new_param['optional'] = False
                        new_params['parameters'].append(new_param)

        return new_params

    def get_current_tool_parameters(self):
        return self.get_bera_tool_parameters(self.tool_name)

    # read selection when tool selected from treeview then call self.update_tool_help
    def tree_update_tool_help(self, event):
        curItem = self.tool_tree.focus()
        self.tool_name = self.tool_tree.item(curItem).get('text').replace("  ", "")
        self.update_tool_help()

    # read selection when tool selected from search results then call self.update_tool_help
    def search_update_tool_help(self, event):
        selection = self.search_results_listbox.curselection()
        self.tool_name = self.search_results_listbox.get(selection[0])
        self.update_tool_help()

    def update_tool_help(self):
        self.out_text.delete('1.0', tk.END)
        for widget in self.arg_scroll_frame.winfo_children():
            widget.destroy()

        k = self.get_bera_tool_help()
        self.print_to_output(k)

        j = self.get_current_tool_parameters()

        param_num = 0
        for p in j['parameters']:
            json_str = json.dumps(
                p, sort_keys=True, indent=2, separators=(',', ': '))
            pt = p['parameter_type']
            if 'ExistingFileOrFloat' in pt:
                ff = FileOrFloat(json_str, self, self.arg_scroll_frame)
                ff.grid(row=param_num, column=0, sticky=tk.NSEW)
                param_num = param_num + 1
            elif 'ExistingFile' in pt or 'NewFile' in pt or 'Directory' in pt:
                fs = FileSelector(json_str, self, self.arg_scroll_frame)
                fs.grid(row=param_num, column=0, sticky=tk.NSEW)
                param_num = param_num + 1
            elif 'FileList' in pt:
                b = MultifileSelector(json_str, self, self.arg_scroll_frame)
                b.grid(row=param_num, column=0, sticky=tk.W)
                param_num = param_num + 1
            elif 'Boolean' in pt:
                b = BooleanInput(json_str, self.arg_scroll_frame)
                b.grid(row=param_num, column=0, sticky=tk.W)
                param_num = param_num + 1
            elif 'OptionList' in pt:
                b = OptionsInput(json_str, self.arg_scroll_frame)
                b.grid(row=param_num, column=0, sticky=tk.W)
                param_num = param_num + 1
            elif ('Float' in pt or 'Integer' in pt or
                  'Text' in pt or 'String' in pt or 'StringOrNumber' in pt or
                  'StringList' in pt or 'VectorAttributeField' in pt):
                b = DataInput(json_str, self.arg_scroll_frame)
                b.grid(row=param_num, column=0, sticky=tk.NSEW)
                param_num = param_num + 1
            else:
                messagebox.showinfo(
                    "Error", "Unsupported parameter type: {}.".format(pt))
        self.update_args_box()
        self.out_text.see("%d.%d" % (1, 0))

    def update_toolbox_icon(self, event):
        curItem = self.tool_tree.focus()
        dictTool = self.tool_tree.item(curItem)  # retrieve the toolbox name
        self.toolbox_name = dictTool.get('text').replace("  ", "")  # delete the space between the icon and text
        self.toolbox_open = dictTool.get('open')  # retrieve whether the toolbox is open or not
        if self.toolbox_open:  # set image accordingly
            self.tool_tree.item(self.toolbox_name, image=self.open_toolbox_icon)
        else:
            self.tool_tree.item(self.toolbox_name, image=self.closed_toolbox_icon)

    def update_search(self, event):
        self.search_list = []
        self.search_string = self.search_text.get().lower()
        self.search_results_listbox.delete(0, 'end')  # empty the search results
        num_results = 0
        for tool in self.tools_list:  # search tool names
            toolLower = tool.lower()
            if toolLower.find(self.search_string) != (-1):  # search string found within tool name
                num_results = num_results + 1
                self.search_results_listbox.insert(num_results,
                                                   tool)  # tool added to listbox and to search results string
                self.search_list.append(tool)
        index = 0

    def get_descriptions(self):
        self.descriptionList = []
        tools = bt.list_tools()
        toolsItems = tools.items()
        for t in toolsItems:
            self.descriptionList.append(t[1])  # second entry in tool dictionary is the description

    def tool_help_button(self):
        # open the user manual section for the current tool
        webbrowser.open_new_tab(self.get_current_tool_parameters()['tech_link'])

    def camel_to_snake(self, s):  # taken from tools_info.py
        _underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
        _underscorer2 = re.compile('([a-z0-9])([A-Z])')
        subbed = _underscorer1.sub(r'\1_\2', s)
        return _underscorer2.sub(r'\1_\2', subbed).lower()

    def refresh_tools(self):
        # refresh lists
        self.tools_and_toolboxes = bt.toolbox('')
        self.get_tools_list()
        # clear self.tool_tree
        self.tool_tree.delete(*self.tool_tree.get_children())
        self.add_tools_to_treeview()

        # Update label
        self.tools_frame["text"] = "{} Available Tools".format(len(self.tools_list))

    def add_tools_to_treeview(self):
        # Add toolboxes and tools to treeview
        index = 0
        for toolbox in self.lower_toolboxes:
            if toolbox.find('/') != (-1):  # toolboxes
                self.tool_tree.insert(toolbox[:toolbox.find('/')], 0, text="  " + toolbox[toolbox.find('/') + 1:],
                                      iid=toolbox[toolbox.find('/') + 1:], tags='toolbox',
                                      image=self.closed_toolbox_icon)
                for tool in self.sorted_tools[index]:  # add tools within toolbox
                    self.tool_tree.insert(toolbox[toolbox.find('/') + 1:], 'end', text="  " + tool,
                                          tags='tool', iid=tool, image=self.tool_icon)
            else:  # sub toolboxes
                self.tool_tree.insert('', 'end', text="  " + toolbox, iid=toolbox, tags='toolbox',
                                      image=self.closed_toolbox_icon)
                for tool in self.sorted_tools[index]:  # add tools within sub toolbox
                    self.tool_tree.insert(toolbox, 'end', text="  " + tool, iid=tool, tags='tool', image=self.tool_icon)
            index = index + 1

    #########################################################
    #               Functions (original)
    def help(self):
        self.out_text.delete('1.0', tk.END)
        self.print_to_output(bt.help())

    def license(self):
        self.out_text.delete('1.0', tk.END)
        self.print_to_output(bt.license())

    def set_directory(self):
        try:
            self.working_dir = filedialog.askdirectory(initialdir=self.working_dir)
            bt.set_working_dir(self.working_dir)
        except:
            messagebox.showinfo(
                "Warning", "Could not set the working directory.")

    def set_procs(self):
        try:
            max_cpu_cores = multiprocessing.cpu_count()
            max_procs = askinteger(
                title="Max CPU cores used",
                prompt="Set the number of processors to be used (maximum: {}, -1: all):".format(max_cpu_cores),
                parent=self, initialvalue=bt.get_max_procs(), minvalue=-1, maxvalue=max_cpu_cores)
            if max_procs:
                self.__max_procs = max_procs
                bt.set_max_procs(self.__max_procs)
        except:
            messagebox.showinfo(
                "Warning", "Could not set the number of processors.")

    def select_exe(self):
        try:
            filename = filedialog.askopenfilename(initialdir=self.exe_path)
            self.exe_path = path.dirname(path.abspath(filename))
            bt.set_whitebox_dir(self.exe_path)
            self.refresh_tools()
        except:
            messagebox.showinfo(
                "Warning", "Could not find WhiteboxTools executable file.")

    def run_tool(self):
        bt.set_working_dir(self.working_dir)

        args = {}
        for widget in self.arg_scroll_frame.winfo_children():
            v = widget.get_value()
            if v and len(v) == 2:
                args[v[0]] = v[1]
            elif not widget.optional:
                messagebox.showinfo(
                    "Error", "Non-optional tool parameter not specified.")
                return

        self.print_line_to_output("")
        self.print_line_to_output("Tool arguments:{}".format(args))
        self.print_line_to_output("")

        # Run the tool and check the return value for an error
        if bt.run_tool(self.current_tool_api, args, self.custom_callback) == 1:
            print("Error running {}".format(self.tool_name))

        else:
            self.run_button["text"] = "Run"
            self.progress_var.set(0)
            self.progress_label['text'] = "Progress:"
            self.progress.update_idletasks()

    def print_to_output(self, value):
        self.out_text.insert(tk.END, value)
        self.out_text.see(tk.END)

    def print_line_to_output(self, value):
        self.out_text.insert(tk.END, value + "\n")
        self.out_text.see(tk.END)

    def cancel_operation(self):
        bt.cancel_op = True
        self.print_line_to_output("Tool operation cancelling...")
        self.progress.update_idletasks()

    def view_code(self):
        webbrowser.open_new_tab(self.get_current_tool_parameters()['tech_link'])

    def update_args_box(self):
        s = ""
        self.current_tool_lbl['text'] = "Current Tool: {}".format(
            self.tool_name)
        # self.spacer['width'] = width=(35-len(self.tool_name))
        # for item in bt.tool_help(self.tool_name).splitlines():
        for item in self.get_bera_tool_help().splitlines():
            if item.startswith("-"):
                k = item.split(" ")
                if "--" in k[1]:
                    value = k[1].replace(",", "")
                else:
                    value = k[0].replace(",", "")

                if "flag" in item.lower():
                    s = s + value + " "
                else:
                    if "file" in item.lower():
                        s = s + value + "='{}' "
                    else:
                        s = s + value + "={} "

        # self.args_value.set(s.strip())

    def custom_callback(self, value):
        """
        A custom callback for dealing with tool output.
        """
        value = str(value)
        if "%" in value:
            try:
                str_array = value.split(" ")
                label = value.replace(
                    str_array[len(str_array) - 1], "").strip()
                progress = float(
                    str_array[len(str_array) - 1].replace("%", "").strip())
                self.progress_var.set(int(progress))
                self.progress_label['text'] = label
            except ValueError as e:
                print("Problem converting parsed data into number: ", e)
            except Exception as e:
                print(e)
        else:
            self.print_line_to_output(value)

        self.update()  # this is needed for cancelling and updating the progress bar

    def select_all(self, event):
        self.out_text.tag_add(tk.SEL, "1.0", tk.END)
        self.out_text.mark_set(tk.INSERT, "1.0")
        self.out_text.see(tk.INSERT)
        return 'break'


class JsonPayload(object):
    def __init__(self, j):
        self.__dict__ = json.loads(j)


def Runner():
    tool_name = None
    if len(sys.argv) > 1:
        tool_name = str(sys.argv[1])
    btr = MainGui(tool_name)

    ico = Image.open(r'img\BERALogo.png')
    photo = ImageTk.PhotoImage(ico)
    btr._root().wm_iconphoto(False, photo)

    btr.mainloop()


if __name__ == '__main__':
    Runner()
