import os
import sys
from PyQt5.QtWidgets import (QApplication, QLineEdit, QFileDialog, QComboBox, QWidget,
                             QPushButton, QLabel, QSlider, QMessageBox,
                             QStyleOptionSlider, QStyle, QToolTip, QAbstractSlider,
                             QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox)

from PyQt5.QtCore import pyqtSignal, Qt, QRect, QPoint
from pathlib import Path
import json
import re

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tools.beratools_main import BeraTools
from tools.common import *

bt = BeraTools()


class ToolWin(QWidget):
    def __init__(self, tool_name, parent=None):
        super(ToolWin, self).__init__(parent)

        self.tool_name = tool_name
        self.current_tool_api = ''
        self.widget_list = []

        self.create_widgets()
        layout = QVBoxLayout()

        for item in self.widget_list:
            layout.addWidget(item)

        layout.addStretch()
        self.setLayout(layout)
        self.setWindowTitle("Tool widgets")

    def get_current_tool_parameters(self):
        tool_params = bt.get_bera_tool_parameters(self.tool_name)
        self.current_tool_api = tool_params['tool_api']
        return tool_params

    def create_widgets(self):
        k = bt.get_bera_tool_info(self.tool_name)
        print(k)
        print('\n')

        j = self.get_current_tool_parameters()

        param_num = 0
        for p in j['parameters']:
            json_str = json.dumps(p, sort_keys=True, indent=2, separators=(',', ': '))
            pt = p['parameter_type']
            widget = None

            if 'ExistingFileOrFloat' in pt:
                widget = FileOrFloat(json_str, None)
                param_num = param_num + 1
            elif 'ExistingFile' in pt or 'NewFile' in pt or 'Directory' in pt:
                widget = FileSelector(json_str, None)
                param_num = param_num + 1
            elif 'FileList' in pt:
                widget = MultifileSelector(json_str, None)
                param_num = param_num + 1
            elif 'Boolean' in pt:
                widget = BooleanInput(json_str)
                param_num = param_num + 1
            elif 'OptionList' in pt:
                widget = OptionsInput(json_str)
                param_num = param_num + 1
            elif ('Float' in pt or 'Integer' in pt or
                  'Text' in pt or 'String' in pt or 'StringOrNumber' in pt or
                  'StringList' in pt or 'VectorAttributeField' in pt):
                widget = DataInput(json_str)
                param_num = param_num + 1
            else:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setText("Unsupported parameter type: {}.".format(pt))
                msg_box.exec()

            param_value = None
            if 'saved_value' in p.keys():
                param_value = p['saved_value']
            if not param_value:
                param_value = p['default_value']
            if param_value is not None:
                if type(widget) is OptionsInput:
                    widget.value = param_value
                elif widget:
                    widget.value = param_value
            else:
                print('No default value found: {}'.format(p['name']))

            # hide optional widgets
            if widget:
                if widget.optional and widget.label:
                    widget.label.setStyleSheet("QLabel { background-color : transparent; color : blue; }")

                if widget.optional and not bt.show_advanced:
                    widget.hide()

            self.widget_list.append(widget)

    def update_widgets(self, values_dict):
        for key, value in values_dict.items():
            for item in self.widget_list:
                if key == item.flag:
                    item.set_value(value)


class FileSelector(QWidget):
    def __init__(self, json_str, runner, master=None, parent=None):
        super(FileSelector, self).__init__(parent)

        # first make sure that the json data has the correct fields
        j = json.loads(json_str)
        self.name = j['name']
        self.description = j['description']
        self.flag = j['flag']
        self.parameter_type = j['parameter_type']
        self.file_type = ""
        if "ExistingFile" in self.parameter_type:
            self.file_type = j['parameter_type']['ExistingFile']
        elif "NewFile" in self.parameter_type:
            self.file_type = j['parameter_type']['NewFile']
        self.optional = j['optional']
        self.value = j['default_value']

        self.runner = runner

        self.layout = QHBoxLayout()
        self.label = QLabel(self.name)
        self.label.setMinimumWidth(BT_LABEL_MIN_WIDTH)
        self.in_file = QLineEdit()
        self.btn_select = QPushButton("...")
        self.btn_select.clicked.connect(self.select_file)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.in_file)
        self.layout.addWidget(self.btn_select)

        self.setLayout(self.layout)

    def select_file(self):
        try:
            dialog = QFileDialog(self)
            dialog.setViewMode(QFileDialog.Detail)
            dialog.setDirectory(str(Path(self.value).parent))
            dialog.selectFile(Path(self.value).name)
            result = None
            file_names = None

            if self.parameter_type == "Directory":
                dialog.setFileMode(QFileDialog.FileMode.Directory)
            elif "ExistingFile" in self.parameter_type or "NewFile" in self.parameter_type:
                file_types = "All files '*.*')"
                if 'RasterAndVector' in self.file_type:
                    file_types = "Shapefiles (*.shp);; Raster files (*.dep *.tif *.tiff *.bil *.flt *.sdat *.rdc *.asc *grd)"
                elif 'Raster' in self.file_type:
                    file_types = "Tiff raster files (*.tif *.tiff);; Other raster files (*.dep *.bil *.flt *.sdat *.rdc *.asc *.grd)"
                elif 'Lidar' in self.file_type:
                    file_types = "LiDAR files (*.las *.zlidar *.laz *.zip)"
                elif 'Vector' in self.file_type:
                    file_types = "Shapefiles (*.shp)"
                elif 'Text' in self.file_type:
                    file_types = "Text files (*.txt);; all files (*.*)"
                elif 'Csv' in self.file_type:
                    file_types = "CSC files (*.csv);; all files (*.*)"
                elif 'Dat' in self.file_type:
                    file_types = "Binary data files (*.dat);; all files (*.*)"
                elif 'Html' in self.file_type:
                    file_types = "HTML files (*.html)"
                elif 'json' in self.file_type or 'JSON' in self.file_type:
                    file_types = "JSON files (*.json)"

                dialog.setNameFilter(file_types)

                if "ExistingFile" in self.parameter_type:
                    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
                else:
                    dialog.setFileMode(QFileDialog.FileMode.AnyFile)

                if dialog.exec_():
                    file_names = dialog.selectedFiles()

                if not file_names:
                    return

                if len(file_names) == 0:
                    print('No file(s) selected.')

                    if file_names[0] == '':
                        print('File name not valid.')
                        return

                # append suffix when not
                # TODO: more consideration for multiple formats
                result = file_names[0]
                file_path = Path(result)
                if result != '':
                    break_loop = False
                    selected_filters = self.get_file_filter_list(dialog.selectedNameFilter())

                    if file_path.suffix not in selected_filters:
                        if selected_filters[0] != '.*':
                            file_path = file_path.with_suffix(selected_filters[0])

                result = str(file_path)
                self.set_value(result)

            # update the working
            # if not self.runner and str(result) != '':
            #     self.runner.working_dir = os.path.dirname(result)
        except:
            t = "file"
            if self.parameter_type == "Directory":
                t = "directory"

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("Could not find {}".format(t))
            msg_box.exec()

    def get_file_filter_list(self, filter_str):
        """
        Extract filters out of full filter string, split int list and replace first '*'
        Result: ['.shp', '.*']
        """
        filter_list = re.search('\((.+?)\)', filter_str).group(1).split(' ')
        filter_list = [item.replace('*', '', 1) for item in filter_list if item != '']
        return filter_list

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value
        self.in_file.setText(Path(self.value).name)
        self.in_file.setToolTip(value)


class FileOrFloat(QWidget):
    def __init__(self, json_str, runner, master=None, parent=None):
        super(FileOrFloat, self).__init__(parent)
        pass


class MultifileSelector(QWidget):
    def __init__(self, json_str, runner, master=None, parent=None):
        super(MultifileSelector, self).__init__(parent)
        pass


class BooleanInput(QWidget):
    def __init__(self, json_str, master=None, parent=None):
        super(BooleanInput, self).__init__(parent)
        pass


class OptionsInput(QWidget):
    def __init__(self, json_str, master=None, parent=None):
        super(OptionsInput, self).__init__(parent)

        # first make sure that the json data has the correct fields
        j = json.loads(json_str)
        self.name = j['name']
        self.description = j['description']
        self.flag = j['flag']
        self.parameter_type = j['parameter_type']
        self.optional = j['optional']
        self.data_type = j['data_type']
        self.default_value = str(j['default_value'])
        self.value = self.default_value  # initialize in event of no default and no selection

        self.label = QLabel(self.name)
        self.label.setMinimumWidth(BT_LABEL_MIN_WIDTH)
        self.combobox = QComboBox()
        self.combobox.currentIndexChanged.connect(self.selection_change)

        i = 1
        default_index = -1
        self.option_list = j['parameter_type']['OptionList']
        if self.option_list:
            self.option_list = [str(item) for item in self.option_list]  # convert to strings
        values = ()
        for v in self.option_list:
            values += (v,)
            if v == str(self.default_value):
                default_index = i - 1
            i = i + 1

        self.combobox.addItems(self.option_list)
        self.combobox.setCurrentIndex(default_index)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.combobox)
        self.setLayout(self.layout)

    def selection_change(self, i):
        self.value = self.option_list[i]


class DataInput(QWidget):
    def __init__(self, json_str, master=None, parent=None):
        super(DataInput, self).__init__(parent)

        # first make sure that the json data has the correct fields
        j = json.loads(json_str)
        self.name = j['name']
        self.description = j['description']
        self.flag = j['flag']
        self.parameter_type = j['parameter_type']
        self.optional = j['optional']
        self.default_value = j['default_value']
        self.data = j['default_value']
        self.label = QLabel(self.name)
        self.label.setMinimumWidth(BT_LABEL_MIN_WIDTH)
        self.data_input = None

        if "Integer" in self.parameter_type:
            self.data_input = QSpinBox()
        elif "Float" in self.parameter_type or "Double" in self.parameter_type:
            self.data_input = QDoubleSpinBox()

        if self.data_input:
            self.data_input.setValue(self.data)

        self.data_input.valueChanged.connect(self.update_value)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.data_input)
        self.setLayout(self.layout)

    def update_value(self):
        self.data = self.data_input.value

    def get_value(self):
        v = self.data
        if v:
            if "Integer" in self.parameter_type:
                return self.flag, int(self.data)
            elif "Float" in self.parameter_type:
                return self.flag, float(self.data)
            elif "Double" in self.parameter_type:
                return self.flag, float(self.data)
            else:  # String or StringOrNumber types
                return self.flag, self.value.get()
        else:
            if not self.optional:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setText("Unspecified non-optional parameter {}.".format(self.flag))
                msg_box.exec()

        return None

class DoubleSlider(QSlider):

    # create our our signal that we can connect to if necessary
    doubleValueChanged = pyqtSignal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(DoubleSlider, self).__init__(Qt.Horizontal)
        self._multi = 10 ** decimals

        self.opt = QStyleOptionSlider()
        self.initStyleOption(self.opt)

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))

    def sliderChange(self, change):
        if change == QAbstractSlider.SliderValueChange:
            sr = self.style().subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
            bottomRightCorner = sr.bottomLeft()
            QToolTip.showText(self.mapToGlobal(QPoint(bottomRightCorner.x(), bottomRightCorner.y())),
                              str(self.value()), self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlg = ToolWin('Raster Line Attributes')
    dlg.show()
    sys.exit(app.exec_())
