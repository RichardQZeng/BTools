import os
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QLineEdit,
    QFileDialog,
    QComboBox,
    QWidget,
    QPushButton,
    QLabel,
    QSlider,
    QMessageBox,
    QStyleOptionSlider,
    QStyle,
    QToolTip,
    QAbstractSlider,
    QHBoxLayout,
    QVBoxLayout,
    QSpinBox,
    QDoubleSpinBox
)

from PyQt5.QtCore import pyqtSignal, Qt, QPoint
from re import search

from common import *


class ToolWidgets(QWidget):
    signal_save_tool_params = pyqtSignal(object)

    def __init__(self, tool_name, tool_args, show_advanced, parent=None):
        super(ToolWidgets, self).__init__(parent)

        self.tool_name = tool_name
        self.show_advanced = show_advanced
        self.current_tool_api = ''
        self.widget_list = []
        self.setWindowTitle("Tool widgets")

        self.create_widgets(tool_args)
        layout = QVBoxLayout()

        for item in self.widget_list:
            layout.addWidget(item)

        self.save_button = QPushButton('Save Parameters')
        self.save_button.clicked.connect(self.save_tool_parameters)
        self.save_button.setFixedSize(200, 50)
        layout.addSpacing(20)
        self.setLayout(layout)

    def get_widgets_arguments(self):
        args = {}
        param_missing = False
        for widget in self.widget_list:
            v = widget.get_value()
            if v and len(v) == 2:
                args[v[0]] = v[1]
            else:
                print(f'[Missing argument]: {widget.name} parameter not specified.', 'missing')
                param_missing = True

        if param_missing:
            args = None

        return args

    def create_widgets(self, tool_args):
        param_num = 0
        for p in tool_args:
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
                widget = MultiFileSelector(json_str, None)
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
            if param_value is None:
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

                if not self.show_advanced and widget.optional:
                    widget.hide()

            self.widget_list.append(widget)

    def update_widgets(self, values_dict):
        for key, value in values_dict.items():
            for item in self.widget_list:
                if key == item.flag:
                    item.set_value(value)

    def save_tool_parameters(self):
        params = {}
        for item in self.widget_list:
            if item.flag:
                params[item.flag] = item.get_value()

        self.signal_save_tool_params.emit(params)

    def load_default_args(self):
        for item in self.widget_list:
            item.set_default_value()


class FileSelector(QWidget):
    def __init__(self, json_str, parent=None):
        super(FileSelector, self).__init__(parent)

        # first make sure that the json data has the correct fields
        params = json.loads(json_str)
        self.name = params['name']
        self.description = params['description']
        self.flag = params['flag']
        self.parameter_type = params['parameter_type']
        self.file_type = ""
        if "ExistingFile" in self.parameter_type:
            self.file_type = params['parameter_type']['ExistingFile']
        elif "NewFile" in self.parameter_type:
            self.file_type = params['parameter_type']['NewFile']
        self.optional = params['optional']

        self.default_value = params['default_value']
        self.value = self.default_value
        if 'saved_value' in params.keys():
            self.value = params['saved_value']

        self.layout = QHBoxLayout()
        self.label = QLabel(self.name)
        self.label.setMinimumWidth(BT_LABEL_MIN_WIDTH)
        self.in_file = QLineEdit(self.value)
        self.btn_select = QPushButton("...")
        self.btn_select.clicked.connect(self.select_file)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.in_file)
        self.layout.addWidget(self.btn_select)

        self.setLayout(self.layout)

        # text changed
        self.in_file.textChanged.connect(self.set_value)

    def select_file(self):
        try:
            dialog = QFileDialog(self)
            dialog.setViewMode(QFileDialog.Detail)
            dialog.setDirectory(str(Path(self.value).parent))
            dialog.selectFile(Path(self.value).name)
            result = None
            file_names = None

            if "Directory" in self.parameter_type:
                dialog.setFileMode(QFileDialog.Directory)

                if dialog.exec_():
                    folder_name = dialog.selectedFiles()
                    print(folder_name)
                    self.set_value(folder_name)
            elif "ExistingFile" in self.parameter_type or "NewFile" in self.parameter_type:
                file_types = "All files '*.*')"
                if 'RasterAndVector' in self.file_type:
                    file_types = """Shapefiles (*.shp);; 
                                    Raster files (*.dep *.tif *.tiff *.bil *.flt *.sdat *.rdc *.asc *grd)"""
                elif 'Raster' in self.file_type:
                    file_types = """Tiff raster files (*.tif *.tiff);; 
                                    Other raster files (*.dep *.bil *.flt *.sdat *.rdc *.asc *.grd)"""
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

                result = file_path.as_posix()
                self.set_value(result)
        except Exception as e:
            print(e)

            t = "file"
            if self.parameter_type == "Directory":
                t = "directory"

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("Could not find {}".format(t))
            msg_box.exec()

    @staticmethod
    def get_file_filter_list(filter_str):
        """
        Extract filters out of full filter string, split int list and replace first '*'
        Result: ['.shp', '.*']
        """
        filter_list = search('\((.+?)\)', filter_str).group(1).split(' ')
        filter_list = [item.replace('*', '', 1) for item in filter_list if item != '']
        return filter_list

    def get_value(self):
        return self.flag, self.value

    def set_value(self, value):
        if type(value) is list:
            if len(value) > 0:
                value = value[0]

        self.value = value
        self.in_file.setText(self.value)
        self.in_file.setToolTip(self.value)

    def set_default_value(self):
        self.value = self.default_value
        self.in_file.setText(self.value)


class FileOrFloat(QWidget):
    def __init__(self, json_str, parent=None):
        super(FileOrFloat, self).__init__(parent)
        pass


class MultiFileSelector(QWidget):
    def __init__(self, json_str, parent=None):
        super(MultiFileSelector, self).__init__(parent)
        pass


class BooleanInput(QWidget):
    def __init__(self, json_str, parent=None):
        super(BooleanInput, self).__init__(parent)
        pass


class OptionsInput(QWidget):
    def __init__(self, json_str, parent=None):
        super(OptionsInput, self).__init__(parent)

        # first make sure that the json data has the correct fields
        params = json.loads(json_str)
        self.name = params['name']
        self.description = params['description']
        self.flag = params['flag']
        self.parameter_type = params['parameter_type']
        self.optional = params['optional']
        self.data_type = params['data_type']

        self.default_value = str(params['default_value'])
        self.value = self.default_value
        if 'saved_value' in params.keys():
            self.value = params['saved_value']

        self.label = QLabel(self.name)
        self.label.setMinimumWidth(BT_LABEL_MIN_WIDTH)
        self.combobox = QComboBox()
        self.combobox.currentIndexChanged.connect(self.selection_change)

        i = 1
        default_index = -1
        self.option_list = params['parameter_type']['OptionList']
        if self.option_list:
            self.option_list = [str(item) for item in self.option_list]  # convert to strings
        values = ()
        for v in self.option_list:
            values += (v,)
            if v == str(self.value):
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

    def set_value(self, value):
        self.value = value
        for v in self.option_list:
            if value == v:
                self.combobox.setCurrentIndex(self.option_list.index(v))

    def set_default_value(self):
        self.value = self.default_value
        for v in self.option_list:
            if self.value == v:
                self.combobox.setCurrentIndex(self.option_list.index(v))

    def get_value(self):
        return self.flag, self.value


class DataInput(QWidget):
    def __init__(self, json_str, parent=None):
        super(DataInput, self).__init__(parent)

        # first make sure that the json data has the correct fields
        params = json.loads(json_str)
        self.name = params['name']
        self.description = params['description']
        self.flag = params['flag']
        self.parameter_type = params['parameter_type']
        self.optional = params['optional']

        self.default_value = params['default_value']
        self.value = self.default_value
        if 'saved_value' in params.keys():
            self.value = params['saved_value']

        self.label = QLabel(self.name)
        self.label.setMinimumWidth(BT_LABEL_MIN_WIDTH)
        self.data_input = None

        if "Integer" in self.parameter_type:
            self.data_input = QSpinBox()
            # TODO: add number range to json file
            self.data_input.setRange(0, 1024)
        elif "Float" in self.parameter_type or "Double" in self.parameter_type:
            self.data_input = QDoubleSpinBox()

        if self.data_input:
            self.data_input.setValue(self.value)

        self.data_input.valueChanged.connect(self.update_value)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.data_input)
        self.setLayout(self.layout)

    def update_value(self):
        self.value = self.data_input.value()

    def get_value(self):
        v = self.value
        if v is not None:
            if "Integer" in self.parameter_type:
                return self.flag, int(self.value)
            elif "Float" in self.parameter_type:
                return self.flag, float(self.value)
            elif "Double" in self.parameter_type:
                return self.flag, float(self.value)
            else:  # String or StringOrNumber types
                return self.flag, self.value
        else:
            if not self.optional:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setText("Unspecified non-optional parameter {}.".format(self.flag))
                msg_box.exec()

        return None

    def set_value(self, value):
        if self.data_input:
            self.data_input.setValue(value)
            self.update_value()

    def set_default_value(self):
        if self.data_input:
            self.data_input.setValue(self.default_value)
            self.update_value()


class DoubleSlider(QSlider):
    # create our signal that we can connect to if necessary
    doubleValueChanged = pyqtSignal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(DoubleSlider, self).__init__(Qt.Horizontal)
        self._multi = 10 ** decimals

        self.opt = QStyleOptionSlider()
        self.initStyleOption(self.opt)

        self.valueChanged.connect(self.emit_double_value_changed)

    def emit_double_value_changed(self):
        value = float(super(DoubleSlider, self).value()) / self._multi
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
            bottom_right_corner = sr.bottomLeft()
            QToolTip.showText(self.mapToGlobal(QPoint(bottom_right_corner.x(), bottom_right_corner.y())),
                              str(self.value()), self)


if __name__ == '__main__':
    from bt_data import BTData

    bt = BTData()

    app = QApplication(sys.argv)
    dlg = ToolWidgets('Raster Line Attributes',
                      bt.get_bera_tool_args('Raster Line Attributes'),
                      bt.show_advanced)
    dlg.show()
    sys.exit(app.exec_())
