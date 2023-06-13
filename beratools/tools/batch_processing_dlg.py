#!/usr/bin/python3
# -*- coding: utf-8 -*-

# QIcon fromtheme
# https://gist.github.com/RichardQZeng/2cf5b6d3d383df2242fda75ddb533baf

import pandas as pd
from PyQt5.QtCore import (Qt, QDir, QItemSelectionModel, QAbstractTableModel, QModelIndex, 
                          QVariant, QSize, QSettings, pyqtSignal)
from PyQt5.QtWidgets import (QMainWindow, QTableView, QApplication, QToolBar, QLineEdit, QComboBox, QAction,
                             QFileDialog, QAbstractItemView, QMessageBox, QWidget, QDockWidget, QFormLayout,
                             QSpinBox, QPushButton, QShortcut, QDialog, QMenuBar, QWidgetAction, QDialogButtonBox)
from PyQt5.QtGui import QIcon, QKeySequence, QTextDocument, QTextCursor, QTextTableFormat
from PyQt5 import QtPrintSupport

from bt_widgets import *

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent=None)
        self._df = df
        self.setChanged = False
        self.dataChanged.connect(self.setModified)

    def setModified(self):
        self.setChanged = True
        print(self.setChanged)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QVariant()
        elif orientation == Qt.Vertical:
            try:
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QVariant()

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.EditRole:
                return self._df.values[index.row()][index.column()]
            elif role == Qt.DisplayRole:
                return self._df.values[index.row()][index.column()]
        return None

    def data_row_dict(self, row):
        return self._df.iloc[row].to_dict()

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        self._df.values[row][col] = value
        self.dataChanged.emit(index, index)
        return True

    def rowCount(self, parent=QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending=order == Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

    def insertRows(self, position, rows=1, index=QModelIndex()):
        print("\n\t\t ...insertRows() Starting position: '%s'" % position, 'with the total rows to be inserted: ', rows)
        self.beginInsertRows(QModelIndex(), position, position + rows - 1)
        # del self._data[position]
        default_row = []
        for i in range(rows):
            self._df.loc[len(self._df)] = self.default_record

        # self.items = self.items[:position] + self.items[position + rows:]
        self.endInsertRows()
        return True

    def removeRows(self, position, rows=1, index=QModelIndex()):
        print("\n\t\t ...removeRows() Starting position: '%s'" % position, 'with the total rows to be removed: ', rows)
        self.beginRemoveRows(QModelIndex(), position, position + rows - 1)
        for i in range(rows):
            self._df.drop(self._df.index[position+i], inplace=True)
            print('removed: {}'.format(position+i))

        self.endRemoveRows()
        return True

    def updateRow(self, row, row_data):
        self._df.loc[row] = row_data

class BPDialog(QDialog):
    # signals
    signal_update_tool_widgets = pyqtSignal(int)

    def __init__(self, tool_name, parent=None):
        super(BPDialog, self).__init__(parent)
        self.setWindowTitle('Batch Processing')
        self.MaxRecentFiles = 5
        self.window_list = []
        self.recent_files = []
        self.settings = QSettings('Richard Zeng', 'Batch Processing')
        self.filename = ""
        self.setGeometry(0, 0, 800, 600)

        # tableview
        self.table_view = QTableView()
        self.table_view.verticalHeader().setVisible(True)
        self.model = PandasModel()
        self.table_view.setModel(self.model)
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_view.setSelectionBehavior(self.table_view.SelectRows)
        self.table_view.setSelectionMode(self.table_view.ExtendedSelection)

        self.table_view.clicked.connect(self.table_view_clicked)
        self.table_view.verticalHeader().sectionClicked.connect(self.table_view_vertical_header_clicked)
        QShortcut(Qt.Key_Up, self.table_view, activated=self.table_view_key_up)
        QShortcut(Qt.Key_Down, self.table_view, activated=self.table_view_key_down)

        # create form
        self.tool_name = tool_name
        self.tool_widgets = ToolWidgets(tool_name)

        # self.createToolBar()
        hbox_widgets = QHBoxLayout()
        hbox_widgets.addWidget(self.table_view, 2)
        hbox_widgets.addWidget(self.tool_widgets, 1)

        # Add project, record related button box
        self.project_btns = QHBoxLayout()
        open_button = QPushButton('Open')
        save_button = QPushButton('Save')
        save_as_button = QPushButton('Save as')
        delete_button = QPushButton('Delete records')
        add_button = QPushButton('Add record')
        print_button = QPushButton('Print')

        open_button.clicked.connect(self.load_csv)
        save_button.setShortcut(QKeySequence.Open)
        save_button.clicked.connect(self.write_csv_update)
        save_button.setShortcut(QKeySequence.Save)
        save_as_button.clicked.connect(self.write_csv)
        save_as_button.setShortcut(QKeySequence.SaveAs)
        delete_button.clicked.connect(self.table_view_delete_records)
        delete_button.setShortcut(QKeySequence.Delete)

        add_button.clicked.connect(self.table_view_add_records)

        self.last_files = QComboBox()
        self.last_files.setFixedWidth(300)
        self.last_files.currentIndexChanged.connect(self.load_recent)

        self.line_find = QLineEdit()
        self.line_find.setPlaceholderText("find")
        self.line_find.setClearButtonEnabled(True)
        self.line_find.setFixedWidth(250)
        self.line_find.returnPressed.connect(self.find_in_table)

        print_button.clicked.connect(self.handle_preview)

        self.project_btns.addWidget(open_button, QDialogButtonBox.ActionRole)
        self.project_btns.addWidget(save_button, QDialogButtonBox.ActionRole)
        self.project_btns.addWidget(save_as_button, QDialogButtonBox.ActionRole)
        self.project_btns.addWidget(delete_button, QDialogButtonBox.ActionRole)
        self.project_btns.addWidget(add_button, QDialogButtonBox.ActionRole)
        self.project_btns.addWidget(self.line_find, QDialogButtonBox.ActionRole)
        self.project_btns.addWidget(print_button, QDialogButtonBox.ActionRole)

        # Add OK/cancel buttons
        self.ok_btn_box = QDialogButtonBox()
        self.ok_btn_box.addButton("Run", QDialogButtonBox.AcceptRole)
        self.ok_btn_box.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.ok_btn_box.addButton("Help", QDialogButtonBox.HelpRole)

        self.ok_btn_box.accepted.connect(self.run)
        self.ok_btn_box.rejected.connect(self.reject)
        self.ok_btn_box.helpRequested.connect(self.help)

        hbox_btns = QHBoxLayout()
        hbox_btns.addLayout(self.project_btns)
        hbox_btns.addWidget(self.ok_btn_box)

        vbox_main = QVBoxLayout()
        vbox_main.addLayout(hbox_widgets)
        vbox_main.addLayout(hbox_btns)
        self.setLayout(vbox_main)

        # delete dialog when close
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.setContentsMargins(10, 10, 10, 10)
        self.read_settings()
        self.table_view.setFocus()
        # self.statusBar().showMessage("Ready", 0)

        # signals
        self.tool_widgets.signal_save_tool_params.connect(self.table_view_update_record)
        self.signal_update_tool_widgets.connect(self.update_tool_widgets)

    def accept(self):
        if self.line_find.hasFocus():
            return

        print("Run the batch processing.")
        QDialog.accept(self)

    def run(self):
        self.accept()

    def reject(self):
        print("Batch processing canceled.")
        self.close()

    def help(self):
        print("Help requested.")

    def table_view_clicked(self, item):
        print('Row, column:{}, {}'.format(item.row(), item.column()))
        self.signal_update_tool_widgets.emit(item.row())

    def table_view_vertical_header_clicked(self, item):
        print('Horizontal header clicked: {}'.format(item))
        self.signal_update_tool_widgets.emit(item)

    def table_view_key_up(self):
        current_row = self.table_view.selectionModel().selectedRows()[-1].row()
        if current_row >= 1:
            self.table_view.selectRow(current_row-1)
            self.signal_update_tool_widgets.emit(current_row - 1)

    def table_view_delete_records(self):
        selected_index = self.table_view.selectionModel().selectedRows()
        rows = [item.row() for item in selected_index]
        rows.sort(reverse=True)

        for i in rows:
            self.model.removeRows(i)

            current_row = i
            if self.model.rowCount() > 0:
                if current_row > self.model.rowCount() - 1:
                    current_row = self.model.rowCount() - 1

                self.table_view.selectRow(current_row)
                self.signal_update_tool_widgets.emit(current_row)

            print('remove row {}'.format(i))

        self.model.submit()

    def table_view_add_records(self):
        self.model.default_record = bt.get_bera_tool_parameters_list(self.tool_name)
        ret = self.model.insertRow(self.model.rowCount())
        if ret:
            count = self.model.rowCount() - 1
            self.table_view.selectRow(count)
            self.signal_update_tool_widgets.emit(count)

            print('Insert row in position {}'.format(count))
            self.model.submit()

    def table_view_update_record(self, row_data):
        current_row = self.table_view.selectionModel().selectedRows()[-1].row()
        self.model.updateRow(current_row, row_data)

    def update_tool_widgets(self, row):
        tool_paramas = self.model.data_row_dict(row)
        self.tool_widgets.update_widgets(tool_paramas)
        print('Update tool parameters for record {}'.format(tool_paramas))

    def table_view_key_down(self):
        current_row = self.table_view.selectionModel().selectedRows()[-1].row()
        if current_row < self.model.rowCount()-1:
            self.table_view.selectRow(current_row+1)
            self.signal_update_tool_widgets.emit(current_row + 1)

    def read_settings(self):
        print("reading settings")
        if self.settings.contains("geometry"):
            self.setGeometry(self.settings.value('geometry'))
        if self.settings.contains("recentFiles"):
            self.recent_files = self.settings.value('recentFiles')
            self.last_files.addItem("last Files")
            self.last_files.addItems(self.recent_files[:15])

    def save_settings(self):
        print("saving settings")
        self.settings.setValue('geometry', self.geometry())
        self.settings.setValue('recentFiles', self.recent_files)

    def closeEvent(self, event):
        print(self.model.setChanged)
        if self.model.setChanged:
            print("is changed, saving?")
            quit_msg = "<b>The document was changed.<br>Do you want to save the changes?</ b>"
            reply = QMessageBox.question(self, 'Save Confirmation', 
                     quit_msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.write_csv_update()
            else:
                print("Settings not saved.")
                return
        else:
            print("nothing changed.")
        self.save_settings()

    def load_recent(self):
        if self.last_files.currentIndex() > 0:
            print(self.last_files.currentText())
            print(self.model.setChanged)
            if self.model.setChanged:
                print("is changed, saving?")
                quit_msg = "<b>The document was changed.<br>Do you want to save the changes?</ b>"
                reply = QMessageBox.question(self, 'Save Confirmation', 
                                             quit_msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.open_csv(self.last_files.currentText())
                else:
                    self.open_csv(self.last_files.currentText())
            else:
                self.open_csv(self.last_files.currentText())

    def open_csv(self, path):
        f = open(path, 'r+b')
        with f:
            df = pd.read_csv(f, sep='\t|;|,|\s+', keep_default_na=False, engine='python',
                             skipinitialspace=True, skip_blank_lines=True)
            f.close()
            self.model = PandasModel(df)
            self.table_view.setModel(self.model)
            self.table_view.resizeColumnsToContents()
            self.table_view.selectRow(0)
            # self.statusBar().showMessage("%s %s" % (path, "loaded"), 0)

    def find_in_table(self):
        self.table_view.clearSelection()
        text = self.line_find.text()
        model = self.table_view.model()
        for column in range(self.model.columnCount()):
            start = model.index(0, column)
            matches = model.match(start, Qt.DisplayRole, text, -1, Qt.MatchContains)
            if matches:
                for index in matches:
                    # print(index.row(), index.column())
                    self.table_view.selectionModel().select(index, QItemSelectionModel.Select)

    def open_file(self, path=None):
        print(self.model.setChanged)
        if  self.model.setChanged == True:
            print("is changed, saving?")
            quit_msg = "<b>The document was changed.<br>Do you want to save the changes?</ b>"
            reply = QMessageBox.question(self, 'Save Confirmation', 
                                         quit_msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.write_csv_update()
            else:
                print("not saved, loading ...")
                return
        path, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.homePath() + "/Dokumente/CSV/",
                                              "CSV Files (*.csv)")
        if path:
            return path

    def load_csv(self):
        file_name = self.open_file()
        if file_name:
            print(file_name + " loaded")
            f = open(file_name, 'r+b')
            with f:
                df = pd.read_csv(f, sep='\t|;|,|\s+', keep_default_na=False, engine='python',
                                 skipinitialspace=True, skip_blank_lines=True)
                f.close()
                self.model = PandasModel(df)
                self.table_view.setModel(self.model)
                self.table_view.resizeColumnsToContents()
                self.table_view.selectRow(0)
        # self.statusBar().showMessage("%s %s" % (fileName, "loaded"), 0)
        self.recent_files.insert(0, file_name)
        self.last_files.insertItem(1, file_name)

    def write_csv(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Open File", self.filename, "CSV Files (*.csv)")
        if file_name:
            print(file_name + " saved")
            f = open(file_name, 'w')
            new_model = self.model
            data_frame = new_model._df.copy()
            data_frame.to_csv(f, sep=',', index=False, header=True, lineterminator='\n')

    def write_csv_update(self):
        if self.filename:
            f = open(self.filename, 'w')
            new_model = self.model
            data_frame = new_model._df.copy()
            data_frame.to_csv(f, sep='\t', index=False, header=False)
            self.model.setChanged = False
            print("%s %s" % (self.filename, "saved"))
            # self.statusBar().showMessage("%s %s" % (self.filename, "saved"), 0)

    def handle_preview(self):
        if self.model.rowCount() == 0:
            self.msg("no rows")
        else:
            dialog = QtPrintSupport.QPrintPreviewDialog()
            dialog.setFixedSize(1000, 700)
            dialog.paintRequested.connect(self.handle_paint_request)
            dialog.exec_()
            print("Print Preview closed")

    def handle_paint_request(self, printer):
        printer.setDocName(self.filename)
        document = QTextDocument()
        cursor = QTextCursor(document)
        model = self.table_view.model()
        table_format = QTextTableFormat()
        table_format.setBorder(0.2)
        table_format.setBorderStyle(3)
        table_format.setCellSpacing(0);
        table_format.setTopMargin(0);
        table_format.setCellPadding(4)
        table = cursor.insertTable(model.rowCount() + 1, model.columnCount(), table_format)
        model = self.table_view.model()

        # get headers
        myheader = []
        for i in range(0, model.columnCount()):
            myheader = model.headerData(i, Qt.Horizontal)
            cursor.insertText(str(myheader))
            cursor.movePosition(QTextCursor.NextCell)
        # get cells
        for row in range(0, model.rowCount()):
            for col in range(0, model.columnCount()):
                index = model.index(row, col)
                cursor.insertText(str(index.data()))
                cursor.movePosition(QTextCursor.NextCell)
        document.print_(printer)

 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = BPDialog('Raster Line Attributes')
    main.show()
    if len(sys.argv) > 1:
        main.open_csv(sys.argv[1])

    sys.exit(app.exec_())