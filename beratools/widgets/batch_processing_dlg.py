#!/usr/bin/python3
# -*- coding: utf-8 -*-

# QIcon fromtheme
# https://gist.github.com/RichardQZeng/2cf5b6d3d383df2242fda75ddb533baf

import pandas as pd
from PyQt5.QtCore import (Qt, QDir, QItemSelectionModel, QAbstractTableModel, QModelIndex, 
                          QVariant, QSize, QSettings, pyqtSignal)
from PyQt5.QtWidgets import (QMainWindow, QTableView, QApplication, QToolBar, QLineEdit, QComboBox, QAction,
                             QFileDialog, QAbstractItemView, QMessageBox, QWidget, QDockWidget, QFormLayout,
                             QSpinBox, QPushButton, QShortcut, QDialog, QMenuBar, QWidgetAction)
from PyQt5.QtGui import QIcon, QKeySequence, QTextDocument, QTextCursor, QTextTableFormat
from PyQt5 import QtPrintSupport

from . bt_widgets import *

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
        # self.items = self.items[:position] + self.items[position + rows:]
        self.endInsertRows()

    def removeRows(self, position, rows=1, index=QModelIndex()):
        print("\n\t\t ...removeRows() Starting position: '%s'" % position, 'with the total rows to be removed: ', rows)
        self.beginRemoveRows(QModelIndex(), position, position + rows - 1)
        for i in range(rows):
            self._df.drop(self._df.index[position+i], inplace=True)
            print('removed: {}'.format(position+i))

        self.endRemoveRows()


class BP_Dialog(QDialog):
    # signals
    sig_update_tool_widgets = pyqtSignal(int)

    def __init__(self, tool_name, parent=None):
        super(BP_Dialog, self).__init__(parent)
        self.setWindowTitle('Batch Processing')
        self.MaxRecentFiles = 5
        self.windowList = []
        self.recentFiles = []
        self.settings = QSettings('Richard Zeng', 'Batch Processing')
        self.filename = ""
        self.setGeometry(0, 0, 800, 600)
        self.table_view = QTableView()
        self.table_view.verticalHeader().setVisible(True)
        # self.lb.setGridStyle(1)
        self.model = PandasModel()
        self.table_view.setModel(self.model)
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_view.setSelectionBehavior(self.table_view.SelectRows)
        self.table_view.setSelectionMode(self.table_view.ExtendedSelection)

        self.setContentsMargins(10, 10, 10, 10)
        self.createToolBar()
        self.readSettings()
        self.table_view.setFocus()
        # self.statusBar().showMessage("Ready", 0)

        # tableview signals
        self.table_view.clicked.connect(self.table_view_clicked)
        self.table_view.verticalHeader().sectionClicked.connect(self.table_view_vertical_header_clicked)
        QShortcut(Qt.Key_Up, self.table_view, activated=self.table_view_key_up)
        QShortcut(Qt.Key_Down, self.table_view, activated=self.table_view_key_down)

        self.sig_update_tool_widgets.connect(self.update_tool_widgets)

        # create form
        self.tool_widgets = ToolWin(tool_name)

        self.createToolBar()
        vbox = QHBoxLayout()
        vbox.addWidget(self.table_view)
        vbox.addWidget(self.tool_widgets)
        vbox.setMenuBar(self.tbar)
        self.setLayout(vbox)

    def table_view_clicked(self, item):
        print('Row, column:{}, {}'.format(item.row(), item.column()))
        self.sig_update_tool_widgets.emit(item.row())

    def table_view_vertical_header_clicked(self, item):
        print('Horizontal header clicked: {}'.format(item))
        self.sig_update_tool_widgets.emit(item)

    def table_view_key_up(self):
        current_row = self.table_view.selectionModel().selectedRows()[-1].row()
        if current_row >= 1:
            self.table_view.selectRow(current_row-1)
            self.sig_update_tool_widgets.emit(current_row-1)

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
                self.sig_update_tool_widgets.emit(current_row)

            print('remove row {}'.format(i))

        self.model.submit()
    def update_tool_widgets(self, row):
        tool_paramas = self.model.data_row_dict(row)
        self.tool_widgets.update_widgets(tool_paramas)
        print('Update tool parameters for record {}'.format(tool_paramas))


    def table_view_key_down(self):
        current_row = self.table_view.selectionModel().selectedRows()[-1].row()
        if current_row < self.model.rowCount()-1:
            self.table_view.selectRow(current_row+1)
            self.sig_update_tool_widgets.emit(current_row+1)

    def readSettings(self):
        print("reading settings")
        if self.settings.contains("geometry"):
            self.setGeometry(self.settings.value('geometry'))
        if self.settings.contains("recentFiles"):
            self.recentFiles = self.settings.value('recentFiles')
            self.lastFiles.addItem("last Files")
            self.lastFiles.addItems(self.recentFiles[:15])

    def saveSettings(self):
        print("saving settings")
        self.settings.setValue('geometry', self.geometry())
        self.settings.setValue('recentFiles', self.recentFiles)

    def closeEvent(self, event):
        print(self.model.setChanged)
        if  self.model.setChanged == True:
            print("is changed, saving?")
            quit_msg = "<b>The document was changed.<br>Do you want to save the changes?</ b>"
            reply = QMessageBox.question(self, 'Save Confirmation', 
                     quit_msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.writeCSV_update()
            else:
                print("not saved, goodbye ...")
                return
        else:
            print("nothing changed. goodbye")
        self.saveSettings()

    def createToolBar(self):
        openAction = QAction(QIcon.fromTheme("document-open"), "Open", self)
        openAction.triggered.connect(self.loadCSV)
        openAction.setShortcut(QKeySequence.Open)

        saveAction = QAction(QIcon.fromTheme("document-save"), "Save", self)
        saveAction.triggered.connect(self.writeCSV_update)
        saveAction.setShortcut(QKeySequence.Save)

        saveAsAction = QAction(QIcon.fromTheme("document-save-as"), "Save as ...", self)
        saveAsAction.triggered.connect(self.writeCSV)
        saveAsAction.setShortcut(QKeySequence.SaveAs)

        deleteAction = QAction(QIcon.fromTheme("edit-delete"), "Delete records", self)
        deleteAction.triggered.connect(self.table_view_delete_records)
        deleteAction.setShortcut(QKeySequence.Delete)

        self.tbar =  QMenuBar()
        self.tbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.tbar.setFixedHeight(30)
        self.tbar.addAction(openAction)
        self.tbar.addAction(saveAction)
        self.tbar.addAction(saveAsAction)
        self.tbar.addAction(deleteAction)

        self.lastFiles = QComboBox()
        self.lastFiles.setFixedWidth(300)
        self.lastFiles.currentIndexChanged.connect(self.loadRecent)
        empty = QWidgetAction(self)
        empty.setDefaultWidget(self.lastFiles)
        self.tbar.addAction(empty)

        findbyText = QAction(QIcon.fromTheme("edit-find-symbolic"), "find", self, triggered = self.findInTable)
        self.lineFind = QLineEdit()
        self.lineFind.addAction(findbyText, 0)
        self.lineFind.setPlaceholderText("find")
        self.lineFind.setClearButtonEnabled(True)
        self.lineFind.setFixedWidth(250)
        self.lineFind.returnPressed.connect(self.findInTable)
        empty = QWidgetAction(self)
        empty.setDefaultWidget(self.lineFind)
        self.tbar.addAction(empty)

        self.previewAction = QAction(QIcon.fromTheme("document-print-preview"), "print", self)
        self.previewAction.triggered.connect(self.handlePreview)
        self.tbar.addAction(self.previewAction)

    def loadRecent(self):
        if self.lastFiles.currentIndex() > 0:
            print(self.lastFiles.currentText())
            print(self.model.setChanged)
            if self.model.setChanged:
                print("is changed, saving?")
                quit_msg = "<b>The document was changed.<br>Do you want to save the changes?</ b>"
                reply = QMessageBox.question(self, 'Save Confirmation', 
                         quit_msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.openCSV(self.lastFiles.currentText())
                else:
                    self.openCSV(self.lastFiles.currentText())
            else:
                self.openCSV(self.lastFiles.currentText())

    def openCSV(self, path):
        f = open(path, 'r+b')
        with f:
            df = pd.read_csv(f, sep='\t|;|,', keep_default_na=False, engine='python',
                             skipinitialspace=True, skip_blank_lines=True)
            f.close()
            self.model = PandasModel(df)
            self.table_view.setModel(self.model)
            self.table_view.resizeColumnsToContents()
            self.table_view.selectRow(0)
            # self.statusBar().showMessage("%s %s" % (path, "loaded"), 0)

    def findInTable(self):
        self.table_view.clearSelection()
        text = self.lineFind.text()
        model = self.table_view.model()
        for column in range(self.model.columnCount()):
            start = model.index(0, column)
            matches = model.match(start, Qt.DisplayRole, text, -1, Qt.MatchContains)
            if matches:
                for index in matches:
                    print(index.row(), index.column())
                    self.table_view.selectionModel().select(index, QItemSelectionModel.Select)

    def openFile(self, path=None):
        print(self.model.setChanged)
        if  self.model.setChanged == True:
            print("is changed, saving?")
            quit_msg = "<b>The document was changed.<br>Do you want to save the changes?</ b>"
            reply = QMessageBox.question(self, 'Save Confirmation', 
                     quit_msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.writeCSV_update()
            else:
                print("not saved, loading ...")
                return
        path, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.homePath() + "/Dokumente/CSV/","CSV Files (*.csv)")
        if path:
            return path

    def loadCSV(self):
        fileName = self.openFile()
        if fileName:
            print(fileName + " loaded")
            f = open(fileName, 'r+b')
            with f:
                df = pd.read_csv(f, sep='\t|;|,', keep_default_na=False, engine='python',
                                 skipinitialspace=True, skip_blank_lines=True)
                f.close()
                self.model = PandasModel(df)
                self.table_view.setModel(self.model)
                self.table_view.resizeColumnsToContents()
                self.table_view.selectRow(0)
        # self.statusBar().showMessage("%s %s" % (fileName, "loaded"), 0)
        self.recentFiles.insert(0, fileName)
        self.lastFiles.insertItem(1, fileName)

    def writeCSV(self):
        fileName, _ = QFileDialog.getSaveFileName(self, "Open File", self.filename,"CSV Files (*.csv)")
        if fileName:
            print(fileName + " saved")
            f = open(fileName, 'w')
            newModel = self.model
            dataFrame = newModel._df.copy()
            dataFrame.to_csv(f, sep='\t', index = False, header = False)

    def writeCSV_update(self):
        if self.filename:
            f = open(self.filename, 'w')
            newModel = self.model
            dataFrame = newModel._df.copy()
            dataFrame.to_csv(f, sep='\t', index = False, header = False)
            self.model.setChanged = False
            print("%s %s" % (self.filename, "saved"))
            # self.statusBar().showMessage("%s %s" % (self.filename, "saved"), 0)

    def handlePreview(self):
        if self.model.rowCount() == 0:
            self.msg("no rows")
        else:
            dialog = QtPrintSupport.QPrintPreviewDialog()
            dialog.setFixedSize(1000, 700)
            dialog.paintRequested.connect(self.handlePaintRequest)
            dialog.exec_()
            print("Print Preview closed")

    def handlePaintRequest(self, printer):
        printer.setDocName(self.filename)
        document = QTextDocument()
        cursor = QTextCursor(document)
        model = self.table_view.model()
        tableFormat = QTextTableFormat()
        tableFormat.setBorder(0.2)
        tableFormat.setBorderStyle(3)
        tableFormat.setCellSpacing(0);
        tableFormat.setTopMargin(0);
        tableFormat.setCellPadding(4)
        table = cursor.insertTable(model.rowCount() + 1, model.columnCount(), tableFormat)
        model = self.table_view.model()
        ### get headers
        myheaders = []
        for i in range(0, model.columnCount()):
            myheader = model.headerData(i, Qt.Horizontal)
            cursor.insertText(str(myheader))
            cursor.movePosition(QTextCursor.NextCell)
        ### get cells
        for row in range(0, model.rowCount()):
           for col in range(0, model.columnCount()):
               index = model.index( row, col )
               cursor.insertText(str(index.data()))
               cursor.movePosition(QTextCursor.NextCell)
        document.print_(printer)

 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = BP_Dialog('Raster Line Attributes')
    main.show()
    if len(sys.argv) > 1:
        main.openCSV(sys.argv[1])

    sys.exit(app.exec_())