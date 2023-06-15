import os
import sys
os.environ['QT_API'] = 'pyqt5'
from qtpy.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QWidget,
                            QPushButton, QGroupBox, QDialog, QDialogButtonBox)
from qtpy.QtCore import (Qt, Signal)
from beratools.pyqtlet2 import L, MapWidget


class MapWindow(QDialog):
    def __init__(self, parent=None):
        # Setting up the widgets and layout
        super(MapWindow, self).__init__(parent)
        self.setWindowTitle('Tiler map')
        self.setGeometry(0, 0, 1200, 800)

        # delete dialog when close
        self.setAttribute(Qt.WA_DeleteOnClose)

        button_1 = QPushButton(self.tr("Button 1"))
        button_2 = QPushButton(self.tr("Button 2"))
        button_3 = QPushButton(self.tr("Button 3"))

        button_1.setFixedSize(120, 50)
        button_2.setFixedSize(120, 50)
        button_3.setFixedSize(120, 50)

        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setSpacing(20)
        button_layout.addStretch()
        button_layout.addWidget(button_1)
        button_layout.addWidget(button_2)
        button_layout.addWidget(button_3)
        button_layout.addStretch()

        # Add OK/cancel buttons
        self.ok_btn_box = QDialogButtonBox(Qt.Vertical)
        self.ok_btn_box.addButton("Run", QDialogButtonBox.AcceptRole)
        self.ok_btn_box.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.ok_btn_box.addButton("Help", QDialogButtonBox.HelpRole)

        self.ok_btn_box.accepted.connect(self.run)
        self.ok_btn_box.rejected.connect(self.cancel)
        self.ok_btn_box.helpRequested.connect(self.help)

        hbox_btns = QHBoxLayout()
        hbox_btns.addWidget(self.ok_btn_box)

        groupbox = QGroupBox('Tiles')
        groupbox.setLayout(hbox_btns)

        central_widget = QWidget()
        map_layout = QHBoxLayout(central_widget)
        map_layout.addWidget(groupbox)

        self.map_widget = MapWidget()
        self.map_widget.setContentsMargins(30, 30, 30, 30)
        map_layout.addWidget(self.map_widget, 10)
        self.setLayout(map_layout)

        # Working with the maps with pyqtlet
        self.map = L.map(self.map_widget)
        self.map.setView([0, 0], 10)  # this is necessary

        L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png').addTo(self.map)

        # add marker layer
        # self.add_marker_layer()

        self.show()

    def add_polygons_to_map(self, polygons):
        self.multipolygon = L.polygon(polygons)
        self.map.addLayer(self.multipolygon)

    def set_view(self, point, zoom):
        self.map.setView(point, 10)

    def add_marker_layer(self):
        self.marker = L.marker([12.934056, -77.610029])
        self.marker.bindPopup('Maps are a treasure.')
        self.map.addLayer(self.marker)

        # Create a icon called markerIcon in the js runtime.
        self.map.runJavaScript('var markerIcon '
                               '= L.icon({iconUrl: "https://leafletjs.com/examples/custom-icons/leaf-red.png"});', 0)

        # Edit the existing python object by accessing it's jsName property
        self.map.runJavaScript(f'{self.marker.jsName}.setIcon(markerIcon);', 0)

    def accept(self):
        print("Run the tiling.")
        QDialog.accept(self)

    def run(self):
        self.accept()

    def cancel(self):
        print("Tiling canceled.")
        self.reject()

    def help(self):
        print("Help requested.")


if __name__ == '__main__':
    # supress web engine logging
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--enable-logging --log-level=3"

    app = QApplication(sys.argv)
    widget = MapWindow()

    # add polygons to map
    polygon_coords_base = [[[17.285044, 78.286671], [16.606174, 80.748015], [17.886816, 83.518482]]]
    widget.add_polygons_to_map(polygon_coords_base)

    polygon_coords = [[[17.385044, 78.486671], [16.506174, 80.648015], [17.686816, 83.218482]],
                      [[13.082680, 80.270718], [12.971599, 77.594563], [15.828126, 78.037279]]]
    widget.add_polygons_to_map(polygon_coords)

    sys.exit(app.exec_())