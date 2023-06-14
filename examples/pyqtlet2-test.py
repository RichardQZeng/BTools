import os
import sys
os.environ['QT_API'] = 'pyqt5'
from qtpy.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from pyqtlet2 import L, MapWidget


class MapWindow(QWidget):
    def __init__(self):
        # Setting up the widgets and layout
        super().__init__()

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

        central_widget = QWidget()
        map_layout = QHBoxLayout(central_widget)
        map_layout.addWidget(button_container)

        self.map_widget = MapWidget()
        self.map_widget.setContentsMargins(30, 30, 30, 30)
        map_layout.addWidget(self.map_widget, 10)
        self.setLayout(map_layout)
        self.setWindowTitle('Tiler map')
        self.setFixedSize(1200, 800)

        # Working with the maps with pyqtlet
        self.map = L.map(self.map_widget)
        self.map.setView([12.97, 77.59], 10)

        L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png').addTo(self.map)

        self.marker = L.marker([12.934056, 77.610029])
        self.marker.bindPopup('Maps are a treasure.')
        self.map.addLayer(self.marker)

        latlang = [[[17.385044, 78.486671], [16.506174, 80.648015], [17.686816, 83.218482]],
                   [[13.082680, 80.270718], [12.971599, 77.594563], [15.828126, 78.037279]]]
        self.multipolygon = L.polygon(latlang)
        self.map.addLayer(self.multipolygon)

        # Create a icon called markerIcon in the js runtime.
        self.map.runJavaScript('var markerIcon = L.icon({iconUrl: "https://leafletjs.com/examples/custom-icons/leaf-red.png"});', 0)

        # Edit the existing python object by accessing it's jsName property
        self.map.runJavaScript(f'{self.marker.jsName}.setIcon(markerIcon);', 0)
        self.show()

    def add_polygons_to_map(self, polygons):
        self.multipolygon = L.polygon(polygons)
        self.map.addLayer(self.multipolygon)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MapWindow()
    sys.exit(app.exec_())