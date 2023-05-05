import os
import sys
os.environ['QT_API'] = 'pyqt5'
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget
from pyqtlet2 import L, MapWidget


class MapWindow(QWidget):
    def __init__(self):
        # Setting up the widgets and layout
        super().__init__()
        self.mapWidget = MapWidget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.mapWidget)
        self.setLayout(self.layout)
        self.setWindowTitle('pyqtlet2 example')

        # Working with the maps with pyqtlet
        self.map = L.map(self.mapWidget)
        self.map.setView([12.97, 77.59], 10)
        L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png').addTo(self.map)
        self.marker = L.marker([12.934056, 77.610029])
        self.marker.bindPopup('Maps are a treasure.')
        self.map.addLayer(self.marker)

        # Create a icon called markerIcon in the js runtime.
        self.map.runJavaScript('var markerIcon = L.icon({iconUrl: "https://leafletjs.com/examples/custom-icons/leaf-red.png"});', 0)

        # Edit the existing python object by accessing it's jsName property
        self.map.runJavaScript(f'{self.marker.jsName}.setIcon(markerIcon);', 0)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MapWindow()
    sys.exit(app.exec_())