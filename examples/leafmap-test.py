import io
import sys

import leafmap.foliumap as leafmap

import geopandas as gpd

from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets
from PyQt5.QtWidgets import QWidget


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initWindow()

    def initWindow(self):
        self.setWindowTitle(self.tr("Leafmap Example"))
        self.setFixedSize(1500, 800)
        self.buttonUI()

    def buttonUI(self):
        button_1 = QtWidgets.QPushButton(self.tr("Button 1"))
        button_2 = QtWidgets.QPushButton(self.tr("Button 2"))
        button_3 = QtWidgets.QPushButton(self.tr("Button 3"))

        button_1.setFixedSize(120, 50)
        button_2.setFixedSize(120, 50)
        button_3.setFixedSize(120, 50)

        self.view = QtWebEngineWidgets.QWebEngineView()
        self.view.setContentsMargins(50, 50, 50, 50)

        central_widget = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central_widget)

        button_container = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(button_container)
        vlay.setSpacing(20)
        vlay.addStretch()
        vlay.addWidget(button_1)
        vlay.addWidget(button_2)
        vlay.addWidget(button_3)
        vlay.addStretch()
        lay.addWidget(button_container)
        lay.addWidget(self.view, stretch=1)
        self.setLayout(lay)

        m = leafmap.Map(center=(40, -100), zoom=4)
        m.add_basemap("OpenTopoMap")
        
        gdf = gpd.read_file("https://github.com/opengeos/leafmap/raw/master/examples/data/cable_geo.geojson")
        filepath = "https://raw.githubusercontent.com/opengeos/leafmap/master/examples/data/us_cities.csv"
        m.add_gdf(gdf, layer_name="Cable lines") 
        m.add_heatmap(filepath,
                      latitude="latitude",
                      longitude='longitude',
                      value="pop_max",
                      name="Heat map",
                      radius=20)
        
        left = 'ESA WorldCover 2021 S2 FCC'
        right = 'ESA WorldCover 2021 S2 TCC'

        m.split_map(left_layer=left, right_layer=right)
        m.add_text(left, position='bottomleft')
        m.add_text(right, position='bottomright')

        self.view.setHtml(m.to_html())


if __name__ == "__main__":
    App = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(App.exec())