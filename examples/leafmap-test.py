import io
import sys

import leafmap.foliumap as leafmap

import geopandas as gpd

from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets
from PyQt5.QtWidgets import QWidget


class MapWindow(QWidget):
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
        self.view.setContentsMargins(30, 30, 30, 30)

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

        self.m = leafmap.Map(center=(40, -100), zoom=4)
        self.m.add_basemap("OpenTopoMap")
        
        gdf = gpd.read_file("https://github.com/opengeos/leafmap/raw/master/examples/data/cable_geo.geojson")
        filepath = "https://raw.githubusercontent.com/opengeos/leafmap/master/examples/data/us_cities.csv"
        self.m.add_gdf(gdf, layer_name="Cable lines")
        # m.add_heatmap(filepath,
        #               latitude="latitude",
        #               longitude='longitude',
        #               value="pop_max",
        #               name="Heat map",
        #               radius=20)
        
        left = 'ESA WorldCover 2021 S2 FCC'
        right = 'ESA WorldCover 2021 S2 TCC'

        # m.split_map(left_layer=left, right_layer=right)
        # m.add_text(left, position='bottomleft')
        # m.add_text(right, position='bottomright')

        # self.view.setHtml(self.m.to_html())

    def add_geojson_to_map(self, geojson):
        style = {
            "stroke": True,
            "color": "#0000ff",
            "weight": 2,
            "opacity": 1,
            "fill": True,
            "fillColor": "#0000ff",
            "fillOpacity": 0.1,
        }

        self.m.add_geojson(geojson, style=style)

    def set_html_to_map(self):
        self.view.setHtml(self.m.to_html())

if __name__ == "__main__":
    App = QtWidgets.QApplication(sys.argv)
    map_window = MapWindow()

    geojson = r'D:\BERA_Tools\training_samples.geojson'
    map_window.add_geojson_to_map(geojson)

    map_window.set_html_to_map()
    map_window.show()

    sys.exit(App.exec())