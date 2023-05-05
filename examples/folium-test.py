import io
import sys

import folium
from folium import plugins

import pandas as pd
import geopandas
from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initWindow()

    def initWindow(self):
        self.setWindowTitle(self.tr("Folium Example"))
        self.buttonUI()

    def buttonUI(self):
        shortPathButton = QtWidgets.QPushButton(self.tr("Button 1"))
        button2 = QtWidgets.QPushButton(self.tr("Button 2"))
        button3 = QtWidgets.QPushButton(self.tr("Button 3"))

        shortPathButton.setFixedSize(120, 50)
        button2.setFixedSize(120, 50)
        button3.setFixedSize(120, 50)

        self.view = QtWebEngineWidgets.QWebEngineView()
        self.view.setContentsMargins(50, 50, 50, 50)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        lay = QtWidgets.QHBoxLayout(central_widget)

        button_container = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(button_container)
        vlay.setSpacing(20)
        vlay.addStretch()
        vlay.addWidget(shortPathButton)
        vlay.addWidget(button2)
        vlay.addWidget(button3)
        vlay.addStretch()
        lay.addWidget(button_container)
        lay.addWidget(self.view, stretch=1)

        m = folium.Map(
            location=[45.5236, -122.6750], tiles="Stamen Toner", zoom_start=13
        )
        df1 = pd.read_csv(r"D:\PyQT\volcano_db.csv")
        df = df1.loc[:, ("Name", "Country", "Latitude", "Longitude", "Type")]
        df.info()

        # Create point geometries
        geometry = geopandas.points_from_xy(df.Longitude, df.Latitude)
        geo_df = geopandas.GeoDataFrame(
            df[["Name", "Country", "Latitude", "Longitude", "Type"]], geometry=geometry
        )
        geo_df.head()

        # Create a geometry list from the GeoDataFrame
        geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]

        # Iterate through list and add a marker for each volcano, color-coded by its type.
        i = 0
        for coordinates in geo_df_list:
            # Place the markers with the popup labels and data
            folium.Marker(location=coordinates).add_to(m)
            i = i + 1

        heat_data = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]
        plugins.HeatMap(heat_data).add_to(m)
        self.view.setHtml(m.get_root().render())
        

if __name__ == "__main__":
    App = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(App.exec())