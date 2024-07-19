import os
import sys

os.environ['QT_API'] = 'pyqt5'
from qtpy.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QWidget, QTreeWidget, QTreeWidgetItem,
                            QPushButton, QGroupBox, QDialog, QDialogButtonBox)
from qtpy.QtCore import (Qt, Signal)
from beratools.third_party.pyqtlet2 import L, MapWidget


class MapWindow(QDialog):
    def __init__(self, parent=None):
        # Setting up the widgets and layout
        super(MapWindow, self).__init__(parent)
        self.setWindowTitle('Tiler map')
        self.setGeometry(0, 0, 1200, 800)

        # delete dialog when close
        self.setAttribute(Qt.WA_DeleteOnClose)

        # Add OK/cancel buttons
        self.ok_btn_box = QDialogButtonBox(Qt.Vertical)
        self.ok_btn_box.addButton("Run Tiler", QDialogButtonBox.AcceptRole)
        self.ok_btn_box.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.ok_btn_box.addButton("Help", QDialogButtonBox.HelpRole)

        self.ok_btn_box.buttons()[0].setFixedSize(120, 40)
        self.ok_btn_box.buttons()[1].setFixedSize(120, 40)
        self.ok_btn_box.buttons()[2].setFixedSize(120, 40)

        self.ok_btn_box.accepted.connect(self.run)
        self.ok_btn_box.rejected.connect(self.cancel)
        self.ok_btn_box.helpRequested.connect(self.help)

        self.info_layout = QVBoxLayout()  # layout reserved for tiles info widgets
        self.vbox_group = QVBoxLayout()
        self.vbox_group.addLayout(self.info_layout)
        self.vbox_group.addStretch()
        self.vbox_group.addWidget(self.ok_btn_box, alignment=Qt.AlignCenter)

        groupbox_info = QGroupBox('Tiles')
        groupbox_info.setLayout(self.vbox_group)

        central_widget = QWidget()
        map_layout = QHBoxLayout(central_widget)
        map_layout.addWidget(groupbox_info)

        groupbox_map = QGroupBox('Map')
        self.map_widget = MapWidget()
        self.map_widget.setContentsMargins(10, 10, 10, 10)
        self.vbox_map = QVBoxLayout()
        self.vbox_map.addWidget(self.map_widget)
        groupbox_map.setLayout(self.vbox_map)
        map_layout.addWidget(groupbox_map, 10)
        self.setLayout(map_layout)

        # Working with the maps with pyqtlet
        self.map = L.map(self.map_widget)
        self.map.setView([0, 0], 10)  # this is necessary

        L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png').addTo(self.map)

        # add marker layer
        # self.add_marker_layer()
        self.show()

    def add_polygons_to_map(self, layer_name, polygons, color):
        style = {'fillOpacity': 0.1, 'color': color}
        vars()[layer_name] = L.polygon(polygons, style)
        self.map.addLayer(vars()[layer_name])

        # this works too. addLayer has to be called first
        # self.map.runJavaScript("var stylePoly = {fillColor:'red',color: 'blue',weight:2,fillOpacity:0.8};", 0)
        # self.map.runJavaScript(f'{self.multipolygon.jsName}.setStyle(stylePoly);', 0)

    def add_polylines_to_map(self, polylines, color):
        style = {'color': color}
        lines = L.polyline(polylines, style)
        self.map.addLayer(lines)

    def set_view(self, point, zoom):
        self.map = self.map.setView(point, zoom)

    # bounds is a pair of corner points, LL and UR
    def fit_bounds(self, bounds):
        # self.map.fitBounds(bounds)
        self.map.runJavaScript(f'{self.map.jsName}.fitBounds(bounds);', 0)

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

    def set_tiles_info(self, tiles_info):
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Item", "Value"])
        item = QTreeWidgetItem(['Tiles'])
        for key, value in tiles_info.items():
            child = QTreeWidgetItem([key, str(value)])
            item.addChild(child)

        tree.insertTopLevelItem(0, item)
        tree.expandAll()

        # add to group widget
        self.vbox_group.insertWidget(0, tree)


if __name__ == '__main__':
    # supress web engine logging
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--enable-logging --log-level=3"

    app = QApplication(sys.argv)
    widget = MapWindow()

    # add polygons to map
    polygon_coords_base = [[[17.285044, 78.286671], [16.606174, 80.748015], [17.886816, 83.518482]]]
    widget.add_polygons_to_map(polygon_coords_base, 'blue')

    polygon_coords = [[[17.385044, 78.486671], [16.506174, 80.648015], [17.686816, 83.218482]],
                      [[13.082680, 80.270718], [12.971599, 77.594563], [15.828126, 78.037279]]]
    widget.add_polygons_to_map(polygon_coords, 'red')

    sys.exit(app.exec_())
