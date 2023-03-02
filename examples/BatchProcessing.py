
from qgis.core import *


from qgis.gui import (
    QgsLayerTreeMapCanvasBridge,
)


# Supply the path to the qgis install location
QgsApplication.setPrefixPath(r"C:\Users\qingyerichard.zeng\miniconda3\envs\pyqgis\Library\python", True)

# Create a reference to the QgsApplication.
# Setting the second argument to True enables the GUI.  We need
# this since this is a custom application.

qgs = QgsApplication([], True)

# load providers
qgs.initQgis()

from processing.gui.BatchAlgorithmDialog import BatchAlgorithmDialog
from processing.core.Processing import Processing
Processing.initialize()
import processing

algs = dict()
for alg in QgsApplication.processingRegistry().algorithms():
    algs[alg.displayName()] = alg.id()
print(algs)

a = BatchAlgorithmDialog(alg)

a.show()

# Write your code here to load some layers, use processing
# algorithms, etc.

qgs.exec_()

# Finally, exitQgis() is called to remove the
# provider and layer registries from memory
qgs.exitQgis()