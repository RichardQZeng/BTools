
======================
BERA Tools
======================
.. figure:: ../../Images/BERALogo.png

.. toctree::
   :maxdepth: 2
   :hidden:

   Installation.rst
   Tools/Tools.rst
   ProgrammingGuide.rst
   Bibliography.rst


BERA Tools is a series of script tools for facilitating the high-resolution mapping and studying of forest lines (petroleum exploration corridors in forested areas) via processing canopy height models (LiDAR or photogrammetry derived raster images where pixel-values represent the ground-height of vegetation).


Motivation
-------------------

Given that the process of manually digitizing detailed small-scale (boreal) forest lines is slow and prone to human error, a semi-automated solution is preferred for large-scale application areas. Additionally, high-resolution CHMs allow for improved forest line spatial analysis.

BERA Tools overview
-------------------------

BERA Tools is cut in various modules:

* **Scripts**, core modules,

  * Tools for preparing data and line delineation out of CHM data.

* **Examples**,

  * Example datasets demonstating usage of tools.

* **Applications**,

  * Sample python scripts using tools Python APIs for batch processing.

Dependencies
--------------
BERA Tools is built using ArcGIS Pro Anaconda distribution, you need to install a clone of Arcpy to make BERA Tools run. Please refer to :doc:`Installation` for setting up the environment.

Cite Us
=======

If you use BERA Tools for a publication, please cite it as::

    @misc{BERA Tools,
      author = "Applied Geospatial Research Group",
      title = "Forest Line Mapper",
      howpublished = "\url{https://github.com/appliedgrg/BERATools}",
    }


Credits
-----------------
This tool is part of the Boreal Ecosystem Recovery and Assessment (BERA) Project, and was developed by the Applied Geospatial Research Group.

License
--------------
BERA Tools is release under the GPL 3.0 (GNU General Public License v3.0). Please refer to the LICENSE file contained in the source for complete license description.


Indices and tables
----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
