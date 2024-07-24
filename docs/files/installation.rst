*************
Installation
*************

BERA Tools is developed using Python libraries. To use BERA Tools, you need to install the proper Python environment first.

.. note::
    BERA Tools no longer supports Arcpy

Anaconda
================
Anaconda installation

env
-------------------
First

.. figure:: Images/installation_settings.png
   :align: center



Install Shapely
---------------


Run BERA Tools
=======================

BERA Tools can be launched by ForestLineMapper.bat in BERA Tools root folder. Before launch BERA Tools, open ForestLineMapper.bat in text editor to check the configuration is correct.

.. code-block:: bat

    @echo off
    set scriptName=beratools.py


ArcGIS Pro Upgrades
====================

.. warning::
   warnig

Error might show up related to ``numpy``:

.. code-block:: console

   RuntimeError: module compiled against API version 0xe but this version of numpy is 0xd
