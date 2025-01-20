# BERA Tools

BERA Tools is successor of [Forest Line Mapper](https://github.com/appliedgrg/flm). It is a toolset for enhanced delineation and attribution of linear disturbances in forests.

<!--![Banner](docs/files/images/BERALogo.png)-->

## [Installation](https://beratools.readthedocs.io/en/latest/installation.html)

BERA Tools is built upon open-source Python libraries. Anaconda is used to manage runtime environments.

Installation Steps:

- Install Miniconda. Download Miniconda from [Miniconda](https://docs.anaconda.com/miniconda/) and install on your machine.
- Launch **Anaconda Promt**. Run the following command to create a new environment. **BERA Tools** will be installed in the new environment at the same time. Download the file [conda_environment.yml](https://github.com/RichardQZeng/BTools/blob/main/conda_environment.yml) first.

   ```bash
   $ conda env create -f conda_environment.yml
   ```

   Wait until the installation is done.
- Activate the **bera** environment and launch BERA Tools:

  ```bash
  $ conda activate bera
  $ beratools
  ```


## [User Guide](http://beratools.beraproject.org/)

Check the user guide for more information.

## [Technical Documentation](https://beratools.readthedocs.io/en/latest/)

BERA Tools provides a series of tools for forest lines processing. Please refer to the technical documentation for programming APIs and algorithms details.

## Credits

This tool is part of the [**Boreal Ecosystem Recovery and Assessment (BERA)**](http://www.beraproject.org/) Project, and is being actively developed by the [**Applied Geospatial Research Group**](https://www.appliedgrg.ca/).

![Logos](docs/files/images/BERALogo.png)
*Copyright (C) 2024  Applied Geospatial Research Group*