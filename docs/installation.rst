Installation
============

Prerequisites
-------------

- Python 3.11 or higher
- Either [uv](https://github.com/astral-sh/uv) package manager (recommended) or conda

Option 1: Installation with uv (Recommended)
--------------------------------------------

1. Clone the repository:

.. code-block:: bash

   git clone <repository-url>
   cd cell-lisca

2. Install all packages using uv:

.. code-block:: bash

   uv sync

3. Install individual packages (if needed):

.. code-block:: bash

   # Install cell-pattern
   cd cell-pattern && uv sync

   # Install cell-filter
   cd cell-filter && uv sync

   # Install cell-grapher
   cd cell-grapher && uv sync

   # Install cell-tensionmap
   cd cell-tensionmap && uv sync

   # Install cell-viewer
   cd cell-viewer && uv sync

Option 2: Installation with conda + pip
---------------------------------------

1. Clone the repository:

.. code-block:: bash

   git clone <repository-url>
   cd cell-lisca

2. Create a new conda environment:

.. code-block:: bash

   conda create -n cell-lisca python=3.11
   conda activate cell-lisca

3. Install each package:

**Cell-filter:**

.. code-block:: bash

   cd cell-filter
   pip install -e .

**Cell-grapher:**

.. code-block:: bash

   cd ../cell-grapher
   pip install -e .

**Cell-viewer:**

.. code-block:: bash

   cd ../cell-viewer
   pip install -e .

**Cell-pattern:**

.. code-block:: bash

   cd ../cell-pattern
   pip install -e .

**Cell-tensionmap:**

.. code-block:: bash

   cd ../cell-tensionmap
   pip install -e .

Verification
------------

To verify installation, run:

.. code-block:: bash

   # Test cell-pattern module
   cell-pattern --help

   # Test cell-filter modules
   cell-filter analysis --help
   cell-filter extract --help

   # Test cell-grapher
   cell-grapher --help

   # Test cell-tensionmap
   cell-tensionmap --help

   # Test cell-viewer
   cell-viewer
   # (should launch the GUI application)

Note: You can also use the ``python -m`` syntax if preferred:

- ``python -m cell_pattern detect`` instead of ``cell-pattern detect``
- ``python -m cell_pattern extract`` instead of ``cell-pattern extract``
- ``python -m cell_filter analysis`` instead of ``cell-filter analysis``
- ``python -m cell_filter extract`` instead of ``cell-filter extract``
- ``python -m cell_grapher`` instead of ``cell-grapher``
- ``python -m cell_tensionmap`` instead of ``cell-tensionmap``
- ``python -m cell_viewer`` instead of ``cell-viewer``

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

- Python >=3.11
- numpy
- matplotlib
- pyyaml

Package-Specific Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**cell-pattern**:

- numpy
- opencv-python
- matplotlib
- nd2
- xarray
- dask
- scikit-image
- scipy
- typer

**cell-filter**:

- numpy
- torch
- matplotlib
- opencv-python
- cellpose>4
- nd2
- xarray
- dask
- scikit-image
- scipy
- tifffile
- networkx
- pyyaml
- typer

**cell-grapher**:

- numpy
- matplotlib
- scikit-image
- networkx
- scipy
- pyyaml
- opencv-python
- btrack

**cell-tensionmap**:

- numpy
- VMSI (TensionMap package)
- typer

**cell-viewer**:

- PySide6
- numpy
- matplotlib
- pyyaml>=6.0.2
