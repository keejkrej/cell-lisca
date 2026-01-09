Cell-LISCA: Live-cell Imaging of Self-organizing Cellular Arrays
============================================================

Welcome to the documentation for Cell-LISCA, a comprehensive toolkit for analyzing micropatterned timelapse microscopy images, focusing on cell tracking, pattern recognition, and topological transitions in cellular arrays.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   overview
   modules/cell-pattern
   modules/cell-filter
   modules/cell-grapher
   modules/cell-tensionmap
   modules/cell-viewer
   workflows
   api/index
   examples
   contributing

Overview
--------

Cell-LISCA is a multi-package Python workspace that provides specialized tools for processing and analyzing microscopy data of cells grown on micropatterns. The suite consists of five integrated applications that work together to provide a complete analysis pipeline:

1. **cell-pattern**: Pattern detection and annotation for micropatterned microscopy images
2. **cell-filter**: Cell counting, filtering, and data extraction with segmentation
3. **cell-grapher**: Cell tracking, region adjacency graph construction, and T1 transition analysis
4. **cell-tensionmap**: Integration with TensionMap for stress tensor inference
5. **cell-viewer**: Interactive visualization and frame selection for microscopy data

Key Features
------------

- **Pattern Detection**: Automatically identify and annotate micropatterns in microscopy images (cell-pattern)
- **Cell Segmentation**: Advanced cell segmentation using Cellpose (cell-filter)
- **Cell Tracking**: Track cells across timeframes with IoU-based alignment (cell-grapher)
- **Topological Analysis**: Analyze T1 transitions and cellular rearrangements (cell-grapher)
- **Tension Inference**: Calculate cellular stress tensors using VMSI (cell-tensionmap)
- **Interactive Visualization**: User-friendly interface for data exploration (cell-viewer)

Data Flow
---------

The Cell-LISCA pipeline follows a structured data flow:

.. graphviz::

    digraph dataflow {
        rankdir=LR;
        node [shape=box, style=rounded];
        
        input [label="ND2 Files\n(patterns, cells)"];
        pattern [label="cell-pattern\nHDF5 with bounding boxes"];
        filter_analysis [label="cell-filter analysis\nHDF5 with cell counts"];
        filter_extract [label="cell-filter extract\nHDF5 with segmentation"];
        npy_export [label="Export to NPY\nFormat for analysis"];
        grapher [label="cell-grapher\nTracking & graphs"];
        tension [label="cell-tensionmap\nStress inference"];
        viewer [label="cell-viewer\nInteractive viz"];
        
        input -> pattern;
        pattern -> filter_analysis;
        filter_analysis -> filter_extract;
        filter_extract -> npy_export;
        npy_export -> grapher;
        npy_export -> tension;
        npy_export -> viewer;
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
