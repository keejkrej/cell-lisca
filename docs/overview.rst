Overview
========

Cell-LISCA is a comprehensive analysis suite for micropatterned timelapse microscopy data, designed to study cellular dynamics in controlled geometric environments.

What is Cell-LISCA?
--------------------

Cell-LISCA (Live-cell Imaging of Self-organizing Cellular Arrays) is a multi-package Python workspace that provides end-to-end analysis of microscopy data from cells grown on micropatterned substrates. The software is specifically designed for:

- **Pattern Detection**: Automatic identification of micropatterns in microscopy images
- **Cell Segmentation**: Advanced cell boundary detection using deep learning
- **Cell Tracking**: Following individual cells through time
- **Topological Analysis**: Detecting and analyzing cellular rearrangements
- **Mechanical Analysis**: Inferring cellular stresses and tensions

Architecture
------------

Cell-LISCA consists of five specialized modules that work together in a pipeline:

.. graphviz::

    digraph architecture {
        graph [rankdir=TB, splines=ortho];
        node [shape=record, style=filled, fillcolor=lightblue];
        
        pattern [label="{cell-pattern|Pattern Detection|• Detect micropatterns\n• Extract bounding boxes\n• Save to HDF5}"];
        filter [label="{cell-filter|Cell Counting & Extraction|• Count cells per pattern\n• Filter sequences\n• Add segmentation}"];
        grapher [label="{cell-grapher|Tracking & Analysis|• IoU-based tracking\n• Graph construction\n• T1 transition detection}"];
        tension [label="{cell-tensionmap|Tension Inference|• VMSI integration\n• Stress tensor calculation}"];
        viewer [label="{cell-viewer|Visualization|• Interactive viewing\n• Frame selection\n• Data export}"];
        
        pattern -> filter [label="HDF5"];
        filter -> grapher [label="NPY"];
        filter -> tension [label="Segmentation"];
        filter -> viewer [label="NPY"];
    }

Data Flow and Formats
---------------------

The Cell-LISCA pipeline uses a structured data flow with multiple file formats:

1. **Input**: ND2 microscopy files
2. **Intermediate**: HDF5 cumulative file (bounding boxes → analysis → extraction)
3. **Analysis**: NPY files with segmentation
4. **Output**: Various formats (CSV, PKL, MP4, PNG)

.. code-block:: text

   ND2 Files
   ├── patterns.nd2 (single channel, single frame)
   └── cells.nd2 (multi-channel, timelapse)
       ↓
   HDF5 Pipeline File
   ├── /bounding_boxes/ (pattern locations)
   ├── /analysis/ (cell counts per frame)
   └── /extracted/ (cropped sequences with segmentation)
       ↓
   NPY Analysis Files
   └── (n_frames, n_channels+2, height, width)
       ↓
   Analysis Results
   ├── tracked_cells.npy
   ├── adjacency_graphs.pkl
   ├── t1_transitions.csv
   └── cell_analysis_video.mp4

Key Concepts
------------

Micropatterns
~~~~~~~~~~~~~
Micropatterns are predefined geometric shapes (e.g., squares, circles) printed on substrates to constrain cell growth. Cell-LISCA automatically detects these patterns and analyzes cells within each pattern independently.

Cell Segmentation
~~~~~~~~~~~~~~~~~
Cell segmentation is the process of identifying cell boundaries in images. Cell-LISCA uses Cellpose, a deep learning-based segmentation tool, to generate accurate cell masks for tracking and analysis.

Region Adjacency Graphs
~~~~~~~~~~~~~~~~~~~~~~
A region adjacency graph represents cells as nodes and cell-cell contacts as edges. These graphs are essential for understanding tissue topology and detecting T1 transitions.

T1 Transitions
~~~~~~~~~~~~~~
T1 transitions are topological rearrangements where neighboring cells exchange partners. These are critical events in tissue morphogenesis and can be automatically detected by Cell-LISCA.

VMSI Integration
~~~~~~~~~~~~~~~~
VMSI (Vertex Model Stress Inference) is a method for calculating cellular stress tensors from segmentation data. Cell-LISCA integrates VMSI to provide mechanical insights.

Applications
------------

Cell-LISCA is designed for researchers studying:

- **Cellular Mechanics**: Understanding how cells generate and respond to forces
- **Tissue Morphogenesis**: Studying how tissues change shape during development
- **Collective Cell Behavior**: Analyzing how cell groups coordinate movement
- **Drug Screening**: Testing how compounds affect cell mechanics and behavior
- **Synthetic Biology**: Engineering cellular assemblies with desired properties

Design Principles
-----------------

Modularity
~~~~~~~~~~
Each module can be used independently or as part of the pipeline. This allows researchers to use only the components they need or to integrate Cell-LISCA into existing workflows.

Reproducibility
~~~~~~~~~~~~~~~
All parameters and settings are saved with the data, ensuring that analyses can be reproduced exactly.

Scalability
~~~~~~~~~~~
The pipeline is designed to handle large datasets, with options for parallel processing and memory optimization.

Extensibility
~~~~~~~~~~~~~
The modular architecture makes it easy to add new analysis methods or integrate with other tools.

Comparison with Other Tools
---------------------------

| Feature | Cell-LISCA | CellProfiler | Ilastik | TrackMate |
|---------|------------|--------------|---------|-----------|
| Micropattern detection | ✓ | ✗ | ✗ | ✗ |
| Cell tracking | ✓ | ✓ | Limited | ✓ |
| T1 transition detection | ✓ | ✗ | ✗ | ✗ |
| Tension inference | ✓ | ✗ | ✗ | ✗ |
| GUI for visualization | ✓ | ✓ | ✓ | ✓ |
| Python API | ✓ | ✓ | Limited | Limited |
| Batch processing | ✓ | ✓ | Limited | Limited |

Performance Considerations
--------------------------

Memory Usage
~~~~~~~~~~~~
- HDF5 format allows efficient access to large datasets
- NPY files are loaded into memory for analysis
- Consider available RAM when processing large timelapses

Processing Speed
~~~~~~~~~~~~~~~~
- Cellpose segmentation benefits from GPU acceleration
- IoU calculations in cell-grapher are CPU-intensive
- HDF5 operations are optimized for speed

Storage Requirements
~~~~~~~~~~~~~~~~~~~~
- Raw ND2 files: Largest component
- HDF5 pipeline file: Moderate size (compressed)
- NPY analysis files: Depends on sequence length
- Output files: Generally small (CSV, PKL)

Citation
--------

If you use Cell-LISCA in your research, please cite:

.. code-block:: text

   Cell-LISCA: Cell Migration Studies with Live Cell Imaging of Single Cell Arrays
   Tianyi Cao
   Year: 2024
