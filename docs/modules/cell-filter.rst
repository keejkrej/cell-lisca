cell-filter: Cell Counting & Extraction
=======================================

cell-filter is the second module in the Cell-LISCA pipeline, responsible for counting cells in patterns and extracting sequences with segmentation masks.

Overview
--------

The cell-filter module works with the HDF5 file created by cell-pattern and performs two main operations:

1. **analysis**: Count cells in each pattern across all frames
2. **extract**: Extract sequences that match criteria and add segmentation masks

The module uses Cellpose for cell segmentation and stores all results in the cumulative HDF5 file.

Commands
--------

analysis
~~~~~~~~

Analyze cell counts for all patterns across all frames in the specified FOV range. Results are appended to the H5 file for later processing.

.. code-block:: bash

   cell-filter analysis [OPTIONS]

Required Options:
^^^^^^^^^^^^^^^^^

.. option:: --cells <PATH>

   Path to cells ND2 file.

.. option:: --h5 <PATH>

   Path to H5 file with bounding boxes (analysis will be appended).

.. option:: --range <RANGE>

   FOV range to process (e.g., '0:10').

Optional Options:
^^^^^^^^^^^^^^^^^

.. option:: --nuclei-channel <INT>

   Channel index for nuclei. Default: ``1``

.. option:: --min-size <INT>

   Minimum object size for nuclei detection. Default: ``15``

.. option:: --debug

   Enable debug logging.

Examples:
^^^^^^^^^

.. code-block:: bash

   # Analyze FOVs 0-10
   cell-filter analysis \
     --cells cells.nd2 \
     --h5 bounding_boxes.h5 \
     --range "0:10" \
     --nuclei-channel 1

extract
~~~~~~~

Extract sequences from analysis results based on cell count criteria. Reads analysis data from H5, finds sequences matching criteria, and extracts cropped image data with segmentation.

.. code-block:: bash

   cell-filter extract [OPTIONS]

Required Options:
^^^^^^^^^^^^^^^^^

.. option:: --cells <PATH>

   Path to cells ND2 file.

.. option:: --h5 <PATH>

   Path to H5 file with bounding boxes and analysis data.

Optional Options:
^^^^^^^^^^^^^^^^^

.. option:: --nuclei-channel <INT>

   Channel index for nuclei. Default: ``1``

.. option:: --n-cells <INT>

   Target number of cells per pattern. Default: ``4``

.. option:: --tolerance-gap <INT>

   Max consecutive frames with wrong cell count. Default: ``6``

.. option:: --min-frames <INT>

   Minimum frames for valid sequences. Default: ``20``

.. option:: --debug

   Enable debug logging.

Examples:
^^^^^^^^^

.. code-block:: bash

   # Extract sequences with exactly 4 cells
   cell-filter extract \
     --cells cells.nd2 \
     --h5 bounding_boxes.h5 \
     --n-cells 4 \
     --tolerance-gap 6 \
     --min-frames 20

Data Flow
----------

The cell-filter module follows a two-stage process:

.. graphviz::

    digraph cellfilter_flow {
        rankdir=LR;
        node [shape=box, style=rounded];
        
        input [label="HDF5 with\nbounding boxes"];
        analysis [label="Cell Count\nAnalysis"];
        with_counts [label="HDF5 with\ncell counts"];
        extract [label="Sequence\nExtraction"];
        with_seg [label="HDF5 with\nsegmentation"];
        
        input -> analysis;
        analysis -> with_counts;
        with_counts -> extract;
        extract -> with_seg;
    }

HDF5 Structure
--------------

After Cell-Pattern
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   bounding_boxes.h5
   ├── /bounding_boxes/
   │   └── (pattern location data)
   └── /metadata/

 After Analysis
~~~~~~~~~~~~~~

.. code-block:: text

   bounding_boxes.h5
   ├── /bounding_boxes/
   │   └── (pattern location data)
   ├── /analysis/
   │   ├── fov_index (int32 array)
   │   ├── pattern_id (int32 array)
   │   ├── frame_index (int32 array)
   │   └── cell_count (int32 array)
   └── /analysis/metadata/
       ├── cells_path (string)
       ├── nuclei_channel (int)
       ├── min_size (int)
       ├── processed_fovs (list)
       └── creation_time (string)

 After Extraction
~~~~~~~~~~~~~~~~~

.. code-block:: text

   bounding_boxes.h5
   ├── /bounding_boxes/
   │   └── (pattern location data)
   ├── /analysis/
   │   └── (cell count data)
   └── /extracted/
       ├── fov_000/
       │   ├── pattern_000/
       │   │   ├── seq_000/
       │   │   │   ├── data (n_frames, n_channels+2, h, w)
       │   │   │   ├── channels (list)
       │   │   │   ├── start_frame (int)
       │   │   │   ├── end_frame (int)
       │   │   │   ├── bbox_x (int)
       │   │   │   ├── bbox_y (int)
       │   │   │   ├── bbox_width (int)
       │   │   │   └── bbox_height (int)
       │   │   └── seq_001/...
       │   └── pattern_001/...
       └── fov_001/...
   └── /extracted attrs:
       ├── n_cells (int)
       ├── tolerance_gap (int)
       ├── min_frames (int)
       └── creation_time (string)

Algorithm Details
-----------------

Cell Counting Algorithm
~~~~~~~~~~~~~~~~~~~~~~~

1. **Load FOV**: Read specific field of view from ND2
2. **Load Patterns**: Get bounding boxes for all patterns in FOV
3. **Iterate Frames**: Process each frame in the timelapse
4. **Crop Pattern**: Extract region around each pattern
5. **Segment Nuclei**: Use size filtering to identify nuclei
6. **Count Objects**: Count nuclei in each pattern
7. **Store Results**: Save frame-by-frame counts to HDF5

Sequence Extraction Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Load Analysis**: Read cell counts from HDF5
2. **Find Valid Sequences**: Identify continuous sequences matching criteria
   - Target cell count (e.g., exactly 4 cells)
   - Allow tolerance gaps (e.g., up to 6 frames with wrong count)
   - Minimum sequence length (e.g., 20 frames)
3. **Extract Images**: For each valid sequence:
   - Crop all channels to pattern bounding box
   - Run Cellpose segmentation on nuclei channel
   - Append segmentation as last channel
4. **Store Sequences**: Save to HDF5 with metadata

Cellpose Segmentation
~~~~~~~~~~~~~~~~~~~~~

- **Model**: Uses Cellpose's cyto2 model by default
- **Input**: Nuclei channel from cropped sequences
- **Output**: Segmentation mask with local cell IDs
- **Channels**: Background=0, cells=1,2,3,...

API Usage
---------

Python API
~~~~~~~~~~

.. code-block:: python

   from cell_filter.analysis import Analyzer
   from cell_filter.extract import Extractor

   # Analyze cell counts
   analyzer = Analyzer(
       cells_path="cells.nd2",
       h5_path="bounding_boxes.h5",
       nuclei_channel=1
   )
   
   summary = analyzer.analyze(
       start_fov=0,
       end_fov=10,
       min_size=15
   )
   
   # Extract sequences
   extractor = Extractor(
       cells_path="cells.nd2",
       h5_path="bounding_boxes.h5",
       nuclei_channel=1
   )
   
   summary = extractor.extract(
       n_cells=4,
       tolerance_gap=6,
       min_frames=20
   )

Core Classes
~~~~~~~~~~~~

.. autoclass:: cell_filter.analysis.core.Analyzer
   :members:
   :undoc-members:

.. autoclass:: cell_filter.extract.core.Extractor
   :members:
   :undoc-members:

.. autoclass:: cell_filter.core.count.CellposeCounter
   :members:
   :undoc-members:

.. autoclass:: cell_filter.core.segmentation.CellposeSegmenter
   :members:
   :undoc-members:

Exporting to NPY
-----------------

After extraction, sequences need to be exported to NPY format for cell-grapher:

.. code-block:: python

   import h5py
   import numpy as np
   
   # Load extracted data
   with h5py.File('bounding_boxes.h5', 'r') as f:
       # Access a specific sequence
       seq_data = f['/extracted/fov_000/pattern_000/seq_000/data'][:]
       
   # Save as NPY
   np.save('sequence.npy', seq_data)

The NPY file has shape: ``(n_frames, n_channels+2, height, width)``

Performance Considerations
--------------------------

Memory Usage
~~~~~~~~~~~~

- Cellpose segmentation benefits from GPU acceleration
- Large sequences may require significant memory
- Consider processing in batches for very large datasets

Processing Speed
~~~~~~~~~~~~~~~~

- Most time is spent on Cellpose segmentation
- Use GPU for 5-10x speedup
- Adjust Cellpose model size for speed/quality tradeoff

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **GPU Acceleration**: Install PyTorch with CUDA support
2. **Batch Processing**: Process multiple patterns simultaneously
3. **Model Selection**: Use smaller Cellpose models for faster processing
4. **Sequence Filtering**: Be strict with criteria to reduce data size

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **No Cells Detected**
   - Check nuclei channel selection
   - Adjust min_size parameter
   - Verify image contrast

2. **Too Many/Few Cells**
   - Adjust Cellpose diameter parameter
   - Try different Cellpose model
   - Check for segmentation errors

3. **No Valid Sequences**
   - Relax tolerance_gap parameter
   - Lower min_frames requirement
   - Check if n_cells criteria is realistic

4. **Memory Errors**
   - Process fewer FOVs at once
   - Use smaller sequences
   - Ensure sufficient RAM

Integration with Pipeline
------------------------

The cell-filter module is the second step in the Cell-LISCA pipeline:

1. **cell-pattern extract** → Creates HDF5 with bounding boxes
2. **cell-filter analysis** → Adds cell count data
3. **cell-filter extract** → Adds segmented sequences
4. **Export to NPY** → For cell-grapher processing

The cumulative HDF5 file serves as the central data store, with each module adding its analysis results while preserving all previous data.
