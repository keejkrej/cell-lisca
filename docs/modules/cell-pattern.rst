cell-pattern: Pattern Detection
================================

cell-pattern is the first module in the Cell-LISCA pipeline, responsible for detecting and annotating micropatterns in microscopy images.

Overview
--------

The cell-pattern module provides two main functions:

1. **detect**: Visualize patterns with bounding boxes (for inspection)
2. **extract**: Extract pattern bounding boxes to HDF5 format (for pipeline)

The module uses OpenCV-based contour analysis to identify micropatterns in single-channel images and extracts their bounding box coordinates for downstream analysis.

Commands
--------

detect
~~~~~~

Visualize patterns with bounding boxes and indices. This command is useful for inspecting pattern detection results.

.. code-block:: bash

   cell-pattern detect [OPTIONS]

Options:
^^^^^^^^

.. option:: --patterns <PATH>

   Path to patterns ND2 file (single channel, single frame).
   Default: ``data/20250806_patterns_after.nd2``

.. option:: --cells <PATH>

   Path to cells ND2 file (multi-channel timelapse).
   Default: ``data/20250806_MDCK_timelapse_crop_fov0004.nd2``

.. option:: --nuclei-channel <INT>

   Channel index for nuclei. Default: ``1``

.. option:: --fov <INT>

   Specific FOV to process. Default: ``0``

.. option:: --fov-all

   Process all FOVs in the dataset.

.. option:: --output <PATH>

   Output directory for saving annotated images.

.. option:: --debug

   Enable debug logging.

Examples:
^^^^^^^^^

.. code-block:: bash

   # Process single FOV and display
   cell-pattern detect --patterns patterns.nd2 --cells cells.nd2 --fov 0

   # Process all FOVs and save images
   cell-pattern detect --patterns patterns.nd2 --cells cells.nd2 --fov-all --output ./output/

extract
~~~~~~~

Extract pattern bounding boxes for all FOVs and save to a single HDF5 file. This is the main pipeline entry point.

.. code-block:: bash

   cell-pattern extract [OPTIONS]

Options:
^^^^^^^^

.. option:: --patterns <PATH>

   Path to patterns ND2 file (single channel, single frame).
   Default: ``data/20250806_patterns_after.nd2``

.. option:: --cells <PATH>

   Path to cells ND2 file (multi-channel timelapse).
   Default: ``data/20250806_MDCK_timelapse_crop_fov0004.nd2``

.. option:: --nuclei-channel <INT>

   Channel index for nuclei. Default: ``1``

.. option:: --output <PATH>

   Output HDF5 file path. Default: ``./bounding_boxes_all_fovs.h5``

.. option:: --debug

   Enable debug logging.

Examples:
^^^^^^^^^

.. code-block:: bash

   # Extract all bounding boxes
   cell-pattern extract --patterns patterns.nd2 --cells cells.nd2 --output ./bounding_boxes.h5

Input Requirements
------------------

Patterns File
~~~~~~~~~~~~~

- **Format**: ND2 file
- **Channels**: Exactly 1 channel
- **Frames**: Exactly 1 frame
- **Content**: Micropattern binary masks or contrast patterns

Cells File
~~~~~~~~~~

- **Format**: ND2 file
- **Channels**: Minimum 2 channels
- **Frames**: Variable (timelapse data)
- **Content**: Cell fluorescence/microscopy data

Output Format
-------------

HDF5 Structure
~~~~~~~~~~~~~~

The extract command creates an HDF5 file with the following structure:

.. code-block:: text

   bounding_boxes_all_fovs.h5
   ├── /bounding_boxes/
   │   ├── fov_index (int32 array)
   │   ├── pattern_id (int32 array)
   │   ├── bbox_x (int32 array)
   │   ├── bbox_y (int32 array)
   │   ├── bbox_width (int32 array)
   │   ├── bbox_height (int32 array)
   │   ├── center_x (int32 array)
   │   ├── center_y (int32 array)
   │   ├── area (int32 array)
   │   ├── patterns_paths (string array)
   │   ├── cells_paths (string array)
   │   ├── image_heights (int32 array)
   │   └── image_widths (int32 array)
   └── /metadata/
       ├── total_fovs (int)
       ├── total_patterns (int)
       ├── processed_fovs (list)
       ├── creation_time (string)
       └── fov_000/, fov_001/, ... (per-FOV attributes)

Visualization Output
~~~~~~~~~~~~~~~~~~~~

The detect command produces PNG images showing:

- Detected patterns with green bounding boxes
- Pattern indices (0, 1, 2, ...) displayed above each pattern
- Pattern boundaries extracted from input microscopy data

API Usage
---------

Python API
~~~~~~~~~~

.. code-block:: python

   from cell_pattern.core import Patterner

   # Initialize patterner
   patterner = Patterner(
       patterns_path="patterns.nd2",
       cells_path="cells.nd2",
       nuclei_channel=1
   )

   # Visualize single FOV
   patterner.plot_view(fov_idx=0, output_path="./output/")

   # Process all FOVs
   for fov_idx in range(patterner.n_fovs):
       patterner.plot_view(fov_idx, "./output/")

   # Extract bounding boxes
   summary = patterner.process_all_fovs_bounding_boxes()
   print(f"Processed {summary['total_patterns']} patterns")

   # Clean up
   patterner.close()

Core Classes
~~~~~~~~~~~~

.. autoclass:: cell_pattern.core.pattern.Patterner
   :members:
   :undoc-members:

.. autoclass:: cell_pattern.core.crop.Cropper
   :members:
   :undoc-members:

.. autoclass:: cell_pattern.core.crop.CropperParameters
   :members:
   :undoc-members:

Algorithm Details
-----------------

Pattern Detection Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Load Patterns**: Read single-channel pattern image
2. **Preprocess**: Apply threshold and noise reduction
3. **Contour Detection**: Find contours using OpenCV
4. **Filter Contours**: Remove contours based on size and shape
5. **Extract Bounding Boxes**: Calculate bounding box for each valid contour
6. **Calculate Centers**: Compute centroid of each pattern
7. **Store Results**: Save all data to HDF5 format

Contour Filtering Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Minimum area threshold (configurable)
- Aspect ratio constraints
- Solidity requirements (area / convex hull area)

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **No Patterns Detected**
   - Check file format (must be single-channel, single-frame)
   - Verify pattern contrast
   - Adjust threshold parameters

2. **Too Many/Too Few Patterns**
   - Check image resolution
   - Verify pattern size
   - Adjust filtering criteria

3. **Incorrect Bounding Boxes**
   - Ensure patterns file matches cells file FOVs
   - Check coordinate system alignment

Performance Tips
~~~~~~~~~~~~~~~~

- Use appropriate image resolution
- Process FOVs in parallel for large datasets
- Ensure sufficient disk space for HDF5 output

Integration with Pipeline
------------------------

The cell-pattern module is the first step in the Cell-LISCA pipeline:

1. **cell-pattern extract** → Creates HDF5 with bounding boxes
2. **cell-filter analysis** → Uses bounding boxes for cell counting
3. **cell-filter extract** → Uses bounding boxes for data extraction

The HDF5 file created by cell-pattern is the cumulative data store for the entire pipeline, with each subsequent module adding its analysis results.
