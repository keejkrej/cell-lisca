Quick Start Guide
=================

This guide will walk you through a typical Cell-LISCA analysis workflow from raw microscopy data to cell tracking and tension inference.

Basic Workflow
---------------

1. **Pattern Detection**: Extract bounding boxes for all micropatterns
2. **Cell Count Analysis**: Count cells in each pattern across all frames
3. **Data Extraction**: Extract sequences with segmentation masks
4. **Cell Tracking**: Track cells and analyze topological transitions
5. **Visualization**: Interactively view and select frames

Step 1: Pattern Detection
--------------------------

First, detect and extract pattern bounding boxes from your microscopy data:

.. code-block:: bash

   cell-pattern extract \
     --patterns /path/to/patterns.nd2 \
     --cells /path/to/cells.nd2 \
     --nuclei-channel 1 \
     --output ./bounding_boxes_all_fovs.h5

This creates an HDF5 file containing bounding boxes for all patterns in all fields of view.

Step 2: Cell Count Analysis
----------------------------

Analyze cell counts for all patterns across the entire timelapse:

.. code-block:: bash

   cell-filter analysis \
     --cells /path/to/cells.nd2 \
     --h5 ./bounding_boxes_all_fovs.h5 \
     --nuclei-channel 1 \
     --range 0:10 \
     --min-size 15

This appends cell count data to the HDF5 file for each pattern in every frame.

Step 3: Data Extraction with Segmentation
------------------------------------------

Extract sequences that match your criteria (e.g., patterns with exactly 4 cells):

.. code-block:: bash

   cell-filter extract \
     --cells /path/to/cells.nd2 \
     --h5 ./bounding_boxes_all_fovs.h5 \
     --nuclei-channel 1 \
     --n-cells 4 \
     --tolerance-gap 6 \
     --min-frames 20

This extracts cropped image sequences with Cellpose segmentation masks and adds them to the HDF5 file.

Step 4: Export to NPY Format
-----------------------------

The extracted sequences need to be exported to NPY format for cell-grapher:

.. code-block:: python

   import h5py
   import numpy as np
   
   # Load extracted data from HDF5
   with h5py.File('./bounding_boxes_all_fovs.h5', 'r') as f:
       # Access a specific sequence
       seq_data = f['/extracted/fov_000/pattern_000/seq_000/data'][:]
       
   # Save as NPY for cell-grapher
   np.save('fov_000_pattern_000_seq_000.npy', seq_data)

Step 5: Cell Tracking and Graph Analysis
----------------------------------------

Run cell tracking and T1 transition analysis:

.. code-block:: bash

   cell-grapher \
     --input fov_000_pattern_000_seq_000.npy \
     --output ./analysis

This generates:
- Tracked cell data with global IDs
- Region adjacency graphs
- T1 transition detection
- Visualization video

Step 6: Interactive Visualization
---------------------------------

Launch cell-viewer to explore your data:

.. code-block:: bash

   cell-viewer

Then use the GUI to:
- Open your NPY files
- Navigate through frames
- Select intervals of interest
- Export selected sequences

Complete Example Script
-----------------------

Here's a complete Python script that automates the workflow:

.. code-block:: python

   #!/usr/bin/env python
   """Complete Cell-LISCA workflow example."""
   
   import subprocess
   import h5py
   import numpy as np
   from pathlib import Path
   
   # Configuration
   PATTERNS_FILE = "/path/to/patterns.nd2"
   CELLS_FILE = "/path/to/cells.nd2"
   NUCLEI_CHANNEL = 1
   N_CELLS = 4
   MIN_SIZE = 15
   TOLERANCE_GAP = 6
   MIN_FRAMES = 20
   FOV_RANGE = "0:10"
   
   # Step 1: Pattern detection
   print("Step 1: Detecting patterns...")
   subprocess.run([
       "cell-pattern", "extract",
       "--patterns", PATTERNS_FILE,
       "--cells", CELLS_FILE,
       "--nuclei-channel", str(NUCLEI_CHANNEL),
       "--output", "./bounding_boxes.h5"
   ], check=True)
   
   # Step 2: Cell count analysis
   print("Step 2: Analyzing cell counts...")
   subprocess.run([
       "cell-filter", "analysis",
       "--cells", CELLS_FILE,
       "--h5", "./bounding_boxes.h5",
       "--nuclei-channel", str(NUCLEI_CHANNEL),
       "--range", FOV_RANGE,
       "--min-size", str(MIN_SIZE)
   ], check=True)
   
   # Step 3: Data extraction
   print("Step 3: Extracting sequences...")
   subprocess.run([
       "cell-filter", "extract",
       "--cells", CELLS_FILE,
       "--h5", "./bounding_boxes.h5",
       "--nuclei-channel", str(NUCLEI_CHANNEL),
       "--n-cells", str(N_CELLS),
       "--tolerance-gap", str(TOLERANCE_GAP),
       "--min-frames", str(MIN_FRAMES)
   ], check=True)
   
   # Step 4: Export to NPY
   print("Step 4: Exporting to NPY format...")
   with h5py.File("./bounding_boxes.h5", "r") as f:
       if "/extracted" in f:
           # Find first available sequence
           for fov in f["/extracted"]:
               for pattern in f[f"/extracted/{fov}"]:
                   for seq in f[f"/extracted/{fov}/{pattern}"]:
                       data = f[f"/extracted/{fov}/{pattern}/{seq}/data"][:]
                       output_file = f"{fov}_{pattern}_{seq}.npy"
                       np.save(output_file, data)
                       print(f"Exported {output_file}")
                       break
                   break
               break
   
   # Step 5: Run cell-grapher on first sequence
   npy_files = list(Path(".").glob("fov_*_pattern_*_seq_*.npy"))
   if npy_files:
       first_file = npy_files[0]
       print(f"Step 5: Running cell-grapher on {first_file}...")
       subprocess.run([
           "cell-grapher",
           "--input", str(first_file),
           "--output", "./analysis"
       ], check=True)
   
   print("Workflow complete!")

Tips and Best Practices
------------------------

1. **Data Organization**: Keep all data files in a structured directory hierarchy
2. **Parameter Tuning**: Adjust ``--min-size`` for cell detection based on your data
3. **Memory Management**: Process large datasets in chunks to avoid memory issues
4. **GPU Acceleration**: Install PyTorch with CUDA support for faster Cellpose segmentation
5. **Quality Control**: Use cell-viewer to visually inspect results before downstream analysis

Common Issues
-------------

1. **Pattern Detection Fails**: Check that your patterns file has the correct format (single channel, single frame)
2. **Poor Cell Segmentation**: Adjust Cellpose model type or diameter parameter
3. **Memory Errors**: Reduce the number of frames processed at once
4. **No T1 Transitions Detected**: Check adjacency method sensitivity in cell-grapher

Next Steps
----------

- Explore the :doc:`workflows` section for advanced analysis pipelines
- Check the :doc:`modules/index` for detailed module documentation
- See the :doc:`examples` page for specific use cases
