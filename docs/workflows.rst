Workflows and Examples
======================

This section provides detailed workflows and examples for common analysis tasks using Cell-LISCA.

Complete Analysis Pipeline
--------------------------

This workflow demonstrates the full Cell-LISCA pipeline from raw ND2 files to final analysis results.

.. code-block:: python

   #!/usr/bin/env python
   """Complete Cell-LISCA analysis pipeline."""
   
   import subprocess
   import h5py
   import numpy as np
   from pathlib import Path
   import pickle
   
   # Configuration
   config = {
       "patterns_file": "data/patterns.nd2",
       "cells_file": "data/cells.nd2",
       "nuclei_channel": 1,
       "n_cells": 4,
       "min_size": 15,
       "tolerance_gap": 6,
       "min_frames": 20,
       "fov_range": "0:5"
   }
   
   def run_pipeline(config):
       """Run the complete Cell-LISCA pipeline."""
       
       # Step 1: Pattern detection
       print("Step 1: Detecting patterns...")
       result = subprocess.run([
           "cell-pattern", "extract",
           "--patterns", config["patterns_file"],
           "--cells", config["cells_file"],
           "--nuclei-channel", str(config["nuclei_channel"]),
           "--output", "./bounding_boxes.h5"
       ], capture_output=True, text=True)
       
       if result.returncode != 0:
           print(f"Pattern detection failed: {result.stderr}")
           return
       
       # Step 2: Cell count analysis
       print("Step 2: Analyzing cell counts...")
       result = subprocess.run([
           "cell-filter", "analysis",
           "--cells", config["cells_file"],
           "--h5", "./bounding_boxes.h5",
           "--nuclei-channel", str(config["nuclei_channel"]),
           "--range", config["fov_range"],
           "--min-size", str(config["min_size"])
       ], capture_output=True, text=True)
       
       if result.returncode != 0:
           print(f"Analysis failed: {result.stderr}")
           return
       
       # Step 3: Data extraction
       print("Step 3: Extracting sequences...")
       result = subprocess.run([
           "cell-filter", "extract",
           "--cells", config["cells_file"],
           "--h5", "./bounding_boxes.h5",
           "--nuclei-channel", str(config["nuclei_channel"]),
           "--n-cells", str(config["n_cells"]),
           "--tolerance-gap", str(config["tolerance_gap"]),
           "--min-frames", str(config["min_frames"])
       ], capture_output=True, text=True)
       
       if result.returncode != 0:
           print(f"Extraction failed: {result.stderr}")
           return
       
       # Step 4: Export to NPY and run analysis
       print("Step 4: Exporting and analyzing...")
       export_and_analyze("./bounding_boxes.h5")
       
       print("Pipeline complete!")
   
   def export_and_analyze(h5_file):
       """Export sequences to NPY and run cell-grapher."""
       
       with h5py.File(h5_file, "r") as f:
           if "/extracted" not in f:
               print("No extracted data found")
               return
           
           # Find first available sequence
           for fov in f["/extracted"]:
               for pattern in f[f"/extracted/{fov}"]:
                   for seq in f[f"/extracted/{fov}/{pattern}"]:
                       # Export to NPY
                       data = f[f"/extracted/{fov}/{pattern}/{seq}/data"][:]
                       npy_file = f"{fov}_{pattern}_{seq}.npy"
                       np.save(npy_file, data)
                       
                       # Run cell-grapher
                       print(f"Analyzing {npy_file}...")
                       subprocess.run([
                           "cell-grapher",
                           "--input", npy_file,
                           "--output", f"analysis_{fov}_{pattern}_{seq}"
                       ])
                       
                       # Run tensionmap on segmentation
                       segmentation = data[:, -1, :, :]  # Last channel
                       seg_file = f"seg_{fov}_{pattern}_{seq}.npy"
                       np.save(seg_file, segmentation)
                       
                       subprocess.run([
                           "cell-tensionmap", "run",
                           "--mask", seg_file,
                           "--output", f"tension_{fov}_{pattern}_{seq}.pkl"
                       ])
                       
                       break
                   break
               break
   
   if __name__ == "__main__":
       run_pipeline(config)

Batch Processing Multiple Datasets
----------------------------------

This example shows how to process multiple datasets automatically.

.. code-block:: python

   import os
   import subprocess
   from pathlib import Path
   
   def batch_process datasets):
       """Process multiple datasets in batch."""
       
       results = []
       
       for dataset in datasets:
           print(f"\nProcessing dataset: {dataset['name']}")
           
           # Create output directory
           output_dir = Path(f"results/{dataset['name']}")
           output_dir.mkdir(parents=True, exist_ok=True)
           
           try:
               # Run pattern detection
               subprocess.run([
                   "cell-pattern", "extract",
                   "--patterns", dataset["patterns"],
                   "--cells", dataset["cells"],
                   "--nuclei-channel", str(dataset["nuclei_channel"]),
                   "--output", str(output_dir / "bounding_boxes.h5")
               ], check=True)
               
               # Run analysis
               subprocess.run([
                   "cell-filter", "analysis",
                   "--cells", dataset["cells"],
                   "--h5", str(output_dir / "bounding_boxes.h5"),
                   "--nuclei-channel", str(dataset["nuclei_channel"]),
                   "--range", dataset["fov_range"],
                   "--min-size", str(dataset["min_size"])
               ], check=True)
               
               # Run extraction
               subprocess.run([
                   "cell-filter", "extract",
                   "--cells", dataset["cells"],
                   "--h5", str(output_dir / "bounding_boxes.h5"),
                   "--nuclei-channel", str(dataset["nuclei_channel"]),
                   "--n-cells", str(dataset["n_cells"]),
                   "--tolerance-gap", str(dataset["tolerance_gap"]),
                   "--min-frames", str(dataset["min_frames"])
               ], check=True)
               
               results.append({
                   "name": dataset["name"],
                   "status": "success",
                   "output": str(output_dir)
               })
               
           except subprocess.CalledProcessError as e:
               results.append({
                   "name": dataset["name"],
                   "status": "failed",
                   "error": str(e)
               })
       
       # Print summary
       print("\nBatch processing complete:")
       for result in results:
           print(f"{result['name']}: {result['status']}")
   
   # Example dataset list
   datasets = [
       {
           "name": "experiment_1",
           "patterns": "data/exp1_patterns.nd2",
           "cells": "data/exp1_cells.nd2",
           "nuclei_channel": 1,
           "fov_range": "0:10",
           "min_size": 15,
           "n_cells": 4,
           "tolerance_gap": 6,
           "min_frames": 20
       },
       {
           "name": "experiment_2",
           "patterns": "data/exp2_patterns.nd2",
           "cells": "data/exp2_cells.nd2",
           "nuclei_channel": 2,
           "fov_range": "0:5",
           "min_size": 20,
           "n_cells": 6,
           "tolerance_gap": 8,
           "min_frames": 30
       }
   ]
   
   batch_process(datasets)

Custom Analysis Workflow
------------------------

This example shows how to create a custom analysis workflow using the Python APIs directly.

.. code-block:: python

   import numpy as np
   import h5py
   from cell_pattern.core import Patterner
   from cell_filter.analysis import Analyzer
   from cell_filter.extract import Extractor
   from cell_grapher.pipeline import analyze_cell_filter_data
   from cell_tensionmap.integration import run_tensionmap_analysis
   
   class CustomWorkflow:
       """Custom analysis workflow with specific requirements."""
       
       def __init__(self, patterns_path, cells_path, output_dir):
           self.patterns_path = patterns_path
           self.cells_path = cells_path
           self.output_dir = Path(output_dir)
           self.output_dir.mkdir(parents=True, exist_ok=True)
           
           # Custom parameters
           self.params = {
               "nuclei_channel": 1,
               "target_cells": [4, 5, 6],  # Multiple target cell counts
               "min_frames": 15,
               "quality_threshold": 0.8
           }
       
       def run(self):
           """Run the custom workflow."""
           
           # Step 1: Pattern detection with quality check
           patterner = Patterner(
               self.patterns_path,
               self.cells_path,
               self.params["nuclei_channel"]
           )
           
           # Extract bounding boxes
           summary = patterner.process_all_fovs_bounding_boxes()
           
           # Custom quality filter
           if summary["total_patterns"] < 10:
               raise ValueError("Too few patterns detected")
           
           patterner.close()
           
           # Step 2: Cell count analysis
           analyzer = Analyzer(
               self.cells_path,
               str(self.output_dir / "bounding_boxes.h5"),
               self.params["nuclei_channel"]
           )
           
           analysis_summary = analyzer.analyze(
               start_fov=0,
               end_fov=10,
               min_size=15
           )
           
           # Step 3: Extract sequences for multiple cell counts
           all_sequences = []
           
           for n_cells in self.params["target_cells"]:
               extractor = Extractor(
                   self.cells_path,
                   str(self.output_dir / "bounding_boxes.h5"),
                   self.params["nuclei_channel"]
               )
               
               extract_summary = extractor.extract(
                   n_cells=n_cells,
                   tolerance_gap=6,
                   min_frames=self.params["min_frames"]
               )
               
               all_sequences.extend(extract_summary["sequences"])
           
           # Step 4: Analyze each sequence
           results = []
           
           for seq_info in all_sequences:
               # Export to NPY
               npy_file = self.export_sequence(seq_info)
               
               # Run cell-grapher
               grapher_results = analyze_cell_filter_data(
                   npy_path=npy_file,
                   output_dir=str(self.output_dir / f"analysis_{seq_info['id']}"),
                   start_frame=seq_info["start_frame"],
                   end_frame=seq_info["end_frame"]
               )
               
               # Run tensionmap
               tension_results = self.run_tension_analysis(npy_file)
               
               # Combine results
               results.append({
                   "sequence": seq_info,
                   "tracking": grapher_results,
                   "tension": tension_results
               })
           
           # Step 5: Generate summary report
           self.generate_report(results)
           
           return results
       
       def export_sequence(self, seq_info):
           """Export a sequence to NPY format."""
           # Implementation details...
           pass
       
       def run_tension_analysis(self, npy_file):
           """Run tension map analysis on segmentation."""
           data = np.load(npy_file)
           segmentation = data[:, -1, :, :]
           
           model = run_tensionmap_analysis(
               segmentation,
               is_labelled=True,
               optimiser="nlopt"
           )
           
           return model
       
       def generate_report(self, results):
           """Generate analysis report."""
           # Implementation details...
           pass

Interactive Data Exploration
---------------------------

This example shows how to use cell-viewer programmatically for data exploration.

.. code-block:: python

   from cell_viewer.main import CellViewer
   import numpy as np
   
   def explore_sequences(h5_file):
       """Interactively explore extracted sequences."""
       
       # Load sequence list
       sequences = []
       
       with h5py.File(h5_file, "r") as f:
           for fov in f["/extracted"]:
               for pattern in f[f"/extracted/{fov}"]:
                   for seq in f[f"/extracted/{fov}/{pattern}"]:
                       sequences.append({
                           "path": f"/extracted/{fov}/{pattern}/{seq}",
                           "fov": fov,
                           "pattern": pattern,
                           "sequence": seq
                       })
       
       # Launch viewer with first sequence
       if sequences:
           viewer = CellViewer()
           
           # Load first sequence
           first_seq = sequences[0]
           data = h5py.File(h5_file, "r")[first_seq["path"] + "/data"][:]
           npy_file = f"temp_{first_seq['fov']}_{first_seq['pattern']}_{first_seq['sequence']}.npy"
           np.save(npy_file, data)
           
           viewer.load_file(npy_file)
           
           # Set up callback for sequence selection
           def on_sequence_selected(seq_idx):
               # Load selected sequence
               seq = sequences[seq_idx]
               data = h5py.File(h5_file, "r")[seq["path"] + "/data"][:]
               np.save(npy_file, data)
               viewer.load_file(npy_file)
               print(f"Loaded {seq['path']}")
           
           # Add sequence selector to viewer
           viewer.add_sequence_selector(sequences, on_sequence_selected)
           
           # Run interactive session
           viewer.run()
           
           # Cleanup
           import os
           os.remove(npy_file)

Performance Optimization
------------------------

This example demonstrates techniques for optimizing performance with large datasets.

.. code-block:: python

   import multiprocessing as mp
   from functools import partial
   import h5py
   import numpy as np
   
   def process_fov(args):
       """Process a single FOV in parallel."""
       fov_idx, cells_path, h5_path, nuclei_channel = args
       
       # Create analyzer for this FOV
       analyzer = Analyzer(
           cells_path,
           h5_path,
           nuclei_channel
       )
       
       # Process only this FOV
       summary = analyzer.analyze(
           start_fov=fov_idx,
           end_fov=fov_idx + 1,
           min_size=15
       )
       
       return fov_idx, summary
   
   def parallel_analysis(cells_path, h5_path, nuclei_channel, n_workers=4):
       """Run cell count analysis in parallel."""
       
       # Get available FOVs
       with h5py.File(h5_path, "r") as f:
           available_fovs = list(f["/bounding_boxes"].keys())
       
       # Create argument list
       args_list = [
           (int(fov.split("_")[1]), cells_path, h5_path, nuclei_channel)
           for fov in available_fovs
       ]
       
       # Run in parallel
       with mp.Pool(n_workers) as pool:
           results = pool.map(process_fov, args_list)
       
       # Combine results
       combined_summary = {
           "total_fovs": len(results),
           "processed_fovs": [r[0] for r in results],
           "summaries": [r[1] for r in results]
       }
       
       return combined_summary
   
   # Memory-efficient data loading
   class MemoryEfficientLoader:
       """Load data in chunks to minimize memory usage."""
       
       def __init__(self, h5_file, chunk_size=10):
           self.h5_file = h5_file
           self.chunk_size = chunk_size
           
       def load_sequences(self, callback):
           """Load sequences in chunks and apply callback."""
           
           with h5py.File(self.h5_file, "r") as f:
               sequences = list(f["/extracted"].keys())
               
               for i in range(0, len(sequences), self.chunk_size):
                   chunk = sequences[i:i + self.chunk_size]
                   
                   # Load chunk data
                   chunk_data = {}
                   for seq in chunk:
                       chunk_data[seq] = f[f"/extracted/{seq}/data"][:]
                   
                   # Process chunk
                   callback(chunk_data)
                   
                   # Clear memory
                   del chunk_data

Troubleshooting Common Issues
-----------------------------

Data Format Problems
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_nd2_file(file_path):
       """Validate ND2 file format."""
       try:
           import nd2
           
           with nd2.ND2File(file_path) as f:
               print(f"File: {file_path}")
               print(f"Shape: {f.shape}")
               print(f"Channels: {f.channel_names}")
               print(f"Frames: {f.shape[0] if len(f.shape) > 2 else 1}")
               
               # Check requirements
               if len(f.shape) < 3:
                   print("Warning: File has less than 3 dimensions")
               
               if f.shape[0] < 10:
                   print("Warning: Fewer than 10 frames")
               
               return True
               
       except Exception as e:
           print(f"Error validating file: {e}")
           return False
   
   def validate_h5_file(file_path):
       """Validate HDF5 pipeline file."""
       try:
           with h5py.File(file_path, "r") as f:
               print(f"File: {file_path}")
               
               # Check required groups
               required_groups = ["/bounding_boxes", "/analysis", "/extracted"]
               for group in required_groups:
                   if group in f:
                       print(f"✓ Found {group}")
                   else:
                       print(f"✗ Missing {group}")
               
               return True
               
       except Exception as e:
           print(f"Error validating file: {e}")
           return False

Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def optimize_parameters(cells_path, patterns_path, nuclei_channel):
       """Find optimal parameters for cell detection."""
       
       # Test different min_size values
       min_sizes = [10, 15, 20, 25, 30]
       results = {}
       
       for min_size in min_sizes:
           print(f"Testing min_size={min_size}")
           
           analyzer = Analyzer(
               cells_path,
               "test_bounding_boxes.h5",
               nuclei_channel
           )
           
           summary = analyzer.analyze(
               start_fov=0,
               end_fov=1,  # Test on single FOV
               min_size=min_size
           )
           
           # Calculate metrics
           avg_cells = np.mean(summary["cell_counts"])
           std_cells = np.std(summary["cell_counts"])
           
           results[min_size] = {
               "avg_cells": avg_cells,
               "std_cells": std_cells,
               "total_detected": len(summary["cell_counts"])
           }
       
       # Find best parameters
       best_size = min(results.keys(), 
                      key=lambda x: results[x]["std_cells"])
       
       print(f"Optimal min_size: {best_size}")
       return best_size

These workflows provide starting points for common analysis tasks. Feel free to adapt them to your specific needs.
