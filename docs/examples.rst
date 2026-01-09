Examples
========

This section provides practical examples for using Cell-LISCA in various research scenarios.

Example 1: Basic Cell Count Analysis
------------------------------------

This example demonstrates how to count cells in micropatterns and filter for specific cell numbers.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from cell_filter.analysis import Analyzer
   
   # Load data
   analyzer = Analyzer(
       cells_path="data/timelapse.nd2",
       h5_path="bounding_boxes.h5",
       nuclei_channel=1
   )
   
   # Analyze cell counts
   summary = analyzer.analyze(
       start_fov=0,
       end_fov=5,
       min_size=15
   )
   
   # Plot cell count distribution
   plt.figure(figsize=(10, 6))
   plt.hist(summary["cell_counts"], bins=20, alpha=0.7)
   plt.xlabel("Number of Cells per Pattern")
   plt.ylabel("Frequency")
   plt.title("Cell Count Distribution")
   plt.savefig("cell_count_distribution.png")
   
   # Find patterns with exactly 4 cells
   four_cell_patterns = [
       (fov, pattern, frame) 
       for fov, pattern, frame, count in summary["records"]
       if count == 4
   ]
   
   print(f"Found {len(four_cell_patterns)} instances with 4 cells")

Example 2: Tracking Cell Divisions
-----------------------------------

This example shows how to track cell divisions using cell-grapher.

.. code-block:: python

   from cell_grapher.pipeline import analyze_cell_filter_data
   import pandas as pd
   
   # Run tracking analysis
   results = analyze_cell_filter_data(
       npy_path="sequence_with_divisions.npy",
       yaml_path="sequence_with_divisions.yaml",
       output_dir="./division_analysis",
       start_frame=0,
       end_frame=100
   )
   
   # Load tracking data
   tracks = pd.read_csv("./division_analysis/cell_tracks.csv")
   
   # Identify divisions (sudden increase in cell count)
   divisions = []
   for frame in range(1, 100):
       cells_frame = tracks[tracks["frame"] == frame]["cell_id"].nunique()
       cells_prev = tracks[tracks["frame"] == frame - 1]["cell_id"].nunique()
       
       if cells_frame > cells_prev:
           divisions.append(frame)
   
   print(f"Detected divisions at frames: {divisions}")
   
   # Visualize tracking results
   plt.figure(figsize=(12, 6))
   plt.plot(tracks.groupby("frame")["cell_id"].nunique())
   plt.scatter(divisions, [tracks[tracks["frame"] == f]["cell_id"].nunique() for f in divisions], 
               color='red', s=100, label='Divisions')
   plt.xlabel("Frame")
   plt.ylabel("Number of Cells")
   plt.legend()
   plt.savefig("cell_divisions.png")

Example 3: Stress Analysis During Contraction
----------------------------------------------

This example demonstrates stress analysis using cell-tensionmap during tissue contraction.

.. code-block:: python

   from cell_tensionmap.integration import run_tensionmap_analysis
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Load segmentation data
   data = np.load("contraction_sequence.npy")
   segmentation = data[:, -1, :, :]  # Last channel
   
   # Analyze stress over time
   stress_tensors = []
   principal_stresses = []
   
   for frame_idx, seg in enumerate(segmentation):
       print(f"Analyzing frame {frame_idx}")
       
       model = run_tensionmap_analysis(
           seg,
           is_labelled=True,
           optimiser="nlopt",
           verbose=False
       )
       
       # Calculate average stress
       avg_stress = np.mean(np.linalg.norm(model.stress_tensors, axis=2))
       stress_tensors.append(model.stress_tensors)
       
       # Principal stresses
       principal = np.linalg.eigvals(model.stress_tensors)
       principal_stresses.append(np.mean(principal, axis=0))
   
   # Plot stress evolution
   principal_stresses = np.array(principal_stresses)
   
   plt.figure(figsize=(10, 6))
   plt.plot(principal_stresses[:, 0], label="Principal Stress 1")
   plt.plot(principal_stresses[:, 1], label="Principal Stress 2")
   plt.xlabel("Frame")
   plt.ylabel("Stress (a.u.)")
   plt.legend()
   plt.title("Stress Evolution During Contraction")
   plt.savefig("stress_evolution.png")

Example 4: Batch Processing Drug Screen
---------------------------------------

This example shows how to process multiple drug conditions automatically.

.. code-block:: python

   import os
   import subprocess
   import pandas as pd
   from pathlib import Path
   
   def analyze_drug_screen(data_dir, output_dir):
       """Analyze multiple drug conditions."""
       
       data_dir = Path(data_dir)
       output_dir = Path(output_dir)
       output_dir.mkdir(exist_ok=True)
       
       # Find all drug conditions
       conditions = {}
       for item in data_dir.iterdir():
           if item.is_dir():
               conditions[item.name] = {
                   "patterns": item / "patterns.nd2",
                   "cells": item / "cells.nd2"
               }
       
       results = []
       
       for drug, paths in conditions.items():
           print(f"\nProcessing {drug}...")
           
           try:
               # Create drug-specific output directory
               drug_dir = output_dir / drug
               drug_dir.mkdir(exist_ok=True)
               
               # Run pipeline
               h5_file = drug_dir / "pipeline.h5"
               
               # Pattern detection
               subprocess.run([
                   "cell-pattern", "extract",
                   "--patterns", str(paths["patterns"]),
                   "--cells", str(paths["cells"]),
                   "--output", str(h5_file)
               ], check=True)
               
               # Cell count analysis
               subprocess.run([
                   "cell-filter", "analysis",
                   "--cells", str(paths["cells"]),
                   "--h5", str(h5_file),
                   "--range", "0:10"
               ], check=True)
               
               # Extract sequences
               subprocess.run([
                   "cell-filter", "extract",
                   "--cells", str(paths["cells"]),
                   "--h5", str(h5_file),
                   "--n-cells", "4"
               ], check=True)
               
               # Analyze results
               summary = analyze_drug_effects(h5_file)
               summary["drug"] = drug
               results.append(summary)
               
           except Exception as e:
               print(f"Failed to process {drug}: {e}")
               results.append({"drug": drug, "error": str(e)})
       
       # Save summary
       df = pd.DataFrame(results)
       df.to_csv(output_dir / "drug_screen_summary.csv", index=False)
       
       return df
   
   def analyze_drug_effects(h5_file):
       """Analyze drug effects from pipeline results."""
       
       # Extract metrics from HDF5
       with h5py.File(h5_file, "r") as f:
           # Get cell counts
           if "/analysis" in f:
               cell_counts = f["/analysis/cell_count"][:]
               avg_cells = np.mean(cell_counts)
               std_cells = np.std(cell_counts)
           else:
               avg_cells = std_cells = 0
           
           # Count extracted sequences
           if "/extracted" in f:
               n_sequences = len(list(f["/extracted"].keys()))
           else:
               n_sequences = 0
       
       return {
           "avg_cell_count": avg_cells,
           "std_cell_count": std_cells,
           "n_sequences": n_sequences
       }

Example 5: Custom Visualization
--------------------------------

This example creates custom visualizations of Cell-LISCA results.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.animation as animation
   from matplotlib.patches import Polygon
   import networkx as nx
   
   def create_custom_video(npy_file, output_file):
       """Create custom visualization video."""
       
       data = np.load(npy_file)
       nuclei = data[:, 1, :, :]  # Nuclei channel
       segmentation = data[:, -1, :, :]  # Segmentation
       
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
       
       def animate(frame):
           ax1.clear()
           ax2.clear()
           
           # Show nuclei
           ax1.imshow(nuclei[frame], cmap="gray")
           ax1.set_title(f"Frame {frame}")
           ax1.axis("off")
           
           # Show segmentation with graph
           ax2.imshow(segmentation[frame], cmap="tab20")
           
           # Build and overlay graph
           graph = build_adjacency_graph(segmentation[frame])
           pos = get_cell_centroids(segmentation[frame])
           
           # Draw edges
           for edge in graph.edges():
               if edge[0] in pos and edge[1] in pos:
                   ax2.plot([pos[edge[0]][0], pos[edge[1]][0]],
                           [pos[edge[0]][1], pos[edge[1]][1]],
                           'w-', alpha=0.5, linewidth=2)
           
           ax2.set_title(f"Cell Graph - Frame {frame}")
           ax2.axis("off")
           
           return ax1, ax2
       
       anim = animation.FuncAnimation(
           fig, animate, frames=len(nuclei),
           interval=100, blit=False
       )
       
       anim.save(output_file, writer="pillow")
       plt.close()
   
   def build_adjacency_graph(segmentation):
       """Build adjacency graph from segmentation."""
       import skimage.measure as measure
       import skimage.segmentation as seg
       
       # Find boundaries
       boundaries = seg.find_boundaries(segmentation)
       
       # Build graph (simplified)
       graph = nx.Graph()
       
       # Add nodes
       for cell_id in np.unique(segmentation):
           if cell_id > 0:
               graph.add_node(cell_id)
       
       # Add edges (simplified - would need proper boundary analysis)
       # This is a placeholder implementation
       return graph
   
   def get_cell_centroids(segmentation):
       """Get centroid positions for each cell."""
       props = measure.regionprops(segmentation)
       return {prop.label: prop.centroid[::-1] for prop in props}

Example 6: Quality Control Metrics
----------------------------------

This example implements quality control for Cell-LISCA analysis.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from skimage import measure
   
   def quality_control_report(npy_file, yaml_file=None):
       """Generate quality control report for NPY data."""
       
       data = np.load(npy_file)
       n_frames, n_channels, height, width = data.shape
       segmentation = data[:, -1, :, :]
       
       report = {
           "file": npy_file,
           "shape": data.shape,
           "metrics": {}
       }
       
       # Check segmentation quality
       for frame in range(0, n_frames, 10):  # Sample every 10th frame
           seg = segmentation[frame]
           
           # Cell count
           n_cells = len(np.unique(seg)) - 1
           
           # Average cell size
           props = measure.regionprops(seg)
           if props:
               avg_size = np.mean([p.area for p in props])
               size_cv = np.std([p.area for p in props]) / avg_size
           else:
               avg_size = size_cv = 0
           
           # Boundary quality
           boundaries = measure.find_boundaries(seg)
           boundary_ratio = np.sum(boundaries) / (height * width)
           
           report["metrics"][frame] = {
               "n_cells": n_cells,
               "avg_cell_size": avg_size,
               "size_cv": size_cv,
               "boundary_ratio": boundary_ratio
           }
       
       # Generate plots
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))
       
       frames = list(report["metrics"].keys())
       
       # Cell count over time
       axes[0, 0].plot(frames, [report["metrics"][f]["n_cells"] for f in frames])
       axes[0, 0].set_title("Cell Count Over Time")
       axes[0, 0].set_xlabel("Frame")
       axes[0, 0].set_ylabel("Number of Cells")
       
       # Average cell size
       axes[0, 1].plot(frames, [report["metrics"][f]["avg_cell_size"] for f in frames])
       axes[0, 1].set_title("Average Cell Size")
       axes[0, 1].set_xlabel("Frame")
       axes[0, 1].set_ylabel("Size (pixels)")
       
       # Size coefficient of variation
       axes[1, 0].plot(frames, [report["metrics"][f]["size_cv"] for f in frames])
       axes[1, 0].set_title("Cell Size Uniformity")
       axes[1, 0].set_xlabel("Frame")
       axes[1, 0].set_ylabel("CV")
       
       # Boundary ratio
       axes[1, 1].plot(frames, [report["metrics"][f]["boundary_ratio"] for f in frames])
       axes[1, 1].set_title("Boundary Ratio")
       axes[1, 1].set_xlabel("Frame")
       axes[1, 1].set_ylabel("Ratio")
       
       plt.tight_layout()
       plt.savefig("quality_control_report.png")
       
       return report

Example 7: Integration with Other Tools
---------------------------------------

This example shows how to integrate Cell-LISCA with other analysis tools.

.. code-block:: python

   import scanpy as sc
   import anndata as ad
   from cell_grapher.pipeline import analyze_cell_filter_data
   
   def export_to_anndata(npy_file, output_file):
       """Export cell tracking data to AnnData for downstream analysis."""
       
       # Run cell-grapher
       results = analyze_cell_filter_data(
           npy_path=npy_file,
           output_dir="./temp_analysis"
       )
       
       # Load tracking data
       tracks = pd.read_csv("./temp_analysis/cell_tracks.csv")
       
       # Create AnnData object
       # Each cell is a sample, each frame is an observation
       observations = []
       
       for cell_id in tracks["cell_id"].unique():
           cell_tracks = tracks[tracks["cell_id"] == cell_id]
           
           # Calculate features
           features = {
               "cell_id": cell_id,
               "n_frames": len(cell_tracks),
               "mean_speed": np.mean(np.sqrt(
                   cell_tracks["x"].diff()**2 + cell_tracks["y"].diff()**2
               )),
               "total_distance": np.sum(np.sqrt(
                   cell_tracks["x"].diff()**2 + cell_tracks["y"].diff()**2
               )),
               "displacement": np.sqrt(
                   (cell_tracks["x"].iloc[-1] - cell_tracks["x"].iloc[0])**2 +
                   (cell_tracks["y"].iloc[-1] - cell_tracks["y"].iloc[0])**2
               )
           }
           
           observations.append(features)
       
       # Create AnnData
       obs_df = pd.DataFrame(observations)
       adata = ad.AnnData(obs=obs_df)
       
       # Save
       adata.write_h5ad(output_file)
       
       return adata
   
   def integrate_with_scanpy(adata_file):
       """Use ScanPy for downstream analysis."""
       
       adata = sc.read_h5ad(adata_file)
       
       # Calculate neighborhood graph
       sc.pp.neighbors(adata, use_rep="X")
       
       # Cluster cells based on behavior
       sc.tl.leiden(adata, resolution=0.5)
       
       # UMAP visualization
       sc.tl.umap(adata)
       sc.pl.umap(adata, color=["leiden", "mean_speed"], save="_cell_behavior.pdf")
       
       return adata

These examples demonstrate various ways to use Cell-LISCA for different research applications. Feel free to adapt and modify them for your specific needs.
