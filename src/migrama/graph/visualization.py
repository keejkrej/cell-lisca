"""
Visualization utilities for cell tracking analysis.
"""

import glob
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class CellVisualizationManager:
    """
    Manages consistent visualization of tracked cells across frames.
    """

    def __init__(self, max_colors: int = 10):
        """
        Initialize visualization manager.
        
        Parameters
        ----------
        max_colors : int
            Maximum number of distinct colors to use
        """
        self.max_colors = max_colors
        self.colors = plt.cm.Set1(np.linspace(0, 1, max_colors))
        self.global_color_map = {}

    def get_color_for_global_id(self, global_id: int) -> tuple[float, ...]:
        """
        Get consistent color for a global cell ID.
        
        Parameters
        ----------
        global_id : int
            Global cell identifier
            
        Returns
        -------
        Tuple[float, ...]
            RGBA color tuple
        """
        if global_id not in self.global_color_map:
            color_idx = (global_id - 1) % len(self.colors)
            self.global_color_map[global_id] = self.colors[color_idx]
        return self.global_color_map[global_id]

    def update_color_map(self, tracking_map: dict[int, int]):
        """
        Update color map with new global IDs.
        
        Parameters
        ----------
        tracking_map : Dict[int, int]
            Mapping from local labels to global IDs
        """
        for global_id in tracking_map.values():
            self.get_color_for_global_id(global_id)

    def create_multi_panel_figure(
        self,
        nuclei_image: np.ndarray,
        cytoplasm_image: np.ndarray,
        tracked_mask: np.ndarray,
        graph: nx.Graph,
        original_mask: np.ndarray,
        tracking_map: dict[int, int],
        frame_idx: int,
        figsize: tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Create a 3-panel visualization figure.
        
        Parameters
        ----------
        nuclei_image : np.ndarray
            Nuclei fluorescence channel
        cytoplasm_image : np.ndarray
            Cytoplasm fluorescence channel
        tracked_mask : np.ndarray
            Segmentation mask with global IDs
        graph : nx.Graph
            Adjacency graph with local labels
        original_mask : np.ndarray
            Original segmentation mask with local labels
        tracking_map : Dict[int, int]
            Mapping from local to global labels
        frame_idx : int
            Frame number for titles
        figsize : Tuple[int, int]
            Figure size
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: Nuclei channel
        axes[0].imshow(nuclei_image, cmap='gray')
        axes[0].set_title(f'Frame {frame_idx} - Nuclei')
        axes[0].axis('off')

        # Panel 2: Cytoplasm channel
        axes[1].imshow(cytoplasm_image, cmap='gray')
        axes[1].set_title(f'Frame {frame_idx} - Cytoplasm')
        axes[1].axis('off')

        # Panel 3: Graph on segmentation with consistent colors
        axes[2].imshow(tracked_mask, cmap='tab20', alpha=0.5)

        # Draw graph with consistent node colors
        self._draw_graph_with_tracking(
            axes[2], graph, original_mask, tracking_map, frame_idx
        )

        plt.tight_layout()
        return fig

    def _plot_nuclei(self, ax: plt.Axes, nuclei_image: np.ndarray, segmentation_mask: np.ndarray, tracking_map: dict[int, int]):
        """Plot nuclei channel without segmentation overlay."""
        # Display grayscale nuclei image only
        ax.imshow(nuclei_image, cmap='gray')
        ax.set_title('Nuclei')
        ax.axis('off')

    def _plot_cytoplasm(self, ax: plt.Axes, cytoplasm_image: np.ndarray, segmentation_mask: np.ndarray, tracking_map: dict[int, int]):
        """Plot cytoplasm channel without segmentation overlay."""
        # Display grayscale cytoplasm image only
        ax.imshow(cytoplasm_image, cmap='gray')
        ax.set_title('Cytoplasm')
        ax.axis('off')

    def _plot_graph(self, ax: plt.Axes, segmentation_mask: np.ndarray, graph: nx.Graph, t1_edge: tuple[int, int] | None, frame_idx: int, tracking_map: dict[int, int] | None = None):
        """Plot segmentation with graph overlay."""
        # Show segmentation as background with colormap (like original)
        ax.imshow(segmentation_mask, cmap='tab20', alpha=0.7)

        # Calculate node positions as centroids
        pos = {}
        node_colors = []
        node_labels = {}

        # Use tracking map if available, otherwise use local labels
        for label in graph.nodes():
            y, x = np.where(segmentation_mask == label)
            if len(x) > 0:
                pos[label] = (np.mean(x), np.mean(y))

                if tracking_map and label in tracking_map:
                    # Use global ID for consistent coloring and labeling
                    global_id = tracking_map[label]
                    node_colors.append(self.get_color_for_global_id(global_id))
                    node_labels[label] = str(global_id)
                else:
                    # Use local label
                    node_colors.append('white')
                    node_labels[label] = str(label)

        # Draw graph components
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_size=300,
            node_color=node_colors, edgecolors='black', linewidths=2
        )
        nx.draw_networkx_edges(
            graph, pos, ax=ax, width=3, edge_color='red'
        )
        nx.draw_networkx_labels(
            graph, pos, node_labels, ax=ax, font_size=12, font_weight='bold'
        )

        # Highlight T1 edge if present
        if t1_edge:
            edge_list = [t1_edge]
            nx.draw_networkx_edges(graph, pos, edgelist=edge_list, ax=ax, edge_color='yellow', width=5)

        # Add edge weights as labels
        edge_labels = {
            (u, v): f"{graph[u][v]['weight']:.0f}" for u, v in graph.edges()
        }
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels, ax=ax, font_size=10
        )

        ax.set_title(f'Segmentation + Graph ({len(graph.edges())} edges)')
        ax.axis('off')

    def _draw_graph_with_tracking(
        self,
        ax: plt.Axes,
        graph: nx.Graph,
        original_mask: np.ndarray,
        tracking_map: dict[int, int],
        frame_idx: int
    ):
        """
        Draw graph with consistent node colors and global ID labels.
        
        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to draw on
        graph : nx.Graph
            NetworkX graph with local labels
        original_mask : np.ndarray
            Original segmentation mask
        tracking_map : Dict[int, int]
            Mapping from local to global labels
        frame_idx : int
            Frame number for title
        """
        # Calculate node positions as centroids
        pos = {}
        node_colors = []
        node_labels = {}

        for label in graph.nodes():
            y, x = np.where(original_mask == label)
            if len(x) > 0:
                pos[label] = (np.mean(x), np.mean(y))
                # Use global ID for consistent coloring and labeling
                global_id = tracking_map[label]
                node_colors.append(self.get_color_for_global_id(global_id))
                node_labels[label] = str(global_id)

        # Draw graph components
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_size=300,
            node_color=node_colors, edgecolors='black', linewidths=2
        )
        nx.draw_networkx_edges(
            graph, pos, ax=ax, width=3, edge_color='red'
        )
        nx.draw_networkx_labels(
            graph, pos, node_labels, ax=ax, font_size=12, font_weight='bold'
        )

        # Add edge weights as labels
        edge_labels = {
            (u, v): f"{graph[u][v]['weight']:.0f}" for u, v in graph.edges()
        }
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels, ax=ax, font_size=10
        )

        ax.set_title(f'Frame {frame_idx} - Segmentation + Graph ({len(graph.edges())} edges)')
        ax.axis('off')

    def create_composite_frame_array(
        self,
        nuclei_image: np.ndarray,
        cytoplasm_image: np.ndarray,
        segmentation_mask: np.ndarray,
        tracking_map: dict[int, int],
        graph: nx.Graph,
        t1_edge: tuple[int, int] | None,
        frame_idx: int
    ) -> np.ndarray:
        """
        Create composite frame visualization as numpy array (in-memory).
        
        Returns
        -------
        np.ndarray
            RGB image array of the composite frame
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        # Create figure similar to save_composite_frame
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Frame {frame_idx} - Cell Analysis', fontsize=16, fontweight='bold')

        # Plot nuclei
        self._plot_nuclei(axes[0], nuclei_image, segmentation_mask, tracking_map)

        # Plot cytoplasm
        self._plot_cytoplasm(axes[1], cytoplasm_image, segmentation_mask, tracking_map)

        # Plot segmentation + graph
        self._plot_graph(axes[2], segmentation_mask, graph, t1_edge, frame_idx, tracking_map)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle

        # Convert figure to numpy array
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Get the RGBA buffer from the figure
        buf = canvas.buffer_rgba()

        # Convert to numpy array (height, width, 4) RGBA
        img_array = np.asarray(buf)

        # Convert RGBA to RGB (drop alpha channel)
        img_rgb = img_array[:, :, :3]

        plt.close(fig)

        return img_rgb


def plot_t1_transition_analysis(
    frame_numbers: list[int],
    t1_edge_weights: list[float],
    output_path: str,
    figsize: tuple[int, int] = (12, 6)
) -> None:
    """
    Create and save T1 transition analysis plot.
    
    Parameters
    ----------
    frame_numbers : List[int]
        Frame numbers
    t1_edge_weights : List[float]
        T1 edge weights for each frame
    output_path : str
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(frame_numbers, t1_edge_weights, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Frame Number')
    plt.ylabel('T1 Edge Weight (boundary length)')
    plt.title('T1 Transition Edge Weight Over Time')
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_t1_data_csv(
    frame_numbers: list[int],
    t1_edge_weights: list[float],
    output_path: str
) -> None:
    """
    Save T1 transition data to CSV file.
    
    Parameters
    ----------
    frame_numbers : List[int]
        Frame numbers
    t1_edge_weights : List[float]
        T1 edge weights for each frame
    output_path : str
        Path to save the CSV file
    """
    import csv

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 't1_edge_weight'])
        for frame, weight in zip(frame_numbers, t1_edge_weights, strict=False):
            writer.writerow([frame, weight])


def create_video_from_frames(
    output_dir: str,
    video_filename: str = "cell_analysis_video.mp4",
    fps: int = 10,
    pattern: str = "frame_*_analysis.png"
) -> str:
    """
    Create a video from frame analysis PNG files.
    
    Parameters
    ----------
    output_dir : str
        Directory containing the frame PNG files
    video_filename : str
        Name of the output video file (default: "cell_analysis_video.mp4")
    fps : int
        Frames per second for the output video (default: 10)
    pattern : str
        Glob pattern to match frame files (default: "frame_*_analysis.png")
        
    Returns
    -------
    str
        Path to the created video file
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for video creation. "
            "Install it with: pip install opencv-python"
        )

    # Find all frame files
    frame_files = sorted(glob.glob(os.path.join(output_dir, pattern)))

    if not frame_files:
        raise ValueError(f"No frame files found matching pattern: {pattern}")

    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        raise ValueError(f"Could not read frame: {frame_files[0]}")

    height, width, layers = first_frame.shape
    size = (width, height)

    # Create video writer
    video_path = os.path.join(output_dir, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, size)

    print(f"Creating video from {len(frame_files)} frames...")

    # Add each frame to the video
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}, skipping...")
            continue

        video_writer.write(frame)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames")

    # Release the video writer
    video_writer.release()

    print(f"Video created successfully: {video_path}")
    return video_path


def create_video_from_arrays(
    video_frames: list[np.ndarray],
    output_dir: str,
    video_filename: str = "cell_analysis_video.mp4",
    fps: int = 10
) -> str:
    """
    Create a video from numpy arrays (frame images).
    
    Parameters
    ----------
    video_frames : List[np.ndarray]
        List of frame images as numpy arrays
    output_dir : str
        Directory to save the video file
    video_filename : str
        Name of the output video file (default: "cell_analysis_video.mp4")
    fps : int
        Frames per second for the output video (default: 10)
        
    Returns
    -------
    str
        Path to the created video file
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for video creation. "
            "Install it with: pip install opencv-python"
        )

    if not video_frames:
        raise ValueError("No video frames provided")

    # Get dimensions from first frame
    first_frame = video_frames[0]
    height, width = first_frame.shape[:2]

    # Create video writer
    video_path = os.path.join(output_dir, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print(f"Creating video from {len(video_frames)} frames...")

    # Add each frame to the video
    for i, frame_array in enumerate(video_frames):
        # Convert RGB to BGR for OpenCV
        if frame_array.ndim == 3 and frame_array.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_array

        # Ensure correct data type
        if frame_bgr.dtype != np.uint8:
            frame_bgr = (frame_bgr * 255).astype(np.uint8)

        video_writer.write(frame_bgr)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(video_frames)} frames")

    # Release the video writer
    video_writer.release()

    print(f"Video created successfully: {video_path}")
    return video_path
