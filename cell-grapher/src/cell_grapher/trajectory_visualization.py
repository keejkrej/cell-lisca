"""
Visualization functions for cell trajectories (nucleus vs cytoplasm).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import matplotlib.cm as cm


def plot_nucleus_cytoplasm_trajectories(
    trajectories: Dict[int, Dict],
    output_path: str,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot nucleus and cytoplasm trajectories together for comparison.

    Parameters
    ----------
    trajectories : Dict[int, Dict]
        Trajectory data from TrajectoryAnalyzer
    output_path : str
        Path to save the plot
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get colormap
    n_tracks = len(trajectories)
    colors = cm.tab20(np.linspace(0, 1, n_tracks))

    for idx, (global_id, traj_data) in enumerate(trajectories.items()):
        nucleus_positions = np.array(traj_data['nucleus_positions'])
        cytoplasm_positions = np.array(traj_data['cytoplasm_positions'])

        if len(nucleus_positions) < 2:
            continue

        # Plot nucleus trajectory (solid line)
        ax.plot(nucleus_positions[:, 1], nucleus_positions[:, 0],
                '-', color=colors[idx], alpha=0.8, linewidth=2.5,
                label=f'Cell {global_id} (Nucleus)')

        # Plot cytoplasm trajectory (dashed line)
        ax.plot(cytoplasm_positions[:, 1], cytoplasm_positions[:, 0],
                '--', color=colors[idx], alpha=0.6, linewidth=2,
                label=f'Cell {global_id} (Cytoplasm)')

        # Add arrows every 10 points to show direction for nucleus
        for i in range(0, len(nucleus_positions) - 1, 10):
            dx = nucleus_positions[i + 1, 1] - nucleus_positions[i, 1]
            dy = nucleus_positions[i + 1, 0] - nucleus_positions[i, 0]
            ax.arrow(nucleus_positions[i, 1], nucleus_positions[i, 0], dx, dy,
                     head_width=3, head_length=3, fc=colors[idx],
                     ec=colors[idx], alpha=0.7, length_includes_head=True)

        # Mark start points (circles)
        ax.plot(nucleus_positions[0, 1], nucleus_positions[0, 0],
                'o', color=colors[idx], markersize=10,
                markeredgecolor='black', markeredgewidth=1.5)

        # Mark end points (squares)
        ax.plot(nucleus_positions[-1, 1], nucleus_positions[-1, 0],
                's', color=colors[idx], markersize=10,
                markeredgecolor='black', markeredgewidth=1.5)

        # Draw connecting lines between nucleus and cytoplasm at key time points
        # Show every 10th frame to avoid clutter
        for i in range(0, len(nucleus_positions), 10):
            ax.plot([nucleus_positions[i, 1], cytoplasm_positions[i, 1]],
                    [nucleus_positions[i, 0], cytoplasm_positions[i, 0]],
                    ':', color=colors[idx], alpha=0.3, linewidth=1)

    ax.set_xlabel('X position (pixels)', fontsize=12)
    ax.set_ylabel('Y position (pixels)', fontsize=12)
    ax.set_title('Cell Trajectories: Nucleus vs Cytoplasm\n(Solid=Nucleus, Dashed=Cytoplasm, Circles=Start, Squares=End)',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()  # Image coordinates have origin at top-left
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Legend - limit to avoid clutter
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 20:
        ax.legend(handles[:20], labels[:20], bbox_to_anchor=(1.05, 1),
                  loc='upper left', fontsize=8, ncol=1)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trajectory plot saved to {output_path}")


def plot_polar_coordinates(
    trajectories: Dict[int, Dict],
    output_path: str,
    origin: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (16, 6)
):
    """
    Plot r and theta vs time for nucleus and cytoplasm (2 subplots: r and theta).

    Parameters
    ----------
    trajectories : Dict[int, Dict]
        Trajectory data from TrajectoryAnalyzer
    output_path : str
        Path to save the plot
    origin : tuple, optional
        Origin (y, x) for polar coordinates. If None, uses center of all trajectories
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get colormap
    n_tracks = len(trajectories)
    colors = cm.tab20(np.linspace(0, 1, n_tracks))

    # Calculate origin if not provided
    if origin is None:
        all_positions = []
        for traj_data in trajectories.values():
            all_positions.extend(traj_data['nucleus_positions'])
            all_positions.extend(traj_data['cytoplasm_positions'])
        all_positions = np.array(all_positions)
        origin = (np.mean(all_positions[:, 0]), np.mean(all_positions[:, 1]))

    origin_y, origin_x = origin

    for idx, (global_id, traj_data) in enumerate(trajectories.items()):
        nucleus_positions = np.array(traj_data['nucleus_positions'])
        cytoplasm_positions = np.array(traj_data['cytoplasm_positions'])
        frames = traj_data['frames']

        if len(nucleus_positions) < 2:
            continue

        # Calculate polar coordinates for nucleus
        dy_n = nucleus_positions[:, 0] - origin_y
        dx_n = nucleus_positions[:, 1] - origin_x
        r_nucleus = np.sqrt(dx_n**2 + dy_n**2)
        theta_nucleus = np.arctan2(dy_n, dx_n) * 180 / np.pi  # Convert to degrees

        # Calculate polar coordinates for cytoplasm
        dy_c = cytoplasm_positions[:, 0] - origin_y
        dx_c = cytoplasm_positions[:, 1] - origin_x
        r_cytoplasm = np.sqrt(dx_c**2 + dy_c**2)
        theta_cytoplasm = np.arctan2(dy_c, dx_c) * 180 / np.pi

        # Plot r vs time (nucleus: solid, cytoplasm: dashed)
        ax1.plot(frames, r_nucleus, '-', color=colors[idx], alpha=0.8,
                 linewidth=2, markersize=4, label=f'Cell {global_id} (Nucleus)')
        ax1.plot(frames, r_cytoplasm, '--', color=colors[idx], alpha=0.6,
                 linewidth=1.5, markersize=3, label=f'Cell {global_id} (Cytoplasm)')

        # Plot theta vs time (nucleus: solid, cytoplasm: dashed)
        ax2.plot(frames, theta_nucleus, '-', color=colors[idx], alpha=0.8,
                 linewidth=2, markersize=4, label=f'Cell {global_id} (Nucleus)')
        ax2.plot(frames, theta_cytoplasm, '--', color=colors[idx], alpha=0.6,
                 linewidth=1.5, markersize=3, label=f'Cell {global_id} (Cytoplasm)')

    # Format r plot
    ax1.set_xlabel('Frame', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distance from origin (pixels)', fontsize=12, fontweight='bold')
    ax1.set_title('Radial Distance (r) vs Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    handles1, labels1 = ax1.get_legend_handles_labels()
    if len(handles1) > 20:
        ax1.legend(handles1[:20], labels1[:20], fontsize=7, ncol=2)
    else:
        ax1.legend(fontsize=7, ncol=2)

    # Format theta plot
    ax2.set_xlabel('Frame', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax2.set_title('Angle (θ) vs Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-180, 180)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    handles2, labels2 = ax2.get_legend_handles_labels()
    if len(handles2) > 20:
        ax2.legend(handles2[:20], labels2[:20], fontsize=7, ncol=2)
    else:
        ax2.legend(fontsize=7, ncol=2)

    plt.suptitle('Polar Coordinates: Nucleus (solid) vs Cytoplasm (dashed)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Polar coordinate plot saved to {output_path}")


def plot_trajectory_visualizations(
    trajectories: Dict[int, Dict],
    output_dir: str,
    origin: Optional[Tuple[float, float]] = None
):
    """
    Generate trajectory visualization plots (2 plots only).

    Parameters
    ----------
    trajectories : Dict[int, Dict]
        Trajectory data from TrajectoryAnalyzer
    output_dir : str
        Directory to save plots
    origin : tuple, optional
        Origin for polar coordinates

    Returns
    -------
    dict
        Paths to generated plots
    """
    import os

    if not trajectories:
        print("Warning: No trajectory data to visualize")
        return {}

    # 1. Nucleus vs Cytoplasm trajectories
    traj_path = os.path.join(output_dir, 'cell_trajectories.png')
    plot_nucleus_cytoplasm_trajectories(trajectories, traj_path)

    # 2. Polar coordinates (r and theta)
    polar_path = os.path.join(output_dir, 'cell_trajectories_polar.png')
    plot_polar_coordinates(trajectories, polar_path, origin=origin)

    return {
        'cell_trajectories_plot': traj_path,
        'cell_trajectories_polar': polar_path
    }
