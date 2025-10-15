#!/usr/bin/env python3
"""
Streamlined test using the integrated cell tracking pipeline.

Data format: (timeframes, channels, height, width)
Channels: [0: pattern, 1: nuclei, 2: fluorescence_cytoplasm, 3: phase_contrast]
We'll use channels 1 (nuclei) and 2 (cytoplasm) for Cellpose segmentation.
"""

from src.cell_tracker.pipeline import analyze_timelapse_data


def main():
    """Main test function using the integrated pipeline"""
    print("=== Cell Tracker Pipeline Analysis ===\n")
    
    # Run the complete analysis using the integrated pipeline
    results = analyze_timelapse_data(
        data_path='data/example.npy',
        output_dir='data/segmentations',
        start_frame=21,
        channels=None  # Will use default channels from pipeline
    )
    
    print("\n=== Analysis Summary ===")
    print(f"Processed {results['total_frames']} frames")
    print(f"Frame range: {results['frame_range'][0]} to {results['frame_range'][1]}")
    print(f"T1 edge weight range: {results['t1_weight_range'][0]:.1f} to {results['t1_weight_range'][1]:.1f}")
    print(f"T1 events detected: {results['t1_events_detected']}")
    
    if results['t1_events']:
        print("\nT1 Events:")
        for event in results['t1_events']:
            print(f"  Frame {event['frame_start']}-{event['frame_end']}: "
                  f"{event['event_type']} (Δ={event['weight_change']:.1f})")

if __name__ == "__main__":
    main()