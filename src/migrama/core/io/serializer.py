"""
HDF5 serialization utilities for dataclasses.
"""

import h5py
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from ..models.data_models import FOVData, Pattern, BoundingBox, AnalysisRecord, ExtractedSequence


class HDF5Serializer:
    """Centralized serialization/deserialization for dataclasses to/from HDF5."""
    
    @staticmethod
    def save_bounding_box(bbox: BoundingBox, group: h5py.Group, prefix: str = ""):
        """Save a BoundingBox to HDF5 group."""
        group.create_dataset(f'{prefix}x', data=bbox.x)
        group.create_dataset(f'{prefix}y', data=bbox.y)
        group.create_dataset(f'{prefix}width', data=bbox.width)
        group.create_dataset(f'{prefix}height', data=bbox.height)
    
    @staticmethod
    def load_bounding_box(group: h5py.Group, prefix: str = "") -> BoundingBox:
        """Load a BoundingBox from HDF5 group."""
        return BoundingBox(
            x=int(group[f'{prefix}x'][()]),
            y=int(group[f'{prefix}y'][()]),
            width=int(group[f'{prefix}width'][()]),
            height=int(group[f'{prefix}height'][()])
        )
    
    @staticmethod
    def save_pattern(pattern: Pattern, group: h5py.Group, index: int = None):
        """Save a Pattern to HDF5 group."""
        prefix = f'pattern_{index}_' if index is not None else ''
        
        # Save pattern metadata
        group.create_dataset(f'{prefix}pattern_id', data=pattern.pattern_id)
        group.create_dataset(f'{prefix}area', data=pattern.area)
        
        # Save center coordinates
        center_group = group.create_group(f'{prefix}center')
        center_group.create_dataset('x', data=pattern.center[0])
        center_group.create_dataset('y', data=pattern.center[1])
        
        # Save bounding box
        bbox_group = group.create_group(f'{prefix}bbox')
        HDF5Serializer.save_bounding_box(pattern.bbox, bbox_group)
    
    @staticmethod
    def load_pattern(group: h5py.Group, index: int = None) -> Pattern:
        """Load a Pattern from HDF5 group."""
        prefix = f'pattern_{index}_' if index is not None else ''
        
        # Load pattern metadata
        pattern_id = int(group[f'{prefix}pattern_id'][()])
        area = int(group[f'{prefix}area'][()])
        
        # Load center coordinates
        center_group = group[f'{prefix}center']
        center = (int(center_group['x'][()]), int(center_group['y'][()]))
        
        # Load bounding box
        bbox_group = group[f'{prefix}bbox']
        bbox = HDF5Serializer.load_bounding_box(bbox_group)
        
        return Pattern(
            pattern_id=pattern_id,
            bbox=bbox,
            center=center,
            area=area
        )
    
    @staticmethod
    def save_fov_data(fov_data: FOVData, group: h5py.Group):
        """Save FOVData to HDF5 group."""
        # Save metadata
        group.attrs['fov_index'] = fov_data.fov_index
        group.attrs['patterns_path'] = fov_data.patterns_path
        group.attrs['cells_path'] = fov_data.cells_path
        group.attrs['extraction_timestamp'] = fov_data.extraction_timestamp or ''
        
        if fov_data.image_shape:
            group.attrs['image_shape'] = fov_data.image_shape
        
        # Save patterns
        patterns_group = group.create_group('patterns')
        for i, pattern in enumerate(fov_data.patterns):
            HDF5Serializer.save_pattern(pattern, patterns_group, i)
    
    @staticmethod
    def load_fov_data(group: h5py.Group) -> FOVData:
        """Load FOVData from HDF5 group."""
        # Load metadata
        fov_index = int(group.attrs['fov_index'])
        patterns_path = group.attrs.get('patterns_path', '')
        cells_path = group.attrs.get('cells_path', '')
        extraction_timestamp = group.attrs.get('extraction_timestamp')
        
        image_shape = None
        if 'image_shape' in group.attrs:
            image_shape = list(group.attrs['image_shape'])
        
        # Load patterns
        patterns = []
        if 'patterns' in group:
            patterns_group = group['patterns']
            for key in sorted(patterns_group.keys()):
                if key.startswith('pattern_'):
                    index = int(key.split('_')[1])
                    pattern = HDF5Serializer.load_pattern(patterns_group, index)
                    patterns.append(pattern)
        
        return FOVData(
            fov_index=fov_index,
            patterns=patterns,
            patterns_path=patterns_path,
            cells_path=cells_path,
            image_shape=image_shape,
            extraction_timestamp=extraction_timestamp
        )
    
    @staticmethod
    def save_analysis_record(record: AnalysisRecord, group: h5py.Group, index: int = None):
        """Save an AnalysisRecord to HDF5 group."""
        prefix = f'record_{index}_' if index is not None else ''
        
        group.create_dataset(f'{prefix}fov_index', data=record.fov_index)
        group.create_dataset(f'{prefix}pattern_id', data=record.pattern_id)
        group.create_dataset(f'{prefix}frame_index', data=record.frame_index)
        group.create_dataset(f'{prefix}cell_count', data=record.cell_count)
    
    @staticmethod
    def load_analysis_record(group: h5py.Group, index: int = None) -> AnalysisRecord:
        """Load an AnalysisRecord from HDF5 group."""
        prefix = f'record_{index}_' if index is not None else ''
        
        return AnalysisRecord(
            fov_index=int(group[f'{prefix}fov_index'][()]),
            pattern_id=int(group[f'{prefix}pattern_id'][()]),
            frame_index=int(group[f'{prefix}frame_index'][()]),
            cell_count=int(group[f'{prefix}cell_count'][()])
        )
    
    @staticmethod
    def save_extracted_sequence(seq: ExtractedSequence, group: h5py.Group):
        """Save an ExtractedSequence to HDF5 group."""
        # Save metadata
        group.attrs['fov_idx'] = seq.fov_idx
        group.attrs['pattern_idx'] = seq.pattern_idx
        group.attrs['seq_idx'] = seq.seq_idx
        group.attrs['start_frame'] = seq.start_frame
        group.attrs['end_frame'] = seq.end_frame
        group.attrs['n_frames'] = seq.n_frames
        group.attrs['data_path'] = seq.data_path or ''
        
        # Save channels as attribute (list of strings)
        if seq.channels:
            group.attrs['channels'] = seq.channels
        
        # Save bbox
        bbox_group = group.create_group('bbox')
        bbox_group.create_dataset('x', data=seq.bbox[0])
        bbox_group.create_dataset('y', data=seq.bbox[1])
        bbox_group.create_dataset('width', data=seq.bbox[2])
        bbox_group.create_dataset('height', data=seq.bbox[3])
        
        # Save actual data if present
        if seq.data is not None:
            group.create_dataset('data', data=seq.data)
    
    @staticmethod
    def load_extracted_sequence(group: h5py.Group) -> ExtractedSequence:
        """Load an ExtractedSequence from HDF5 group."""
        # Load metadata
        fov_idx = int(group.attrs['fov_idx'])
        pattern_idx = int(group.attrs['pattern_idx'])
        seq_idx = int(group.attrs['seq_idx'])
        start_frame = int(group.attrs['start_frame'])
        end_frame = int(group.attrs['end_frame'])
        n_frames = int(group.attrs['n_frames'])
        data_path = group.attrs.get('data_path') or None
        
        # Load channels
        channels = []
        if 'channels' in group.attrs:
            channels = list(group.attrs['channels'])
        
        # Load bbox
        bbox_group = group['bbox']
        bbox = (
            int(bbox_group['x'][()]),
            int(bbox_group['y'][()]),
            int(bbox_group['width'][()]),
            int(bbox_group['height'][()])
        )
        
        # Load actual data if present
        data = None
        if 'data' in group:
            data = group['data'][()]
        
        return ExtractedSequence(
            fov_idx=fov_idx,
            pattern_idx=pattern_idx,
            seq_idx=seq_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            n_frames=n_frames,
            channels=channels,
            bbox=bbox,
            data_path=data_path,
            data=data
        )
