"""
HDF5 I/O utilities for bounding box data.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def save_bounding_boxes_hdf5(
    all_fovs_data: Dict[int, Dict[str, Any]], 
    output_path: str | Path,
    compression: str = "gzip"
) -> None:
    """
    Save bounding box data for all FOVs to a single HDF5 file.
    
    Parameters
    ----------
    all_fovs_data : dict
        Dictionary mapping FOV indices to their bounding box data
        (output from Patterner.extract_bounding_boxes() for each FOV)
    output_path : str or Path
        Output HDF5 file path
    compression : str
        Compression algorithm: 'gzip', 'lzf', 'szip', or None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare containers for all data
    fov_indices = []
    pattern_ids = []
    bbox_x = []
    bbox_y = []
    bbox_width = []
    bbox_height = []
    center_x = []
    center_y = []
    areas = []
    patterns_paths = []
    cells_paths = []
    image_heights = []
    image_widths = []
    
    total_patterns = 0
    
    # Collect data from all FOVs
    for fov_idx, fov_data in all_fovs_data.items():
        patterns = fov_data["patterns"]
        total_patterns += len(patterns)
        
        for pattern in patterns:
            fov_indices.append(fov_idx)
            pattern_ids.append(pattern["pattern_id"])
            bbox_x.append(pattern["bbox"]["x"])
            bbox_y.append(pattern["bbox"]["y"])
            bbox_width.append(pattern["bbox"]["width"])
            bbox_height.append(pattern["bbox"]["height"])
            center_x.append(pattern["center"]["x"])
            center_y.append(pattern["center"]["y"])
            areas.append(pattern["area"])
            patterns_paths.append(fov_data["patterns_path"])
            cells_paths.append(fov_data["cells_path"])
            if fov_data["image_shape"]:
                image_heights.append(fov_data["image_shape"][0])
                image_widths.append(fov_data["image_shape"][1])
            else:
                image_heights.append(-1)
                image_widths.append(-1)
    
    if total_patterns == 0:
        logger.warning("No pattern data to save")
        return
    
    # Save to HDF5
    with h5py.File(output_path, 'w') as h5f:
        # Create main dataset
        bounding_boxes = h5f.create_group('bounding_boxes')
        
        # Store all data as structured arrays
        bounding_boxes.create_dataset('fov_index', data=np.array(fov_indices, dtype=np.int32),  
                                   compression=compression)
        bounding_boxes.create_dataset('pattern_id', data=np.array(pattern_ids, dtype=np.int32),
                                   compression=compression)
        bounding_boxes.create_dataset('bbox_x', data=np.array(bbox_x, dtype=np.int32),
                                   compression=compression)
        bounding_boxes.create_dataset('bbox_y', data=np.array(bbox_y, dtype=np.int32),
                                   compression=compression)
        bounding_boxes.create_dataset('bbox_width', data=np.array(bbox_width, dtype=np.int32),
                                   compression=compression)
        bounding_boxes.create_dataset('bbox_height', data=np.array(bbox_height, dtype=np.int32),
                                   compression=compression)
        bounding_boxes.create_dataset('center_x', data=np.array(center_x, dtype=np.int32),
                                   compression=compression)
        bounding_boxes.create_dataset('center_y', data=np.array(center_y, dtype=np.int32),
                                   compression=compression)
        bounding_boxes.create_dataset('area', data=np.array(areas, dtype=np.int32),
                                   compression=compression)
        
        # Store string arrays as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        bounding_boxes.create_dataset('patterns_path', data=np.array(patterns_paths, dtype=dt),
                                   compression=compression)
        bounding_boxes.create_dataset('cells_path', data=np.array(cells_paths, dtype=dt),
                                   compression=compression)
        bounding_boxes.create_dataset('image_height', data=np.array(image_heights, dtype=np.int32),
                                   compression=compression)
        bounding_boxes.create_dataset('image_width', data=np.array(image_widths, dtype=np.int32),
                                   compression=compression)
        
        # Store metadata
        metadata_group = h5f.create_group('metadata')
        metadata_group.create_dataset('total_fovs', data=len(all_fovs_data), dtype=np.int32)
        metadata_group.create_dataset('total_patterns', data=total_patterns, dtype=np.int32)
        metadata_group.create_dataset('processed_fovs', data=list(all_fovs_data.keys()), dtype=np.int32)
        metadata_group.attrs['creation_time'] = datetime.now().isoformat()
        metadata_group.attrs['cell_pattern_version'] = '1.0'
        metadata_group.attrs['description'] = 'Bounding boxes from cell-pattern pattern detection'
        
        # Store per-FOV metadata
        for fov_idx, fov_data in all_fovs_data.items():
            fov_group = metadata_group.create_group(f'fov_{fov_idx:03d}')
            fov_group.attrs['n_patterns'] = len(fov_data['patterns'])
            fov_group.attrs['patterns_path'] = fov_data['patterns_path']
            fov_group.attrs['cells_path'] = fov_data['cells_path']
            if fov_data['image_shape']:
                fov_group.attrs['image_shape'] = fov_data['image_shape']
            if fov_data.get('extraction_timestamp'):
                fov_group.attrs['extraction_timestamp'] = fov_data['extraction_timestamp']
    
    # Log results
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved {total_patterns} pattern records from {len(all_fovs_data)} FOVs to {output_path}")
    logger.info(f"HDF5 file size: {file_size_mb:.1f} MB")
    logger.info(f"Compression: {compression} - typical size reduction: 60-80% vs JSON")


def load_bounding_boxes_hdf5(file_path: str | Path) -> Dict[str, Any]:
    """
    Load bounding box data from an HDF5 file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to HDF5 file
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'metadata': file metadata
        - 'fovs': list of available FOV indices  
        - 'total_patterns': total number of patterns
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Bounding box HDF5 file not found: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as h5f:
            metadata = {}
            
            # Read main metadata
            if 'metadata' in h5f:
                metadata_group = h5f['metadata']
                metadata['total_fovs'] = int(metadata_group['total_fovs'][()])
                metadata['total_patterns'] = int(metadata_group['total_patterns'][()])
                metadata['processed_fovs'] = list(metadata_group['processed_fovs'][()])
                
                # Copy attributes
                for key, value in metadata_group.attrs.items():
                    metadata[key] = value
                
                # Store per-FOV metadata reference
                metadata['fov_groups'] = {}
                for fov_idx in metadata['processed_fovs']:
                    fov_key = f'fov_{fov_idx:03d}'
                    if fov_key in metadata_group:
                        fov_metadata = {}
                        for key, value in metadata_group[fov_key].attrs.items():
                            fov_metadata[key] = value
                        metadata['fov_groups'][fov_idx] = fov_metadata
            
            logger.info(f"Loaded HDF5 file: {metadata['total_patterns']} patterns across {metadata['total_fovs']} FOVs")
            return metadata
            
    except Exception as e:
        raise ValueError(f"Error reading HDF5 file {file_path}: {e}")


def get_fov_bounding_boxes(file_path: str | Path, fov_idx: int) -> Dict[str, Any]:
    """
    Extract bounding boxes for a specific FOV from HDF5 file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to HDF5 file
    fov_idx : int
        Field of view index
        
    Returns
    -------
    dict
        Dictionary containing bounding box data for the specified FOV
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Bounding box HDF5 file not found: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as h5f:
            if 'bounding_boxes' not in h5f:
                raise ValueError("No bounding box data found in HDF5 file")
            
            bbox_group = h5f['bounding_boxes']
            
            # Load all arrays and filter by FOV
            fov_indices = bbox_group['fov_index'][:]
            pattern_mask = fov_indices == fov_idx
            
            if not np.any(pattern_mask):
                logger.warning(f"No bounding box data found for FOV {fov_idx}")
                return {"patterns": []}
            
            # Extract fov data
            patterns = []
            for i in np.where(pattern_mask)[0]:
                pattern = {
                    "pattern_id": int(bbox_group['pattern_id'][i]),
                    "bbox": {
                        "x": int(bbox_group['bbox_x'][i]),
                        "y": int(bbox_group['bbox_y'][i]),
                        "width": int(bbox_group['bbox_width'][i]),
                        "height": int(bbox_group['bbox_height'][i])
                    },
                    "center": {
                        "x": int(bbox_group['center_x'][i]),
                        "y": int(bbox_group['center_y'][i])
                    },
                    "area": int(bbox_group['area'][i])
                }
                patterns.append(pattern)
            
            # Sort by pattern_id
            patterns.sort(key=lambda p: p['pattern_id'])
            
            # Extract FOV metadata
            fov_data = {
                "patterns": patterns,
                "n_patterns": len(patterns),
                "fov_index": fov_idx
            }
            
            # Add additional metadata if available
            if 'metadata' in h5f and f'fov_{fov_idx:03d}' in h5f['metadata']:
                fov_metadata_group = h5f['metadata'][f'fov_{fov_idx:03d}']
                fov_data['patterns_path'] = fov_metadata_group.attrs.get('patterns_path', '')
                fov_data['cells_path'] = fov_metadata_group.attrs.get('cells_path', '')
                
                image_shape = fov_metadata_group.attrs.get('image_shape')
                if image_shape:
                    fov_data['image_shape'] = list(image_shape)
            
            logger.info(f"Found {len(patterns)} patterns for FOV {fov_idx}")
            return fov_data
            
    except Exception as e:
        raise ValueError(f"Error reading FOV {fov_idx} from HDF5 file {file_path}: {e}")


def get_available_fovs(file_path: str | Path) -> List[int]:
    """
    Get list of available FOV indices from HDF5 file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to HDF5 file
        
    Returns
    -------
    list[int]
        List of available FOV indices
    """
    metadata = load_bounding_boxes_hdf5(file_path)
    return metadata.get('processed_fovs', [])


# =============================================================================
# Analysis H5 Functions (cell-filter analysis stage)
# =============================================================================


def append_analysis_h5(
    h5_path: str | Path,
    analysis_data: Dict[str, Any],
    compression: str = "gzip"
) -> None:
    """
    Append analysis data (cell counts per FOV/pattern/frame) to an existing H5 file.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to existing HDF5 file (must contain bounding_boxes group)
    analysis_data : dict
        Dictionary containing:
        - 'records': list of {fov_index, pattern_id, frame_index, cell_count}
        - 'metadata': {cells_path, nuclei_channel, min_size, processed_fovs}
    compression : str
        Compression algorithm
    """
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    records = analysis_data['records']
    metadata = analysis_data['metadata']
    
    if not records:
        logger.warning("No analysis records to save")
        return
    
    # Extract arrays from records
    fov_indices = np.array([r['fov_index'] for r in records], dtype=np.int32)
    pattern_ids = np.array([r['pattern_id'] for r in records], dtype=np.int32)
    frame_indices = np.array([r['frame_index'] for r in records], dtype=np.int32)
    cell_counts = np.array([r['cell_count'] for r in records], dtype=np.int32)
    
    with h5py.File(h5_path, 'a') as h5f:
        # Remove existing analysis group if present
        if 'analysis' in h5f:
            del h5f['analysis']
        
        # Create analysis group
        analysis_group = h5f.create_group('analysis')
        
        # Store data arrays
        analysis_group.create_dataset('fov_index', data=fov_indices, compression=compression)
        analysis_group.create_dataset('pattern_id', data=pattern_ids, compression=compression)
        analysis_group.create_dataset('frame_index', data=frame_indices, compression=compression)
        analysis_group.create_dataset('cell_count', data=cell_counts, compression=compression)
        
        # Store metadata
        analysis_metadata = analysis_group.create_group('metadata')
        analysis_metadata.attrs['cells_path'] = metadata.get('cells_path', '')
        analysis_metadata.attrs['nuclei_channel'] = metadata.get('nuclei_channel', -1)
        analysis_metadata.attrs['min_size'] = metadata.get('min_size', 15)
        analysis_metadata.attrs['creation_time'] = datetime.now().isoformat()
        analysis_metadata.attrs['total_records'] = len(records)
        
        # Store processed FOVs
        processed_fovs = metadata.get('processed_fovs', [])
        analysis_metadata.create_dataset('processed_fovs', data=np.array(processed_fovs, dtype=np.int32))
    
    logger.info(f"Appended {len(records)} analysis records to {h5_path}")


def load_analysis_h5(file_path: str | Path) -> Dict[str, Any]:
    """
    Load analysis metadata from H5 file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to HDF5 file
        
    Returns
    -------
    dict
        Dictionary containing analysis metadata
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"H5 file not found: {file_path}")
    
    with h5py.File(file_path, 'r') as h5f:
        if 'analysis' not in h5f:
            raise ValueError("No analysis data found in H5 file")
        
        analysis_group = h5f['analysis']
        metadata_group = analysis_group['metadata']
        
        metadata = {
            'cells_path': metadata_group.attrs.get('cells_path', ''),
            'nuclei_channel': int(metadata_group.attrs.get('nuclei_channel', -1)),
            'min_size': int(metadata_group.attrs.get('min_size', 15)),
            'creation_time': metadata_group.attrs.get('creation_time', ''),
            'total_records': int(metadata_group.attrs.get('total_records', 0)),
            'processed_fovs': list(metadata_group['processed_fovs'][:]) if 'processed_fovs' in metadata_group else []
        }
        
        return metadata


def get_analysis_for_fov(file_path: str | Path, fov_idx: int) -> Dict[int, Dict[int, int]]:
    """
    Get analysis data (cell counts) for a specific FOV.
    
    Parameters
    ----------
    file_path : str or Path
        Path to HDF5 file
    fov_idx : int
        Field of view index
        
    Returns
    -------
    dict
        Dictionary mapping pattern_id -> {frame_index -> cell_count}
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"H5 file not found: {file_path}")
    
    with h5py.File(file_path, 'r') as h5f:
        if 'analysis' not in h5f:
            raise ValueError("No analysis data found in H5 file")
        
        analysis_group = h5f['analysis']
        
        # Load arrays
        fov_indices = analysis_group['fov_index'][:]
        pattern_ids = analysis_group['pattern_id'][:]
        frame_indices = analysis_group['frame_index'][:]
        cell_counts = analysis_group['cell_count'][:]
        
        # Filter by FOV
        mask = fov_indices == fov_idx
        
        if not np.any(mask):
            logger.warning(f"No analysis data found for FOV {fov_idx}")
            return {}
        
        # Build result dictionary
        result: Dict[int, Dict[int, int]] = {}
        for i in np.where(mask)[0]:
            pattern_id = int(pattern_ids[i])
            frame_idx = int(frame_indices[i])
            count = int(cell_counts[i])
            
            if pattern_id not in result:
                result[pattern_id] = {}
            result[pattern_id][frame_idx] = count
        
        return result


def get_all_analysis(file_path: str | Path) -> List[Dict[str, int]]:
    """
    Load all analysis records from H5 file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to HDF5 file
        
    Returns
    -------
    list
        List of {fov_index, pattern_id, frame_index, cell_count} records
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"H5 file not found: {file_path}")
    
    with h5py.File(file_path, 'r') as h5f:
        if 'analysis' not in h5f:
            raise ValueError("No analysis data found in H5 file")
        
        analysis_group = h5f['analysis']
        
        fov_indices = analysis_group['fov_index'][:]
        pattern_ids = analysis_group['pattern_id'][:]
        frame_indices = analysis_group['frame_index'][:]
        cell_counts = analysis_group['cell_count'][:]
        
        records = []
        for i in range(len(fov_indices)):
            records.append({
                'fov_index': int(fov_indices[i]),
                'pattern_id': int(pattern_ids[i]),
                'frame_index': int(frame_indices[i]),
                'cell_count': int(cell_counts[i])
            })
        
        return records


# =============================================================================
# Extracted H5 Functions (cell-filter extract stage)
# =============================================================================


def append_extracted_sequence(
    h5_path: str | Path,
    fov_idx: int,
    pattern_idx: int,
    seq_idx: int,
    data: np.ndarray,
    metadata: Dict[str, Any],
    compression: str = "gzip"
) -> None:
    """
    Append an extracted sequence to the H5 file.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file
    fov_idx : int
        Field of view index
    pattern_idx : int
        Pattern index
    seq_idx : int
        Sequence index
    data : np.ndarray
        Image data array (n_frames, n_channels+2, h, w)
    metadata : dict
        Sequence metadata (start_frame, end_frame, channels, bbox)
    compression : str
        Compression algorithm
    """
    h5_path = Path(h5_path)
    
    with h5py.File(h5_path, 'a') as h5f:
        # Create extracted group if not exists
        if 'extracted' not in h5f:
            h5f.create_group('extracted')
        
        extracted_group = h5f['extracted']
        
        # Create FOV group if not exists
        fov_key = f'fov_{fov_idx:03d}'
        if fov_key not in extracted_group:
            extracted_group.create_group(fov_key)
        
        fov_group = extracted_group[fov_key]
        
        # Create pattern group if not exists
        pattern_key = f'pattern_{pattern_idx:03d}'
        if pattern_key not in fov_group:
            fov_group.create_group(pattern_key)
        
        pattern_group = fov_group[pattern_key]
        
        # Create sequence group
        seq_key = f'seq_{seq_idx:03d}'
        if seq_key in pattern_group:
            del pattern_group[seq_key]
        
        seq_group = pattern_group.create_group(seq_key)
        
        # Store data
        seq_group.create_dataset('data', data=data, compression=compression)
        
        # Store metadata as attributes
        seq_group.attrs['start_frame'] = metadata.get('start_frame', 0)
        seq_group.attrs['end_frame'] = metadata.get('end_frame', 0)
        
        # Store channel names
        channels = metadata.get('channels', [])
        if channels:
            dt = h5py.special_dtype(vlen=str)
            seq_group.create_dataset('channels', data=np.array(channels, dtype=dt))
        
        # Store bounding box
        bbox = metadata.get('bbox')
        if bbox:
            seq_group.attrs['bbox_x'] = bbox[0]
            seq_group.attrs['bbox_y'] = bbox[1]
            seq_group.attrs['bbox_width'] = bbox[2]
            seq_group.attrs['bbox_height'] = bbox[3]
    
    logger.debug(f"Saved sequence fov_{fov_idx:03d}/pattern_{pattern_idx:03d}/seq_{seq_idx:03d}")


def finalize_extracted_metadata(
    h5_path: str | Path,
    metadata: Dict[str, Any]
) -> None:
    """
    Add final metadata after extraction is complete.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file
    metadata : dict
        Extraction metadata (n_cells, tolerance_gap, min_frames, etc.)
    """
    h5_path = Path(h5_path)
    
    with h5py.File(h5_path, 'a') as h5f:
        if 'extracted' not in h5f:
            h5f.create_group('extracted')
        
        extracted_group = h5f['extracted']
        
        # Store extraction parameters
        extracted_group.attrs['n_cells'] = metadata.get('n_cells', 0)
        extracted_group.attrs['tolerance_gap'] = metadata.get('tolerance_gap', 0)
        extracted_group.attrs['min_frames'] = metadata.get('min_frames', 0)
        extracted_group.attrs['creation_time'] = datetime.now().isoformat()
        extracted_group.attrs['cells_path'] = metadata.get('cells_path', '')
    
    logger.info(f"Finalized extraction metadata in {h5_path}")


def load_extracted_sequence(
    file_path: str | Path,
    fov_idx: int,
    pattern_idx: int,
    seq_idx: int
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Load an extracted sequence from H5 file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to HDF5 file
    fov_idx : int
        Field of view index
    pattern_idx : int
        Pattern index
    seq_idx : int
        Sequence index
        
    Returns
    -------
    tuple
        (data array, metadata dict)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"H5 file not found: {file_path}")
    
    with h5py.File(file_path, 'r') as h5f:
        path = f'extracted/fov_{fov_idx:03d}/pattern_{pattern_idx:03d}/seq_{seq_idx:03d}'
        
        if path not in h5f:
            raise ValueError(f"Sequence not found: {path}")
        
        seq_group = h5f[path]
        
        # Load data
        data = seq_group['data'][:]
        
        # Load metadata
        metadata = {
            'start_frame': int(seq_group.attrs.get('start_frame', 0)),
            'end_frame': int(seq_group.attrs.get('end_frame', 0)),
        }
        
        # Load channels if present
        if 'channels' in seq_group:
            metadata['channels'] = list(seq_group['channels'][:].astype(str))
        
        # Load bbox if present
        if 'bbox_x' in seq_group.attrs:
            metadata['bbox'] = (
                int(seq_group.attrs['bbox_x']),
                int(seq_group.attrs['bbox_y']),
                int(seq_group.attrs['bbox_width']),
                int(seq_group.attrs['bbox_height'])
            )
        
        return data, metadata


def list_extracted_sequences(file_path: str | Path) -> List[Dict[str, int]]:
    """
    List all extracted sequences in H5 file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to HDF5 file
        
    Returns
    -------
    list
        List of {fov_idx, pattern_idx, seq_idx} dicts
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"H5 file not found: {file_path}")
    
    sequences = []
    
    with h5py.File(file_path, 'r') as h5f:
        if 'extracted' not in h5f:
            return sequences
        
        extracted_group = h5f['extracted']
        
        for fov_key in extracted_group.keys():
            if not fov_key.startswith('fov_'):
                continue
            fov_idx = int(fov_key.split('_')[1])
            
            fov_group = extracted_group[fov_key]
            for pattern_key in fov_group.keys():
                if not pattern_key.startswith('pattern_'):
                    continue
                pattern_idx = int(pattern_key.split('_')[1])
                
                pattern_group = fov_group[pattern_key]
                for seq_key in pattern_group.keys():
                    if not seq_key.startswith('seq_'):
                        continue
                    seq_idx = int(seq_key.split('_')[1])
                    
                    sequences.append({
                        'fov_idx': fov_idx,
                        'pattern_idx': pattern_idx,
                        'seq_idx': seq_idx
                    })
    
    return sequences
