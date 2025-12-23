
###readme###
'''
This is starting predicting not from teh center of the nifti file but it takes the larger area from the masked nifti file and fomr there it start predicting with a paramter name
predict_all when set true will predict all the slcie including the accurast/anchor slice and when set False it will predict only the unknown slices. in this we rmeove the non brian clf and kept only 1 classifier and 1 regressor
Here we redined the refineing logic rather than earlier dice we are now using the inclusion score
Sentivity a larger value will do nothing 10 or more and a smaller value will do more changes
PARALLELIZATION OPTIMIZATIONS:
- Model Training: Brain and Boundary specialists now train in parallel using ThreadPoolExecutor
- Feature Extraction: Different feature types (GLCM, LBP, Gabor, etc.) computed in parallel batches
- Slice Processing: Individual slice processing optimized with parallel image preprocessing and post-processing
- Memory Management: Improved GPU memory usage with better batching and cleanup
- Forward/Backward Prediction: Optimized prediction pipeline with parallel sub-operations
- Performance Monitoring: Added detailed timing and performance metrics
Set use_parallel=True to enable all parallelization features for faster processing.
'''
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys
import os
import os
# Add this line at the VERY TOP of your script
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import gc
import itertools
from pathlib import Path
import concurrent.futures
# --- CPU Libraries ---
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, disk
# from scipy import ndimage as ndi # Use alias to avoid name conflicts
from sklearn.cluster import KMeans
from skimage.morphology import ball, remove_small_holes, binary_closing, disk
from scipy.ndimage import (
    label, # For labeling connected components
    gaussian_filter,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_closing,
    binary_opening,
    median_filter
)

import shutil
import tempfile
import ants  # Required for registration
import dicom2nifti # Required for DICOM handling
import pydicom # Required for DICOM export
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
# --- GPU Libraries ---
import torch
import torch.nn.functional as F
import cupy as cp
import xgboost as xgb
from cuml.ensemble import RandomForestClassifier as cuRF

# Optimize CuPy memory pool for better GPU performance
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=2**30)  # 1GB memory pool limit
# --- GPU-Accelerated Image Processing (with fallbacks) ---
try:
    from cucim.skimage.morphology import remove_small_objects as cucim_remove_small_objects
    from cucim.skimage.measure import label as cucim_label, regionprops as cucim_regionprops
    from cupyx.scipy.ndimage import gaussian_filter as cupy_gaussian_filter
    from cupyx.scipy.ndimage import binary_dilation as cupy_binary_dilation
    from cupyx.scipy.ndimage import binary_erosion as cupy_binary_erosion
    from cupyx.scipy.ndimage import binary_fill_holes as cupy_binary_fill_holes
    from cupyx.scipy.ndimage import binary_closing as cupy_binary_closing
    from cupyx.scipy.ndimage import binary_opening as cupy_binary_opening
    GPU_LIBRARIES_AVAILABLE = True
    print("✅ Successfully imported CuPy and cuCIM for GPU acceleration.")
except ImportError as e:
    print(f"⚠️ Warning: A required GPU library is missing ({e}). Refinement will fall back to CPU.")
    GPU_LIBRARIES_AVAILABLE = False
# import multiprocessing
# Initialize GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    raise RuntimeError("This script requires a CUDA-enabled GPU.")
# Precompute reusable components
GABOR_PARAMS = {
    'sigma_values': [0.5, 1.0, 2.0],
    'theta_values': [0, np.pi/4, np.pi/2],
    'lambd_values': [2, 4, 8],
    'gamma_values': [0.5, 1.0, 2.0],
    'psi_values': [0, np.pi/2]
}
# GABOR_PARAMS = {
# 'sigma_values': [1.0],
# 'theta_values': [0, np.pi/4, np.pi/2],
# 'lambd_values': [2],
# 'gamma_values': [1.0],
# 'psi_values': [0, np.pi/2]
# }
HEAD_MASK_CACHE = {}

def calculate_centroid(mask_slice_binary: np.ndarray) -> Optional[np.ndarray]:
    moments = cv2.moments(mask_slice_binary)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return np.array([cx, cy])
    return None

def calculate_shape_properties(mask_slice_binary: np.ndarray, area: Union[int, float]) -> Dict[str, float]:
    if area == 0: return {'perimeter': 0.0, 'solidity': 0.0, 'circularity': 0.0}
    contours, _ = cv2.findContours(mask_slice_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return {'perimeter': 0.0, 'solidity': 0.0, 'circularity': 0.0}
    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)
    if area == 0: return {'perimeter': perimeter, 'solidity': 0.0, 'circularity': 0.0}
    convex_hull = cv2.convexHull(largest_contour)
    convex_hull_area = cv2.contourArea(convex_hull)
    solidity = area / convex_hull_area if convex_hull_area > 0 else 0.0
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    return {'perimeter': perimeter, 'solidity': solidity, 'circularity': circularity}

def calculate_intensity_properties(original_slice_data: np.ndarray, mask_slice_binary: np.ndarray) -> Dict[str, float]:
    masked_pixels = original_slice_data[mask_slice_binary > 0]
    if masked_pixels.size > 0:
        return {'mean_intensity': np.mean(masked_pixels), 'std_intensity': np.std(masked_pixels)}
    return {'mean_intensity': np.nan, 'std_intensity': np.nan}

def calculate_all_properties_extended(original_data: np.ndarray, mask_data: np.ndarray, property_flags: Dict[str, bool]) -> Dict[int, Dict[str, Any]]:
    all_properties = {}
    num_slices = mask_data.shape[-1]
    for i in tqdm(range(num_slices), desc="Calculating Slice Properties"):
        mask_slice_binary = (mask_data[:, :, i] > 0).astype(np.uint8)
        original_slice = original_data[:, :, i]
        area = np.sum(mask_slice_binary)
        current_props = {'area': float(area), 'centroid': calculate_centroid(mask_slice_binary) if area > 0 else None}
        if property_flags.get('shape') and area > 0:
            current_props.update(calculate_shape_properties(mask_slice_binary, area))
        if property_flags.get('intensity'):
            current_props.update(calculate_intensity_properties(original_slice, mask_slice_binary))
        all_properties[i] = current_props
    return all_properties

def get_property_difference(val1: Any, val2: Any) -> float:
    if val1 is None or val2 is None or np.any(np.isnan(val1)) or np.any(np.isnan(val2)): return np.inf
    if isinstance(val1, np.ndarray): return euclidean(val1, val2) if val1.shape == val2.shape else np.inf
    try: return float(np.abs(val1 - val2))
    except TypeError: return np.inf

def precompute_inter_slice_differences(all_props: Dict[int, Dict[str, Any]], valid_slices: List[int], props_to_check: List[str]) -> Dict[str, List[float]]:
    diffs = {prop: [] for prop in props_to_check}
    if len(valid_slices) < 2: return diffs
    for i in tqdm(range(len(valid_slices) - 1), desc="Precomputing Slice Differences"):
        s1, s2 = valid_slices[i], valid_slices[i+1]
        for prop in props_to_check:
            diff_val = get_property_difference(all_props[s1].get(prop), all_props[s2].get(prop))
            if np.isfinite(diff_val): diffs[prop].append(diff_val)
    return diffs

def find_reliable_ranges_from_diffs(valid_slices: List[int], all_props: Dict[int, Dict[str, Any]], diffs: Dict[str, List[float]], percentiles: Dict[str, float], props_to_check: List[str], min_len: int) -> List[List[int]]:
    if len(valid_slices) < min_len: return []
    thresholds = {p: np.percentile(diffs.get(p, [np.inf]), percentiles.get(p, 100)) for p in props_to_check}
    print(f"Thresholds calculated: { {k: f'{v:.2f}' for k, v in thresholds.items()} }")
    
    ranges = []
    for i in tqdm(range(len(valid_slices)), desc="Finding Reliable Ranges"):
        end_idx = i
        for j in range(i, len(valid_slices) - 1):
            s1, s2 = valid_slices[j], valid_slices[j+1]
            is_stable = all(get_property_difference(all_props[s1].get(p), all_props[s2].get(p)) <= thresholds.get(p, np.inf) for p in props_to_check)
            if not is_stable: break
            end_idx = j + 1
        if end_idx - i + 1 >= min_len:
            ranges.append([valid_slices[i], valid_slices[end_idx]])
            
    if not ranges: return []
    ranges.sort(key=lambda r: r[1] - r[0], reverse=True)
    final_ranges = [ranges[0]]
    for start, end in ranges[1:]:
        if not any(start >= ex_start and end <= ex_end for ex_start, ex_end in final_ranges):
            final_ranges.append([start, end])
    return sorted(final_ranges)


# --- Main Preprocessing Function ---

def preprocessing(
    input_path: str,
    template_path: str = 'mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin.nii',
    template_mask_path: str = 'mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin_mask.nii',
    output_dir: Optional[str] = None,
    percentiles_dict: Optional[Dict[str, float]] = None,
    properties_to_check: List[str] = None,
    min_range_length: int = 8,
    remove_scout: bool = False
) -> Tuple[List[List[int]], str, str]:
    """
    Performs full preprocessing: handles DICOM/NIfTI input, generates brain mask
    via registration, finds reliable slice ranges, and returns the results.
    
    Parameters:
    -----------
    input_path : str
        Path to either a DICOM directory or a NIfTI file (.nii or .nii.gz)
    template_path : str
        Path to template brain image for registration
    template_mask_path : str
        Path to template brain mask for registration
    output_dir : str, optional
        Directory to save outputs. If None, uses the input directory
    percentiles_dict : Dict[str, float], optional
        Percentile thresholds for properties. Default: {'centroid': 85.0, 'area': 85.0}
    properties_to_check : List[str], optional
        Properties to check for range finding. Default: ['centroid', 'area']
    min_range_length : int
        Minimum number of consecutive slices to form a valid range
    remove_scout : bool
        If True, removes scout slice from DICOM input (first slice). Default: False
        
    Returns:
    --------
    Tuple[List[List[int]], str, str]
        - mask_slice_ranges: List of [start, end] slice index ranges
        - nifti_file_path: Path to the NIfTI file used for processing
        - mask_nifti_path: Path to the registered brain mask NIfTI file
    """
    
    # Set defaults
    if percentiles_dict is None:
        percentiles_dict = {'centroid': 85.0, 'area': 85.0}
    if properties_to_check is None:
        properties_to_check = ['centroid', 'area']
    
    print("="*80)
    print("🚀 Starting Preprocessing Pipeline")
    print("="*80)

    print("\n--- Step 1: Handling Input and Generating Brain Mask ---")
    
    temp_dir = None
    source_image_path = None
    base_name = ""
    is_dicom_input = False

    try:
        # Determine if input is DICOM directory or NIfTI file
        if os.path.isdir(input_path):
            is_dicom_input = True
            # Sanitize folder name: replace spaces with underscores
            base_name = os.path.basename(os.path.normpath(input_path)).replace(' ', '_')
            
            # Set output directory
            if output_dir is None:
                output_dir = input_path
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Convert DICOM to NIfTI
            temp_dir = tempfile.mkdtemp()
            clean_dicom_dir = os.path.join(temp_dir, 'clean_dicoms')
            nifti_output_dir = os.path.join(temp_dir, 'nifti_output')
            os.makedirs(clean_dicom_dir)
            os.makedirs(nifti_output_dir)

            dicoms = [pydicom.dcmread(os.path.join(input_path, f)) for f in os.listdir(input_path) if f.lower().endswith('.dcm')]
            sorted_dicoms = sorted(dicoms, key=lambda d: d.InstanceNumber)

            if remove_scout:
                print(f"Found {len(sorted_dicoms)} DICOMs. Removing scout slice and keeping {len(sorted_dicoms) - 1}.")
                # Skip first slice (scout)
                for dcm in sorted_dicoms[1:]:
                    shutil.copy(dcm.filename, clean_dicom_dir)
            else:
                print(f"Found {len(sorted_dicoms)} DICOMs. Keeping all slices (scout removal disabled).")
                for dcm in sorted_dicoms:
                    shutil.copy(dcm.filename, clean_dicom_dir)

            dicom2nifti.convert_directory(clean_dicom_dir, nifti_output_dir, compression=True, reorient=False)
            
            converted_files = os.listdir(nifti_output_dir)
            if not converted_files:
                raise RuntimeError("❌ Error: dicom2nifti failed to create a NIfTI file.")
            converted_file = os.path.join(nifti_output_dir, converted_files[0])

            # Save the converted NIfTI to output directory
            source_image_path = os.path.join(output_dir, f"{base_name}_converted.nii.gz")
            shutil.copy(converted_file, source_image_path)
            print(f"💾 Saved converted NIfTI to: {source_image_path}")

        # Handle NIfTI input
        elif input_path.lower().endswith(('.nii', '.nii.gz')):
            source_image_path = input_path
            base_name = os.path.basename(input_path).replace('.nii.gz', '').replace('.nii', '')
            
            # Set output directory
            if output_dir is None:
                output_dir = os.path.dirname(input_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
            raise ValueError(f"❌ Error: Input path is not a valid NIfTI file or DICOM directory.")

        # Perform Registration
        print("\n🧠 Loading images for registration...")
        moving_image = ants.image_read(source_image_path)
        fixed_image = ants.image_read(template_path)
        template_mask = ants.image_read(template_mask_path)
        
        print("⏳ Performing registration (this may take several minutes)...")
        reg = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN')
        
        print("🔄 Applying inverse transform to create the brain mask...")
        warped_mask = ants.apply_transforms(fixed=moving_image, moving=template_mask, transformlist=reg['invtransforms'], interpolator='nearestNeighbor')
        
        output_mask_path = os.path.join(output_dir, f"{base_name}_registered.nii.gz")
        print(f"💾 Saving generated brain mask to: {output_mask_path}")
        ants.image_write(warped_mask, output_mask_path)

        # --- Step 2: Find Anchor Range ---
        print("\n--- Step 2: Finding Anchor Range ---")
        
        original_data = nib.load(source_image_path).get_fdata()
        mask_data = nib.load(output_mask_path).get_fdata()
        
        if original_data.shape != mask_data.shape:
             raise ValueError(f"❌ Shape Mismatch: Original {original_data.shape}, Mask {mask_data.shape}")

        print(f"Image and mask shapes are valid: {original_data.shape}")
        
        all_slice_properties = calculate_all_properties_extended(original_data, mask_data, {'shape': True, 'intensity': True})
        valid_slices = sorted([s for s, p in all_slice_properties.items() if p['centroid'] is not None])

        mask_slice_ranges = []
        if len(valid_slices) >= min_range_length:
            precomputed_diffs = precompute_inter_slice_differences(all_slice_properties, valid_slices, properties_to_check)
            mask_slice_ranges = find_reliable_ranges_from_diffs(valid_slices, all_slice_properties, precomputed_diffs, percentiles_dict, properties_to_check, min_range_length)
        else:
            print(f"⚠️ Insufficient valid slices ({len(valid_slices)}) found to predict ranges.")

        print(f"\n✅ Predicted Reliable Ranges: {mask_slice_ranges}")
        print("\n✨ Preprocessing complete! ✨")
        
        return mask_slice_ranges, source_image_path, output_mask_path

    except Exception as e:
        print(f"❌ An error occurred during the preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if temp_dir:
            print(f"🧹 Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


def get_texture_feature_names(lbp_n_points):
    distances = [2]
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    texture_feature_names = (
        [f'{prop}_dist{2}_{angle}' for angle in angles_deg for prop in properties] +
        ['skewness', 'kurtosis', 'entropy'] +
        [f'lbp_{i}' for i in range(lbp_n_points + 2)] +
        ['sobel_mean', 'sobel_std', 'canny_density', 'laplacian_mean', 'laplacian_std'] +
        [f"gabor_s{round(sigma,2)}_t{round(theta,2)}_l{round(lambd,2)}_g{round(gamma,2)}_p{round(psi,2)}_{stat}"
         for sigma, theta, lambd, gamma, psi in itertools.product(*GABOR_PARAMS.values()) for stat in ['mean', 'std']] +
        ['wavelet_cA_mean', 'wavelet_cA_std', 'wavelet_cH_mean', 'wavelet_cH_std',
         'wavelet_cV_mean', 'wavelet_cV_std', 'wavelet_cD_mean', 'wavelet_cD_std']
    )
    return texture_feature_names
def remove_small_holes_gpu(mask_gpu, area_threshold):
    """
    GPU equivalent of remove_small_holes: Invert, remove small objects, invert back.
    Exact match to CPU version for accuracy.
    """
    inverted = 1 - mask_gpu
    cleaned = cucim_remove_small_objects(inverted, min_size=area_threshold)
    return 1 - cleaned
def apply_surgical_grade_3d_consistency_gpu(volume_data_gpu, slice_range):
    """
    Fully GPU-accelerated version of surgical-grade 3D consistency.
    All operations use CuPy/CuCIM equivalents for accuracy.
    """
    print("Applying surgical-grade 3D processing (GPU)...")
   
    # Step 1: Anisotropic smoothing
    volume_float = volume_data_gpu.astype(cp.float32)
    smoothed_volume = cp.zeros_like(volume_float)
   
    for i in range(volume_data_gpu.shape[2]):
        current_slice = volume_float[:, :, i]
        smoothed_slice = cupy_gaussian_filter(current_slice, sigma=1.0)
        smoothed_volume[:, :, i] = smoothed_slice
   
    # Step 2: 3D bilateral filtering simulation
    print("Applying 3D bilateral filtering (GPU)...")
    for i in range(1, volume_data_gpu.shape[2] - 1):
        current = smoothed_volume[:, :, i]
        prev_slice = smoothed_volume[:, :, i - 1]
        next_slice = smoothed_volume[:, :, i + 1]
       
        weight_prev = cp.exp(-((current - prev_slice) ** 2) / 0.1)
        weight_next = cp.exp(-((current - next_slice) ** 2) / 0.1)
        weight_current = cp.ones_like(current)
       
        total_weight = weight_prev + weight_next + weight_current
        result = (weight_current * current + weight_prev * prev_slice + weight_next * next_slice) / total_weight
        smoothed_volume[:, :, i] = result
   
    # Step 3: Conservative thresholding
    print("Applying conservative binary conversion (GPU)...")
    binary_volume = cp.zeros_like(smoothed_volume, dtype=cp.uint8)
   
    for i in range(volume_data_gpu.shape[2]):
        slice_data = smoothed_volume[:, :, i]
        original_has_brain = cp.sum(volume_data_gpu[:, :, i] > 0.5) > 100
       
        if not original_has_brain:
            binary_volume[:, :, i] = cp.zeros_like(slice_data, dtype=cp.uint8)
        else:
            threshold = 0.7
            binary_slice = (slice_data > threshold).astype(cp.uint8)
            binary_volume[:, :, i] = binary_slice
   
    # Step 4: Conservative morphological operations
    print("Applying conservative morphological operations (GPU)...")
    selem = _create_gpu_disk(1) # Assuming you have this function
   
    for i in range(volume_data_gpu.shape[2]):
        current_slice = binary_volume[:, :, i]
       
        if cp.sum(current_slice) == 0:
            continue
           
        current_slice = cucim_remove_small_objects(current_slice, min_size=50)
       
        if cp.sum(current_slice) > 0:
            current_slice = cupy_binary_closing(current_slice, structure=selem)
            current_slice = remove_small_holes_gpu(current_slice, area_threshold=25).astype(cp.uint8)
   
        binary_volume[:, :, i] = current_slice
   
    # Step 5: Edge-aware consistency
    print("Applying edge-aware consistency (GPU)...")
    for slice_idx in range(1, binary_volume.shape[2] - 1):
        current_slice = binary_volume[:, :, slice_idx]
        prev_slice = binary_volume[:, :, slice_idx - 1]
        next_slice = binary_volume[:, :, slice_idx + 1]
       
        prev_overlap = cp.sum(current_slice & prev_slice) / (cp.sum(current_slice | prev_slice) + 1e-8)
        next_overlap = cp.sum(current_slice & next_slice) / (cp.sum(current_slice | next_slice) + 1e-8)
       
        consistency_threshold = 0.7
       
        if prev_overlap < consistency_threshold or next_overlap < consistency_threshold:
            neighbor_avg = (prev_slice.astype(cp.float32) + next_slice.astype(cp.float32)) / 2.0
            current_smooth = cupy_gaussian_filter(current_slice.astype(cp.float32), sigma=1.0)
            blended = 0.6 * current_smooth + 0.4 * neighbor_avg
            binary_volume[:, :, slice_idx] = (blended > 0.5).astype(cp.uint8)
   
    # Step 6: Final hole filling
    print("Final cleanup (GPU)...")
    for slice_idx in range(binary_volume.shape[2]):
        binary_volume[:, :, slice_idx] = cupy_binary_fill_holes(binary_volume[:, :, slice_idx]).astype(cp.uint8)
   
    return binary_volume
def parse_slice_ranges(slice_ranges, total_slices):
    """
    Parses a list of ranges and single slice indices into a set of all known slices.
    Example input: [[30, 50], 55, [64, 90]]
    """
    known_slices_set = set()
    for item in slice_ranges:
        if isinstance(item, int):
            known_slices_set.add(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
            known_slices_set.update(range(start, end + 1))
   
    all_slices = set(range(total_slices))
    unknown_slices_set = all_slices - known_slices_set
   
    return known_slices_set, sorted(list(unknown_slices_set))
def local_binary_pattern_torch(image, radius, method='uniform'):
    points = 8 * radius
    height, width = image.shape
    angles = 2 * torch.pi / points * torch.arange(points, dtype=torch.float32, device=device)
    offsets_x = radius * torch.cos(angles)
    offsets_y = radius * torch.sin(angles)
    w_norm = 2.0 / (width - 1)
    h_norm = 2.0 / (height - 1)
    bit_string = torch.zeros((height, width, points), device=device, dtype=torch.bool)
    for k in range(points):
        dx = offsets_x[k]
        dy = offsets_y[k]
        shift_x = -dx * w_norm
        shift_y = -dy * h_norm
        theta = torch.tensor([[[1, 0, shift_x], [0, 1, shift_y]]], device=device, dtype=torch.float32)
        grid = F.affine_grid(theta, (1, 1, height, width), align_corners=False)
        sampled = F.grid_sample(image.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(0).squeeze(0)
        bit_string[..., k] = sampled >= image
    rolled = torch.roll(bit_string, shifts=1, dims=-1)
    diff = bit_string != rolled
    transitions = diff.sum(dim=-1, dtype=torch.int32)
    sum_1s = bit_string.sum(dim=-1, dtype=torch.uint8)
    lbp_value = torch.where(transitions <= 2, sum_1s, torch.tensor(points + 1, dtype=torch.uint8, device=device))
    return lbp_value.float()
def compute_glcm_batch(patches, distances, angles_deg, glcm_levels):
    angles = [np.deg2rad(ang) for ang in angles_deg]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    num_patches = patches.shape[0]
    glcm_features = []
    # Precompute indices
    i_indices = torch.arange(glcm_levels, device=device)
    j_indices = i_indices.clone()
    i_matrix = i_indices.view(1, -1, 1)
    j_matrix = j_indices.view(1, 1, -1)
    for d in distances:
        for ang in angles:
            dy = int(round(d * np.sin(ang)))
            dx = int(round(d * np.cos(ang)))
            dy_abs, dx_abs = abs(dy), abs(dx)
            # Calculate valid region
            h = patches.shape[1] - dy_abs
            w = patches.shape[2] - dx_abs
            if h <= 0 or w <= 0:
                glcm_features.extend([torch.zeros(num_patches, device=device)] * len(properties))
                continue
            y_start = max(0, -dy)
            y_end = min(patches.shape[1], patches.shape[1] - dy)
            x_start = max(0, -dx)
            x_end = min(patches.shape[2], patches.shape[2] - dx)
            if y_end <= y_start or x_end <= x_start:
                glcm_features.extend([torch.zeros(num_patches, device=device)] * len(properties))
                continue
            # Extract paired patches
            patch_left = patches[:, y_start:y_end, x_start:x_end]
            patch_right = patches[:, y_start+dy:y_end+dy, x_start+dx:x_end+dx]
            # Compute GLCM
            pair_indices = patch_left * glcm_levels + patch_right
            flat_pair_indices = pair_indices.view(num_patches, -1)
            patch_indices = torch.arange(num_patches, device=device).unsqueeze(1).expand(-1, flat_pair_indices.shape[1])
            global_indices = patch_indices * (glcm_levels ** 2) + flat_pair_indices
            hist = torch.bincount(global_indices.flatten(), minlength=num_patches * (glcm_levels ** 2))
            glcm = hist.view(num_patches, glcm_levels, glcm_levels).float()
            glcm /= glcm.sum(dim=(1, 2), keepdim=True) + 1e-6
            # Vectorized feature calculations
            contrast = torch.sum((i_matrix - j_matrix)**2 * glcm, dim=(1, 2))
            dissimilarity = torch.sum(torch.abs(i_matrix - j_matrix) * glcm, dim=(1, 2))
            homogeneity = torch.sum(glcm / (1.0 + (i_matrix - j_matrix)**2), dim=(1, 2))
            energy = torch.sum(glcm**2, dim=(1, 2))
            asm = energy
            mean_i = torch.sum(i_matrix * glcm.sum(dim=2, keepdim=True), dim=(1, 2))
            mean_j = torch.sum(j_matrix * glcm.sum(dim=1, keepdim=True), dim=(1, 2))
            std_i = torch.sqrt(torch.sum((i_matrix - mean_i.view(-1, 1, 1))**2 * glcm.sum(dim=2, keepdim=True), dim=(1, 2)))
            std_j = torch.sqrt(torch.sum((j_matrix - mean_j.view(-1, 1, 1))**2 * glcm.sum(dim=1, keepdim=True), dim=(1, 2)))
            correlation = torch.sum((i_matrix - mean_i.view(-1, 1, 1)) * (j_matrix - mean_j.view(-1, 1, 1)) * glcm, dim=(1, 2)) / (std_i * std_j + 1e-6)
            glcm_features.extend([contrast, dissimilarity, homogeneity, energy, correlation, asm])
    return torch.stack(glcm_features, dim=1)
def compute_batch_features_optimized(batch, lbp_n_points, glcm_levels,
                                   sobel_kernel_x, sobel_kernel_y, laplacian_kernel):
    """
    Optimized batch feature computation with reduced memory allocations
    """
    device = batch['patches'].device
    batch_size = batch['patches'].shape[0]
   
    # Use existing compute functions but with memory optimizations
    distances = [2]
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
   
    # Compute features in sequence to avoid memory spikes
    mask_sum = batch['mask'].sum(dim=(1, 2))
    valid_mask = mask_sum > 0
   
    # GLCM Features (only compute if needed)
    glcm_features = torch.zeros(batch_size, len(properties) * len(distances) * len(angles_deg),
                               device=device, dtype=torch.float32)
    if valid_mask.any() and batch['glcm'].shape[0] > 0:
        valid_glcm = batch['glcm'][valid_mask]
        min_val = torch.amin(valid_glcm, dim=(1, 2), keepdim=True)
        max_val = torch.amax(valid_glcm, dim=(1, 2), keepdim=True)
        glcm_valid_mask = (max_val > min_val).squeeze() & torch.isfinite(valid_glcm).all(dim=(1, 2))
       
        if glcm_valid_mask.any():
            glcm_features_valid = compute_glcm_batch(valid_glcm[glcm_valid_mask],
                                                   distances, angles_deg, glcm_levels)
            # Create a mask for valid_mask positions that also satisfy glcm_valid_mask
            valid_indices = torch.where(valid_mask)[0]
            glcm_valid_indices = valid_indices[glcm_valid_mask]
            glcm_features[glcm_valid_indices] = glcm_features_valid
   
    # Histogram Features
    hist_features = torch.zeros(batch_size, 3, device=device, dtype=torch.float32)
    if valid_mask.any():
        data = batch['patches'][valid_mask].view(valid_mask.sum(), -1)
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True) + 1e-8
        centered = data - mean
        skewness = (centered**3).mean(dim=1) / (std.squeeze()**3)
        kurt = (centered**4).mean(dim=1) / (std.squeeze()**4) - 3
       
        # Efficient entropy calculation
        hist = torch.stack([torch.histc(d, bins=32, min=0.0, max=1.0) # Reduced bins
                          for d in data])
        hist_normalized = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -torch.sum(hist_normalized * torch.log(hist_normalized + 1e-8), dim=1)
       
        hist_features[valid_mask] = torch.stack([skewness, kurt, entropy], dim=1)
   
    # LBP Features
    lbp_hist = torch.zeros(batch_size, lbp_n_points + 2, device=device, dtype=torch.float32)
    if valid_mask.any():
        lbp_values = batch['lbp'][valid_mask].view(valid_mask.sum(), -1)
        # Vectorized histogram calculation
        bins = torch.arange(lbp_n_points + 3, device=device)
        hist = torch.stack([torch.histc(lbp_vec, bins=lbp_n_points + 2, min=0, max=lbp_n_points + 1)
                          for lbp_vec in lbp_values])
        hist_normalized = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        lbp_hist[valid_mask] = hist_normalized
   
    # Edge Features
    edge_features = torch.zeros(batch_size, 5, device=device, dtype=torch.float32)
    if valid_mask.any():
        patches_valid = batch['patches'][valid_mask].unsqueeze(1)
        num_valid = patches_valid.shape[0]
       
        sobel_x = F.conv2d(patches_valid, sobel_kernel_x, padding=1)
        sobel_y = F.conv2d(patches_valid, sobel_kernel_y, padding=1)
        sobel = torch.sqrt(sobel_x**2 + sobel_y**2)
        canny = ((sobel > 100) & (sobel < 200)).float()
        laplacian = F.conv2d(patches_valid, laplacian_kernel, padding=1)
       
        sobel_flat = sobel.view(num_valid, -1)
        canny_flat = canny.view(num_valid, -1)
        laplacian_flat = laplacian.view(num_valid, -1)
       
        edge_features[valid_mask, 0] = sobel_flat.mean(dim=1)
        edge_features[valid_mask, 1] = sobel_flat.std(dim=1)
        edge_features[valid_mask, 2] = canny_flat.mean(dim=1)
        edge_features[valid_mask, 3] = laplacian_flat.mean(dim=1)
        edge_features[valid_mask, 4] = laplacian_flat.std(dim=1)
   
    # Gabor Features
    gabor_features = torch.zeros(batch_size, batch['gabor'].shape[1] * 2,
                                device=device, dtype=torch.float32)
    if valid_mask.any():
        gabor_data = batch['gabor'][valid_mask].view(valid_mask.sum(), batch['gabor'].shape[1], -1)
        gabor_means = gabor_data.mean(dim=2)
        gabor_stds = gabor_data.std(dim=2)
        gabor_features[valid_mask] = torch.cat([gabor_means, gabor_stds], dim=1)
   
    # Wavelet Features
    wavelet_features = torch.zeros(batch_size, 8, device=device, dtype=torch.float32)
    if valid_mask.any():
        low = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=device).view(1, 1, 2, 2)
        high = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], device=device).view(1, 1, 2, 2)
        img_float = batch['patches'][valid_mask].unsqueeze(1).float()
       
        cA = F.conv2d(img_float, low, stride=2).squeeze(1)
        cH = F.conv2d(img_float, high, stride=2).squeeze(1)
        cV = F.conv2d(img_float, high.transpose(2, 3), stride=2).squeeze(1)
        cD = F.conv2d(img_float, -high, stride=2).squeeze(1)
       
        wavelet_features[valid_mask, 0] = cA.mean(dim=(1, 2))
        wavelet_features[valid_mask, 1] = cA.std(dim=(1, 2))
        wavelet_features[valid_mask, 2] = cH.mean(dim=(1, 2))
        wavelet_features[valid_mask, 3] = cH.std(dim=(1, 2))
        wavelet_features[valid_mask, 4] = cV.mean(dim=(1, 2))
        wavelet_features[valid_mask, 5] = cV.std(dim=(1, 2))
        wavelet_features[valid_mask, 6] = cD.mean(dim=(1, 2))
        wavelet_features[valid_mask, 7] = cD.std(dim=(1, 2))
   
    return torch.cat([glcm_features, hist_features, lbp_hist, edge_features,
                     gabor_features, wavelet_features], dim=1)
def extract_head_mask_gpu(image_gpu):
    """
    Fully GPU-accelerated head mask extraction.
    Accepts and returns a CuPy array.
    NOTE: Caching is omitted for simplicity in this pure GPU version.
    """
    # 2. Apply GPU-based Gaussian smoothing
    smoothed_gpu = cupy_gaussian_filter(image_gpu, sigma=1.0)

    # 3. Simple thresholding on GPU
    threshold = smoothed_gpu.mean()
    head_mask_gpu = (smoothed_gpu > threshold).astype(cp.uint8)

    # 4. GPU-based morphological operations to clean up the mask
    selem_closing_gpu = _create_gpu_disk(radius=5)
    closed_gpu = cupy_binary_closing(head_mask_gpu, structure=selem_closing_gpu)
    
    # 5. Fill holes on GPU
    filled_gpu = cupy_binary_fill_holes(closed_gpu)
    
    # 6. Remove small objects on GPU
    cleaned_gpu = cucim_remove_small_objects(filled_gpu.astype(bool), min_size=500)

    return cleaned_gpu.astype(cp.uint8)
def extract_non_brain_mask(original_slice_np, segmentation_slice_np):
    original_img = sitk.GetImageFromArray(original_slice_np.T)
    segmentation_img = sitk.GetImageFromArray(segmentation_slice_np.T)
    head_mask_img = sitk.LiThreshold(original_img, 0, 1)
    # Ensure C-contiguity for arrays passed to OpenCV
    head_mask_slice = np.ascontiguousarray((sitk.GetArrayFromImage(head_mask_img).T > 0).astype(np.uint8))
    brain_mask_slice = np.ascontiguousarray((sitk.GetArrayFromImage(segmentation_img).T > 0).astype(np.uint8))
    head_contours, _ = cv2.findContours(head_mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not head_contours:
        return np.zeros_like(head_mask_slice)
    hull = cv2.convexHull(np.concatenate(head_contours))
    brain_contours, _ = cv2.findContours(brain_mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not brain_contours:
        return np.zeros_like(head_mask_slice)
    Cb = max(brain_contours, key=cv2.contourArea)
    non_brain_mask = np.zeros_like(head_mask_slice, dtype=np.uint8)
    cv2.drawContours(non_brain_mask, [hull], -1, 1, thickness=cv2.FILLED)
    cv2.drawContours(non_brain_mask, [Cb], -1, 0, thickness=cv2.FILLED)
    return non_brain_mask
def extract_features_mixed_optimized(image_slice, image_slice_raw, mixed_mask, fine_params,
                                   slice_idx, global_min, global_max, total_slices,
                                   gabor_kernels_stacked, sobel_kernel_x, sobel_kernel_y,
                                   laplacian_kernel, grid=False, return_df=True):
    """
    Optimized version with batch processing and reduced memory overhead
    """
    patch_size = fine_params['patch_size']
    stride = patch_size if grid else fine_params['stride']
    lbp_radius = fine_params['lbp_radius']
    glcm_levels = fine_params['glcm_levels']
    valid_patch_threshold = fine_params['valid_patch_threshold']
    lbp_n_points = 8 * lbp_radius
   
    # Early return if mask is empty
    if np.sum(mixed_mask) == 0:
        if return_df:
            texture_feature_names = get_texture_feature_names(lbp_n_points)
            base_columns = ['slice_idx', 'y', 'x', 'norm_x', 'norm_y', 'norm_slice']
            feature_names = base_columns + texture_feature_names
            return pd.DataFrame(columns=feature_names)
        else:
            return None, None, None, None, None
    # Move data to GPU in single batch
    with torch.cuda.device(device):
        image_slice_t = torch.tensor(image_slice, device=device, dtype=torch.float32)
        image_slice_raw_t = torch.tensor(image_slice_raw, device=device, dtype=torch.float32)
        mixed_mask_t = torch.tensor(mixed_mask, device=device, dtype=torch.float32)
    # Compute LBP and GLCM in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        lbp_future = executor.submit(local_binary_pattern_torch, image_slice_t, lbp_radius)
        # GLCM normalization
        image_slice_glcm = (image_slice_raw_t - global_min) / (global_max - global_min + 1e-6) * (glcm_levels - 1)
        image_slice_glcm = image_slice_glcm.round().to(torch.uint8)
        lbp_image = lbp_future.result()
    height, width = image_slice.shape
    ph, pw = patch_size
    sh, sw = stride
   
    # Vectorized patch creation
    def create_patches_fast(tensor):
        patches = tensor.unfold(0, ph, sh).unfold(1, pw, sw)
        return patches.contiguous().view(-1, ph, pw)
   
    patches = create_patches_fast(image_slice_t)
    mask_patches = create_patches_fast(mixed_mask_t)
    lbp_patches = create_patches_fast(lbp_image)
    glcm_patches = create_patches_fast(image_slice_glcm)
   
    # Filter valid patches
    mixed_ratios = mask_patches.mean(dim=(1, 2))
    valid_idx = mixed_ratios >= valid_patch_threshold
   
    if not valid_idx.any():
        if return_df:
            texture_feature_names = get_texture_feature_names(lbp_n_points)
            base_columns = ['slice_idx', 'y', 'x', 'norm_x', 'norm_y', 'norm_slice']
            feature_names = base_columns + texture_feature_names
            return pd.DataFrame(columns=feature_names)
        else:
            return None, None, None, None, None
    # Extract only valid patches to save memory
    valid_patches = {
        'image': patches[valid_idx],
        'mask': mask_patches[valid_idx],
        'lbp': lbp_patches[valid_idx],
        'glcm': glcm_patches[valid_idx]
    }
   
    # Compute features with optimized batch size
    features = compute_features_gpu_optimized(
        image_slice_t,
        valid_patches['image'],
        valid_patches['mask'],
        valid_patches['lbp'],
        (ph, pw),
        (sh, sw),
        valid_idx,
        lbp_n_points,
        glcm_levels,
        valid_patches['glcm'],
        gabor_kernels_stacked, sobel_kernel_x, sobel_kernel_y, laplacian_kernel
    )
   
    # Process coordinates efficiently
    y_coords = torch.arange(0, height - ph + 1, sh, device=device)
    x_coords = torch.arange(0, width - pw + 1, sw, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)[valid_idx]
   
    norm_x = coords[:, 1] / (width - pw)
    norm_y = coords[:, 0] / (height - ph)
    norm_slice_val = slice_idx / (total_slices - 1)
   
    if not return_df:
        return features, coords, norm_x, norm_y, norm_slice_val
   
    # Efficient DataFrame creation
    texture_feature_names = get_texture_feature_names(lbp_n_points)
    num_coords = len(coords)
   
    # Pre-allocate arrays
    data_arrays = {
        'slice_idx': np.full(num_coords, slice_idx, dtype=np.int32),
        'y': coords[:, 0].cpu().numpy().astype(np.int32),
        'x': coords[:, 1].cpu().numpy().astype(np.int32),
        'norm_x': norm_x.cpu().numpy().astype(np.float32),
        'norm_y': norm_y.cpu().numpy().astype(np.float32),
        'norm_slice': np.full(num_coords, float(norm_slice_val), dtype=np.float32),
    }
   
    features_np = features.cpu().numpy().astype(np.float32)
    coords_df = pd.DataFrame(data_arrays)
    features_df = pd.DataFrame(features_np, columns=texture_feature_names)
   
    # Clean up GPU memory
    del patches, mask_patches, lbp_patches, glcm_patches, valid_patches
    torch.cuda.empty_cache()
   
    return pd.concat([coords_df, features_df], axis=1)
def compute_features_gpu_optimized(image_slice_t, patches, mask_patches, lbp_patches,
                                 patch_size, stride, valid_idx, lbp_n_points,
                                 glcm_levels, glcm_patches,
                                 gabor_kernels_stacked, sobel_kernel_x, sobel_kernel_y, laplacian_kernel,
                                 batch_size=4096): # Reduced batch size for better memory management
    """
    Optimized feature computation with better memory management and parallelization
    """
    device = patches.device
    num_patches = patches.shape[0]
    ph, pw = patch_size
    sh, sw = stride
   
    # Pre-compute Gabor responses for entire slice (more efficient)
    with torch.no_grad():
        input_img = image_slice_t.unsqueeze(0).unsqueeze(0)
        gabor_responses = F.conv2d(input_img, gabor_kernels_stacked, padding=15)
        gabor_responses = gabor_responses.squeeze(0)
       
        # Extract Gabor patches efficiently
        gabor_patches = gabor_responses.unfold(1, ph, sh).unfold(2, pw, sw)
        gabor_patches = gabor_patches.permute(1, 2, 0, 3, 4).reshape(-1, gabor_responses.shape[0], ph, pw)
        gabor_patches = gabor_patches[valid_idx]
   
    num_batches = (num_patches + batch_size - 1) // batch_size
    features_list = []
   
    # Process batches with memory cleanup
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_patches)
       
        batch_data = {
            'patches': patches[start:end],
            'mask': mask_patches[start:end],
            'lbp': lbp_patches[start:end],
            'glcm': glcm_patches[start:end],
            'gabor': gabor_patches[start:end]
        }
       
        batch_features = compute_batch_features_optimized(batch_data, lbp_n_points, glcm_levels,
                                                         sobel_kernel_x, sobel_kernel_y, laplacian_kernel)
        features_list.append(batch_features)
       
        # Clean up intermediate tensors
        if i < num_batches - 1: # Keep last batch for return
            del batch_data
            torch.cuda.empty_cache()
   
    # Clean up large tensors
    del gabor_responses, gabor_patches
    torch.cuda.empty_cache()
   
    return torch.cat(features_list, dim=0)

def generate_brain_training_data(image_slice_t, head_mask_t, brain_mask_t, params,
                                 slice_idx, total_slices, gabor_kernels_stacked, sobel_kernel_x,
                                 sobel_kernel_y, laplacian_kernel, image_slice_glcm, train_brain_on_all):
    """
    Generates training data for the brain specialist using GPU-resident tensors.
    """
    height, width = image_slice_t.shape
    ph, pw = params['patch_size']
    sh, sw = params['stride']
    lbp_radius = params['lbp_radius']
    glcm_levels = params['glcm_levels']
    valid_patch_threshold = params['valid_patch_threshold']
    lbp_n_points = 8 * lbp_radius

    # All inputs (image_slice_t, head_mask_t, brain_mask_t, image_slice_glcm) are already on the GPU.
    # No CPU processing or data transfers are needed here.

    non_brain_mask_t = torch.clamp(head_mask_t.float() - brain_mask_t.float(), 0, 1)

    # LBP calculation on GPU
    lbp_image = local_binary_pattern_torch(image_slice_t, lbp_radius)

    # Create patches from GPU tensors
    def create_patches(tensor):
        return tensor.unfold(0, ph, sh).unfold(1, pw, sw).reshape(-1, ph, pw)

    patches = create_patches(image_slice_t)
    head_mask_patches = create_patches(head_mask_t)
    brain_mask_patches = create_patches(brain_mask_t)
    non_brain_mask_patches = create_patches(non_brain_mask_t)
    lbp_patches = create_patches(lbp_image)
    glcm_patches = create_patches(image_slice_glcm)

    # Filter valid patches based on head mask overlap
    head_ratios = head_mask_patches.float().mean(dim=(1, 2))
    valid_idx = head_ratios >= valid_patch_threshold
    
    if not valid_idx.any():
        return pd.DataFrame()

    # Extract valid patches
    valid_patches = {
        'image': patches[valid_idx],
        'mask': head_mask_patches[valid_idx],
        'lbp': lbp_patches[valid_idx],
        'glcm': glcm_patches[valid_idx],
        'brain': brain_mask_patches[valid_idx],
        'non_brain': non_brain_mask_patches[valid_idx]
    }

    # Compute features entirely on GPU
    features = compute_features_gpu_optimized(
        image_slice_t,
        valid_patches['image'],
        valid_patches['mask'],
        valid_patches['lbp'],
        (ph, pw),
        (sh, sw),
        valid_idx,
        lbp_n_points,
        glcm_levels,
        valid_patches['glcm'],
        gabor_kernels_stacked, sobel_kernel_x, sobel_kernel_y, laplacian_kernel
    )

    # Process labels with conditional logic based on train_brain_on_all
    brain_ratios = valid_patches['brain'].float().mean(dim=(1, 2))
    
    if train_brain_on_all:
        labels = torch.zeros_like(brain_ratios, dtype=torch.int64)
        labels[brain_ratios > 0.95] = 2  # Definite brain
        labels[brain_ratios < 0.05] = 0  # Definite non-brain
        labels[(brain_ratios >= 0.05) & (brain_ratios <= 0.95)] = 1  # Mixed
        final_valid_patches_idx = torch.arange(len(features), device=device)
    else:
        final_valid_patches_idx = torch.where(brain_ratios == 1.0)[0]
        if len(final_valid_patches_idx) == 0:
            return pd.DataFrame()
        labels = torch.full((len(final_valid_patches_idx),), 2, dtype=torch.int64, device=device)

    # Process coordinates
    y_range = torch.arange(0, height - ph + 1, sh, device=device)
    x_range = torch.arange(0, width - pw + 1, sw, device=device)
    yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
    all_coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)[valid_idx]

    coords = all_coords[final_valid_patches_idx]
    norm_x = coords[:, 1] / (width - pw)
    norm_y = coords[:, 0] / (height - ph)
    norm_slice_val = slice_idx / (total_slices - 1)
    
    features = features[final_valid_patches_idx]

    # Create DataFrame (this is the main point of GPU -> CPU transfer)
    num_coords = len(coords)
    texture_feature_names = get_texture_feature_names(lbp_n_points)
    data_dict = {
        'slice_idx': np.full(num_coords, slice_idx),
        'y': coords[:, 0].cpu().numpy(),
        'x': coords[:, 1].cpu().numpy(),
        'norm_x': norm_x.cpu().numpy(),
        'norm_y': norm_y.cpu().numpy(),
        'norm_slice': np.full(num_coords, float(norm_slice_val)),
        'label': labels.cpu().numpy(),
    }
    features_np = features.cpu().numpy()
    for i, name in enumerate(texture_feature_names):
        data_dict[name] = features_np[:, i]
        
    return pd.DataFrame(data_dict)


def generate_combined_training_data(image_slice, image_slice_raw, brain_mask_slice, vague_mask_slice, coarse_params, fine_params,
                                    slice_idx, global_min, global_max, total_slices, gabor_kernels_stacked, sobel_kernel_x,
                                    sobel_kernel_y, laplacian_kernel, is_known_slice, kernel=5, use_boundary_specialist=True, train_brain_on_all=True):
    """
    GPU-centric combined function. Moves data to GPU once and keeps it there.
    """
    # === 1. MOVE ALL DATA TO GPU ONCE ===
    image_slice_t = torch.tensor(image_slice, device=device, dtype=torch.float32)
    image_slice_raw_gpu = cp.asarray(image_slice_raw, dtype=cp.float32)
    brain_mask_gpu = cp.asarray(brain_mask_slice, dtype=cp.uint8)
    
    # === 2. PERFORM ALL PRE-PROCESSING ON GPU ===
    # Head Mask Extraction on GPU
    head_mask_gpu = extract_head_mask_gpu(image_slice_raw_gpu)
    
    # Boundary Belt Creation on GPU
    boundary_belt_gpu = None
    if use_boundary_specialist:
        selem_gpu = _create_gpu_disk(kernel)
        dilated_gpu = cupy_binary_dilation(brain_mask_gpu, structure=selem_gpu)
        eroded_gpu = cupy_binary_erosion(brain_mask_gpu, structure=selem_gpu)
        boundary_belt_gpu = cp.logical_and(dilated_gpu, cp.logical_not(eroded_gpu)).astype(cp.uint8)

    # LBP and GLCM calculations (already GPU-based, but now use GPU inputs)
    image_slice_glcm_cupy = (image_slice_raw_gpu - global_min) / (global_max - global_min + 1e-6) * (coarse_params['glcm_levels'] - 1)
    image_slice_glcm_cupy = image_slice_glcm_cupy.round()

    # Convert from CuPy array to PyTorch tensor BEFORE using .to()
    image_slice_glcm = torch.as_tensor(image_slice_glcm_cupy, device=device).to(torch.uint8)
    
    # Convert CuPy arrays to PyTorch tensors for consistency in downstream functions
    head_mask_t = torch.as_tensor(head_mask_gpu, device=device)
    brain_mask_t = torch.as_tensor(brain_mask_gpu, device=device)
    
    # === 3. CALL WORKERS WITH GPU-RESIDENT DATA ===
    # Brain data (coarse params)
    brain_df = generate_brain_training_data(
        image_slice_t, head_mask_t, brain_mask_t, coarse_params,
        slice_idx, total_slices, gabor_kernels_stacked, sobel_kernel_x, sobel_kernel_y, 
        laplacian_kernel, image_slice_glcm, train_brain_on_all=train_brain_on_all
    )
    
    # Boundary data (fine params, if enabled)
    boundary_df = pd.DataFrame()
    if use_boundary_specialist:
        boundary_belt_t = torch.as_tensor(boundary_belt_gpu, device=device)
        boundary_df = generate_boundary_training_data(
            image_slice_t, head_mask_t, brain_mask_t, boundary_belt_t, fine_params,
            slice_idx, total_slices, gabor_kernels_stacked, sobel_kernel_x, 
            sobel_kernel_y, laplacian_kernel, image_slice_glcm
        )
        
    return brain_df, boundary_df

def generate_boundary_training_data(image_slice_t, head_mask_t, brain_mask_t, boundary_belt_t, params,
                                      slice_idx, total_slices, gabor_kernels_stacked, sobel_kernel_x,
                                      sobel_kernel_y, laplacian_kernel, image_slice_glcm):
    """
    Generates training data for the boundary specialist using GPU-resident tensors.
    """
    height, width = image_slice_t.shape
    ph, pw = params['patch_size']
    sh, sw = params['stride']
    lbp_radius = params['lbp_radius']
    glcm_levels = params['glcm_levels']
    lbp_n_points = 8 * lbp_radius

    # All inputs are already on the GPU.
    # The expensive boundary_belt creation is done in the parent function.

    # LBP calculation on GPU
    lbp_image = local_binary_pattern_torch(image_slice_t, lbp_radius)

    # Create patches from GPU tensors
    def create_patches(tensor):
        return tensor.unfold(0, ph, sh).unfold(1, pw, sw).reshape(-1, ph, pw)

    patches = create_patches(image_slice_t)
    head_mask_patches = create_patches(head_mask_t)
    brain_mask_patches = create_patches(brain_mask_t)
    boundary_patches = create_patches(boundary_belt_t)
    lbp_patches = create_patches(lbp_image)
    glcm_patches = create_patches(image_slice_glcm)

    # Filter valid patches: overlap with boundary > 0 and head ratio >= 0.2
    boundary_sums = boundary_patches.sum(dim=(1, 2))
    head_ratios = head_mask_patches.float().mean(dim=(1, 2))
    valid_idx = (boundary_sums > 0) & (head_ratios >= 0.2)
    
    if not valid_idx.any():
        return pd.DataFrame()

    # Extract valid patches
    valid_patches = {
        'image': patches[valid_idx],
        'mask': head_mask_patches[valid_idx],
        'lbp': lbp_patches[valid_idx],
        'glcm': glcm_patches[valid_idx],
        'brain': brain_mask_patches[valid_idx]
    }

    # Compute features entirely on GPU
    features = compute_features_gpu_optimized(
        image_slice_t,
        valid_patches['image'],
        valid_patches['mask'],
        valid_patches['lbp'],
        (ph, pw),
        (sh, sw),
        valid_idx,
        lbp_n_points,
        glcm_levels,
        valid_patches['glcm'],
        gabor_kernels_stacked, sobel_kernel_x, sobel_kernel_y, laplacian_kernel
    )

    # Process labels (ratios)
    brain_ratios = valid_patches['brain'].float().mean(dim=(1, 2))

    # Process coordinates
    y_range = torch.arange(0, height - ph + 1, sh, device=device)
    x_range = torch.arange(0, width - pw + 1, sw, device=device)
    yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)[valid_idx]
    norm_x = coords[:, 1] / (width - pw)
    norm_y = coords[:, 0] / (height - ph)
    norm_slice_val = slice_idx / (total_slices - 1)

    # Create DataFrame
    num_coords = len(coords)
    texture_feature_names = get_texture_feature_names(lbp_n_points)
    data_dict = {
        'slice_idx': np.full(num_coords, slice_idx),
        'y': coords[:, 0].cpu().numpy(),
        'x': coords[:, 1].cpu().numpy(),
        'norm_x': norm_x.cpu().numpy(),
        'norm_y': norm_y.cpu().numpy(),
        'norm_slice': np.full(num_coords, float(norm_slice_val)),
        'brain_ratio': brain_ratios.cpu().numpy(),
    }
    features_np = features.cpu().numpy()
    for i, name in enumerate(texture_feature_names):
        data_dict[name] = features_np[:, i]
        
    return pd.DataFrame(data_dict)

def prob_voting(patch_probas, patch_positions, image_shape, patch_size, class_idx=None):
    height, width = image_shape
    ph, pw = patch_size
    sum_array = np.zeros((height, width), dtype=np.float32)
    count_array = np.zeros((height, width), dtype=np.int32)
   
    # Convert tensor to numpy if needed
    if isinstance(patch_positions, torch.Tensor):
        patch_positions = patch_positions.cpu().numpy()
   
    for proba, (y, x) in zip(patch_probas, patch_positions):
        y_end = min(y + ph, height)
        x_end = min(x + pw, width)
        if class_idx is not None:
            sum_array[y:y_end, x:x_end] += proba[class_idx]
        else:
            sum_array[y:y_end, x:x_end] += proba
        count_array[y:y_end, x:x_end] += 1
    avg_array = np.zeros_like(sum_array)
    valid = count_array > 0
    avg_array[valid] = sum_array[valid] / count_array[valid]
    return avg_array
def pixel_voting(patch_classes, patch_positions, image_shape, patch_size):
    height, width = image_shape
    ph, pw = patch_size
    vote_array = np.zeros((height, width, 3), dtype=np.float32)
   
    # Convert tensor to numpy if needed
    if isinstance(patch_positions, torch.Tensor):
        patch_positions = patch_positions.cpu().numpy()
   
    for cls, (y, x) in zip(patch_classes, patch_positions):
        y_end = min(y + ph, height)
        x_end = min(x + pw, width)
        vote_array[y:y_end, x:x_end, int(cls)] += 1
    return np.argmax(vote_array, axis=2)
def create_new_masked_data(masked_data, slice_range):
    new_data = np.zeros_like(masked_data)
    m, n = slice_range
    new_data[:, :, m:n+1] = masked_data[:, :, m:n+1]
    return new_data
def identify_redundant_features(csv_data, correlation_threshold=0.9):
    metadata_cols = ['slice_idx', 'y', 'x', 'norm_x', 'norm_y', 'norm_slice', 'label', 'brain_ratio']
    feature_cols = [col for col in csv_data.columns if col not in metadata_cols]
    if not feature_cols:
        print("Error: No features found after excluding metadata.")
        return []
    X = csv_data[feature_cols].astype(np.float32)
    if X.empty:
        print("Warning: Feature matrix is empty.")
        return []
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    return to_drop
def refine_brain_mask_contextual_gpu(predicted_mask_gpu, anchor_mask_gpu):
    """Fully GPU-accelerated version of contextual refinement."""
    # Always use GPU version since we have the libraries available
    
    # Ensure predicted_mask_gpu is boolean
    predicted_mask_gpu = (predicted_mask_gpu > 0).astype(cp.uint8)
    
    # Label connected components
    labeled_pred_gpu = cucim_label(predicted_mask_gpu)
    num_components = int(cp.max(labeled_pred_gpu)) if cp.any(labeled_pred_gpu) else 0
    
    if num_components == 0:
        return predicted_mask_gpu
    
    refined_mask_gpu = cp.zeros_like(predicted_mask_gpu, dtype=cp.uint8)
    
    INCLUSION_THRESHOLD = 0.50
    passed_inclusion_check_regions = []
    
    # cucim_regionprops returns a list of dicts with properties
    regions = cucim_regionprops(labeled_pred_gpu)
    
    for region in regions:
        # Extract needed props (area, label)
        area = region['area']
        label_val = region['label']
        
        component_mask_gpu = (labeled_pred_gpu == label_val)
        intersection_area = cp.sum(component_mask_gpu & anchor_mask_gpu)
        
        if area > 0 and (intersection_area / area) > INCLUSION_THRESHOLD:
            passed_inclusion_check_regions.append(region)
    
    if passed_inclusion_check_regions:
        # Sort by area (descending)
        passed_inclusion_check_regions.sort(key=lambda r: r['area'], reverse=True)
        areas = cp.array([r['area'] for r in passed_inclusion_check_regions])
        
        SIGNIFICANT_DROP_THRESHOLD = 0.40
        ABSOLUTE_MIN_AREA = 20
        indices_to_keep = []
        if len(areas) > 1:
            percentage_drops = (areas[:-1] - areas[1:]) / areas[:-1]
            max_drop = cp.max(percentage_drops)
            max_drop_idx = int(cp.asnumpy(cp.argmax(percentage_drops)))  # Convert to Python int
            if max_drop > SIGNIFICANT_DROP_THRESHOLD:
                indices_to_keep = list(range(max_drop_idx + 1))
            else:
                indices_to_keep = list(range(len(areas)))
        elif len(areas) == 1:
            indices_to_keep = [0]
        
        for i in indices_to_keep:
            region = passed_inclusion_check_regions[i]
            if region['area'] >= ABSOLUTE_MIN_AREA:
                refined_mask_gpu[labeled_pred_gpu == region['label']] = 1
    
    if cp.any(refined_mask_gpu):
        # Apply GPU morphological operations
        refined_mask_gpu = cupy_binary_fill_holes(refined_mask_gpu)
        selem = _create_gpu_disk(2)
        refined_mask_gpu = cupy_binary_closing(refined_mask_gpu, structure=selem)
        selem_small = _create_gpu_disk(1)
        refined_mask_gpu = cupy_binary_opening(refined_mask_gpu, structure=selem_small)
        
        # Smoothing on GPU
        smoothed = cupy_gaussian_filter(refined_mask_gpu.astype(cp.float32), sigma=1.0)
        refined_mask_gpu = (smoothed > 0.5).astype(cp.uint8)
        
        # Remove small objects on GPU, ensuring boolean input
        refined_mask_gpu = cucim_remove_small_objects(refined_mask_gpu.astype(bool), min_size=50)
        refined_mask_gpu = cupy_binary_fill_holes(refined_mask_gpu)
    
    return refined_mask_gpu.astype(cp.uint8)
def apply_temporal_consistency_gpu(current_mask_gpu, anchor_mask_gpu, alpha=0.3):
    """
    GPU version of temporal consistency. Preserves accuracy with CuPy operations.
    """
    if not cp.any(anchor_mask_gpu):
        return current_mask_gpu
   
    current_float = current_mask_gpu.astype(cp.float32)
    anchor_float = anchor_mask_gpu.astype(cp.float32)
   
    current_smooth = cupy_gaussian_filter(current_float, sigma=1.5)
    anchor_smooth = cupy_gaussian_filter(anchor_float, sigma=1.5)
   
    blended = (1 - alpha) * current_smooth + alpha * anchor_smooth
   
    result = (blended > 0.5).astype(cp.uint8)
   
    return result
# REPLACE the entire old generate_plots function with this new one.
def generate_plots(slice_idx, image_slice, working_roi, p_b_map,
                   uncertain_zone, boundary_ratio_map, pre_refinement_mask,
                   final_refined_mask, output_dir):
    """
    Generates a panel of debug plots for the 2-specialist workflow.
    Thread-safe version with proper error handling.
    """
    try:
        # Ensure all inputs are valid numpy arrays
        if not isinstance(image_slice, np.ndarray) or image_slice.size == 0:
            print(f"Warning: Invalid image_slice for slice {slice_idx}, skipping plot")
            return
       
        # Create figure with proper backend handling
        plt.ioff() # Turn off interactive mode
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Slice {slice_idx} - 2-Specialist Workflow", fontsize=16)
        # --- Row 1: Input and Brain Specialist Output ---
        axes[0, 0].imshow(image_slice, cmap='gray')
        axes[0, 0].set_title('1. Original Slice')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(image_slice, cmap='gray')
        if working_roi is not None and working_roi.size > 0:
            axes[0, 1].imshow(working_roi, cmap='Oranges', alpha=0.5)
        axes[0, 1].set_title('2. Working ROI (Dilated Vague Mask)')
        axes[0, 1].axis('off')
       
        axes[0, 2].imshow(image_slice, cmap='gray')
        if p_b_map is not None and p_b_map.size > 0:
            axes[0, 2].imshow(p_b_map, cmap='Greens', alpha=0.6, vmin=0, vmax=1)
        axes[0, 2].set_title('3. Brain Specialist Probs (in ROI)')
        axes[0, 2].axis('off')
        axes[0, 3].imshow(image_slice, cmap='gray')
        if p_b_map is not None and working_roi is not None and p_b_map.size > 0 and working_roi.size > 0:
            definite_brain_mask = (p_b_map > 0.5) & working_roi.astype(bool)
            axes[0, 3].imshow(definite_brain_mask, cmap='Greens', alpha=0.7)
        axes[0, 3].set_title('4. Definite Brain Mask')
        axes[0, 3].axis('off')
        # --- Row 2: Boundary Specialist and Final Output ---
        if uncertain_zone is not None and uncertain_zone.size > 0:
            axes[1, 0].imshow(uncertain_zone, cmap='gray')
        axes[1, 0].set_title('5. Uncertain Zone for Boundary Splst')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(image_slice, cmap='gray')
        if boundary_ratio_map is not None and boundary_ratio_map.size > 0:
            axes[1, 1].imshow(boundary_ratio_map, cmap='plasma', alpha=0.7, vmin=0, vmax=1)
        axes[1, 1].set_title('6. Boundary Specialist Output')
        axes[1, 1].axis('off')
        if pre_refinement_mask is not None and pre_refinement_mask.size > 0:
            axes[1, 2].imshow(pre_refinement_mask, cmap='gray')
        axes[1, 2].set_title('7. Aggregated Mask (Pre-Refinement)')
        axes[1, 2].axis('off')
        if final_refined_mask is not None and final_refined_mask.size > 0:
            axes[1, 3].imshow(final_refined_mask, cmap='gray')
        axes[1, 3].set_title('8. Final Refined Mask')
        axes[1, 3].axis('off')
       
        plt.tight_layout(rect=[0, 0, 1, 0.96])
       
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
       
        # Save with error handling
        output_path = os.path.join(output_dir, f'slice_{slice_idx}.png')
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
       
    except Exception as e:
        print(f"Warning: Failed to generate plot for slice {slice_idx}: {e}")
        plt.close('all') # Clean up any open figures
   
    finally:
        gc.collect()
def apply_crf_refinement(image_slice, initial_mask, probability_maps=None, edge_weight=0.3, smoothness_weight=0.1):
    """
    Apply Conservative CRF refinement - much more conservative to preserve accuracy
    Only apply light smoothing without changing overall structure
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    except ImportError:
        print("Warning: pydensecrf not available, skipping CRF refinement")
        return np.ascontiguousarray(initial_mask.astype(np.uint8))
   
    # If initial mask is empty, return as-is (prevent false positives)
    if np.sum(initial_mask) == 0:
        return np.ascontiguousarray(initial_mask.astype(np.uint8))
   
    # Ensure all inputs are C-contiguous
    image_slice = np.ascontiguousarray(image_slice.astype(np.float32))
    initial_mask = np.ascontiguousarray(initial_mask.astype(np.uint8))
   
    h, w = image_slice.shape
    n_labels = 2 # background and brain
   
    # Create very conservative probability maps
    prob_brain = initial_mask.astype(np.float32)
    prob_bg = 1.0 - prob_brain
   
    # Be much more conservative - trust the initial prediction more
    confidence = 0.95 # Higher confidence in initial prediction
    prob_brain = np.clip(prob_brain * confidence + (1-confidence) * 0.5, 0.05, 0.95)
    prob_bg = 1.0 - prob_brain
   
    # Ensure C-contiguity
    prob_brain = np.ascontiguousarray(prob_brain)
    prob_bg = np.ascontiguousarray(prob_bg)
   
    # Stack probabilities and ensure C-contiguity
    probs = np.ascontiguousarray(np.stack([prob_bg, prob_brain], axis=2))
    probs = np.ascontiguousarray(probs.transpose(2, 0, 1).reshape(n_labels, -1))
   
    # Ensure probabilities sum to 1
    probs = probs / (probs.sum(axis=0, keepdims=True) + 1e-8)
    probs = np.ascontiguousarray(probs.astype(np.float32))
   
    # Create CRF
    d = dcrf.DenseCRF2D(w, h, n_labels)
   
    # Set unary potentials with higher weight (trust initial prediction more)
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
   
    # Much weaker pairwise potentials to avoid over-smoothing
    # Gaussian pairwise potential (smoothness) - very weak
    pairwise_gaussian = create_pairwise_gaussian(sdims=(2, 2), shape=(h, w))
    d.addPairwiseEnergy(pairwise_gaussian, compat=smoothness_weight * 3) # Reduced from 10 to 3
   
    # Bilateral pairwise potential (edge-aware) - very weak
    if len(image_slice.shape) == 2:
        # Convert grayscale to RGB for bilateral
        image_rgb = np.ascontiguousarray(np.stack([image_slice, image_slice, image_slice], axis=2))
    else:
        image_rgb = image_slice
   
    # Ensure C-contiguity and normalize image to [0, 255]
    image_rgb = np.ascontiguousarray(image_rgb)
    image_norm = ((image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min() + 1e-8) * 255).astype(np.uint8)
    image_norm = np.ascontiguousarray(image_norm)
   
    pairwise_bilateral = create_pairwise_bilateral(sdims=(3, 3), schan=(0.5,), img=image_norm, chdim=2)
    d.addPairwiseEnergy(pairwise_bilateral, compat=edge_weight * 3) # Reduced from 10 to 3
   
    # Fewer inference iterations to prevent over-smoothing
    Q = d.inference(3) # Reduced from 5 to 3 iterations
    MAP = np.argmax(Q, axis=0).reshape(h, w)
   
    return np.ascontiguousarray(MAP.astype(np.uint8))
def apply_surgical_grade_3d_consistency(volume_data, slice_range):
    """
    Apply surgical-grade 3D consistency with special handling for edge slices
    """
   
   
    print("Applying surgical-grade 3D processing...")
   
    # Ensure input is C-contiguous
    volume_data = np.ascontiguousarray(volume_data)
   
    # Step 1: Anisotropic smoothing (preserve slice boundaries but smooth within slices)
    volume_float = np.ascontiguousarray(volume_data.astype(np.float32))
    # Use standard smoothing for all slices
    smoothed_volume = np.ascontiguousarray(np.zeros_like(volume_float))
   
    for i in range(volume_data.shape[2]):
        current_slice = np.ascontiguousarray(volume_float[:, :, i])
        # Standard smoothing for all slices
        smoothed_slice = gaussian_filter(current_slice, sigma=1.0)
        smoothed_volume[:, :, i] = np.ascontiguousarray(smoothed_slice)
   
    # Step 2: Apply 3D bilateral filtering for edge-preserving smoothing
    print("Applying 3D bilateral filtering...")
    # Simulate bilateral filtering in 3D for all applicable slices
    for i in range(1, volume_data.shape[2] - 1):
        current = np.ascontiguousarray(smoothed_volume[:, :, i])
        prev_slice = np.ascontiguousarray(smoothed_volume[:, :, i - 1])
        next_slice = np.ascontiguousarray(smoothed_volume[:, :, i + 1])
       
        # Weighted average based on similarity
        weight_prev = np.exp(-((current - prev_slice) ** 2) / 0.1)
        weight_next = np.exp(-((current - next_slice) ** 2) / 0.1)
        weight_current = np.ones_like(current)
       
        total_weight = weight_prev + weight_next + weight_current
        result = (
            weight_current * current +
            weight_prev * prev_slice +
            weight_next * next_slice
        ) / total_weight
        smoothed_volume[:, :, i] = np.ascontiguousarray(result)
   
    # Step 3: Convert to binary with conservative thresholding to prevent false positives
    print("Applying conservative binary conversion...")
    binary_volume = np.ascontiguousarray(np.zeros_like(smoothed_volume, dtype=np.uint8))
   
    for i in range(volume_data.shape[2]):
        slice_data = np.ascontiguousarray(smoothed_volume[:, :, i])
       
        # Check if original slice had any brain tissue to prevent false positives
        original_has_brain = np.sum(volume_data[:, :, i] > 0.5) > 100 # Require at least 100 pixels
       
        if not original_has_brain:
            # If original slice had no brain, don't add any
            binary_volume[:, :, i] = np.zeros_like(slice_data, dtype=np.uint8)
        else:
            # Use a single conservative threshold for all slices
            threshold = 0.7
            binary_slice = (slice_data > threshold).astype(np.uint8)
            binary_volume[:, :, i] = np.ascontiguousarray(binary_slice)
   
    # Step 4: Conservative morphological operations - only clean up, don't add tissue
    print("Applying conservative morphological operations...")
    for i in range(volume_data.shape[2]):
        current_slice = np.ascontiguousarray(binary_volume[:, :, i])
       
        # Skip morphology if slice is empty to prevent false positives
        if np.sum(current_slice) == 0:
            continue
           
        # Apply the same conservative operations for all slices
        # Remove small isolated objects first
        current_slice = remove_small_objects(current_slice.astype(bool), min_size=50).astype(np.uint8)
       
        # Only if slice still has content, apply gentle closing
        if np.sum(current_slice) > 0:
           
            selem = disk(1)
            current_slice = binary_closing(current_slice, selem)
            # Remove holes that are reasonably small
            current_slice = remove_small_holes(current_slice.astype(bool), area_threshold=25).astype(np.uint8)
   
        binary_volume[:, :, i] = np.ascontiguousarray(current_slice)
   
    # Step 5: Edge-aware consistency enforcement
    print("Applying edge-aware consistency...")
    # m, n = slice_range
   
    for slice_idx in range(binary_volume.shape[2]):
        if slice_idx == 0 or slice_idx == binary_volume.shape[2] - 1:
            continue
           
        current_slice = binary_volume[:, :, slice_idx]
        prev_slice = binary_volume[:, :, slice_idx - 1]
        next_slice = binary_volume[:, :, slice_idx + 1]
       
        # Calculate consistency metrics
        prev_overlap = np.sum(current_slice & prev_slice) / (np.sum(current_slice | prev_slice) + 1e-8)
        next_overlap = np.sum(current_slice & next_slice) / (np.sum(current_slice | next_slice) + 1e-8)
       
        # Use a single, stricter consistency threshold for all slices
        consistency_threshold = 0.7
       
        if prev_overlap < consistency_threshold or next_overlap < consistency_threshold:
            # Apply corrective smoothing
            neighbor_avg = (prev_slice.astype(np.float32) + next_slice.astype(np.float32)) / 2.0
            current_smooth = gaussian_filter(current_slice.astype(np.float32), sigma=1.0)
           
            # Use standard blending for all slices
            blended = 0.6 * current_smooth + 0.4 * neighbor_avg
           
            binary_volume[:, :, slice_idx] = (blended > 0.5).astype(np.uint8)
   
    # Step 6: Final hole filling and cleanup
    print("Final cleanup...")
   
    for slice_idx in range(binary_volume.shape[2]):
        binary_volume[:, :, slice_idx] = binary_fill_holes(binary_volume[:, :, slice_idx]).astype(np.uint8)
   
    return binary_volume
def generate_combined_training_data_wrapper(args):
    """Wrapper for combined brain/boundary training data generation per slice."""
    (image_slice, image_slice_raw, brain_mask_slice, vague_mask_slice, coarse_params, fine_params,
     slice_idx, global_min, global_max, total_slices, GABOR_KERNELS_STACKED,
     SOBEL_KERNEL_X, SOBEL_KERNEL_Y, LAPLACIAN_KERNEL, is_known_slice, kernel, train_brain_on_all) = args
   
    return generate_combined_training_data(
        image_slice, image_slice_raw, brain_mask_slice, vague_mask_slice, coarse_params, fine_params,
        slice_idx, global_min, global_max, total_slices, GABOR_KERNELS_STACKED,
        SOBEL_KERNEL_X, SOBEL_KERNEL_Y, LAPLACIAN_KERNEL, is_known_slice, kernel, 
        use_boundary_specialist=True, train_brain_on_all=train_brain_on_all
    )
def gpu_memory_cleanup():
    """Simple GPU memory cleanup function"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except:
        pass
def normalize_features(X, mean, std):
    return (X - mean) / (std + 1e-8)
def precompute_lbp(image_slice, params):
    image_t = torch.tensor(image_slice, device=device).float()
    return local_binary_pattern_torch(image_t, params['lbp_radius'])
### -------------------------------------------------------------------------- ###
### --- START: ADD THE FOLLOWING 3 FUNCTIONS TO YOUR MAIN SCRIPT --- ###
### -------------------------------------------------------------------------- ###
### --- END: NEW FUNCTIONS TO ADD --- ###
def prepare_specialist_data_gpu(train_df, selected_features_name):
    """
    Performs feature selection and normalization statistics calculation on the GPU (CuPy).
    This eliminates a major CPU bottleneck (the ~1 min and 45 sec gaps).
    """
    metadata_cols = ['slice_idx', 'y', 'x', 'norm_x', 'norm_y', 'norm_slice', 'label', 'brain_ratio']
   
    # Identify initial feature columns
    feature_cols = [col for col in train_df.columns if col not in metadata_cols]
    if not feature_cols:
        return [], pd.Series(), pd.Series(), []
    # 1. Transfer raw features to GPU (CPU -> GPU)
    X_cpu = train_df[feature_cols].astype(np.float32)
    X_gpu = cp.array(X_cpu.values)
    # 2. GPU-Accelerated Redundancy Identification (CuPy correlation)
    start_time = time.time()
   
    # Calculate correlation matrix on GPU
    corr_matrix_gpu = cp.corrcoef(X_gpu, rowvar=False)
    corr_matrix_cpu = cp.asnumpy(corr_matrix_gpu)
   
    # Perform CPU-bound redundancy selection logic on the result
    corr_df = pd.DataFrame(corr_matrix_cpu, index=feature_cols, columns=feature_cols)
    upper_tri = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
    redundant_features = [column for column in upper_tri.columns if any(upper_tri[column].abs() > 0.9)]
   
    selected_features = [col for col in feature_cols if col not in redundant_features]
    print(f"[{selected_features_name}] GPU Prep Time (Feature ID & CuPy Corr): {time.time() - start_time:.2f} seconds. Dropped: {len(redundant_features)} out of {len(feature_cols)} total features")
    # 3. GPU-Accelerated Statistics Calculation
    # Filter GPU array to selected features only
    selected_indices = [feature_cols.index(f) for f in selected_features]
    X_selected_gpu = X_gpu[:, selected_indices]
   
    mean_gpu = X_selected_gpu.mean(axis=0)
    std_gpu = X_selected_gpu.std(axis=0)
    # 4. Transfer minimal results to CPU
    mean_data = pd.Series(cp.asnumpy(mean_gpu), index=selected_features)
    std_data = pd.Series(cp.asnumpy(std_gpu), index=selected_features)
   
    return selected_features, mean_data, std_data, redundant_features
def optimized_memory_cleanup():
    """Aggressive memory cleanup for long-running processes"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
def _create_gpu_disk(radius):
        """Custom function to create a disk structuring element on the GPU."""
        y, x = cp.ogrid[-radius:radius+1, -radius:radius+1]
        disk = x**2 + y**2 <= radius**2
        return disk.astype(cp.uint8)
def dice_coefficient_gpu(mask1, mask2):
        """Computes Dice on CuPy arrays."""
        intersection = cp.sum(mask1 & mask2)
        total_sum = cp.sum(mask1) + cp.sum(mask2)
        if total_sum == 0:
            return 1.0 # Return a float
        return 2. * intersection / (total_sum + 1e-8)
def _identify_abnormal_slices_gpu(volume_data_gpu, iqr_sensitivity_factor, dice_floor):
    """GPU version of abnormal slice identification."""
    total_slices = volume_data_gpu.shape[2]
   
    # Calculate areas on GPU and transfer to CPU
    areas = [cp.sum(volume_data_gpu[:, :, i]) for i in range(total_slices)]
    areas_cpu = np.array([float(area) for area in areas]) # Convert CuPy scalars to Python floats
   
    # Calculate dice scores on GPU
    dice_list = [1.0] # First slice has no previous slice, so set dice to 1.0
    for i in range(1, total_slices):
        dice = dice_coefficient_gpu(volume_data_gpu[:, :, i], volume_data_gpu[:, :, i - 1])
        dice_list.append(float(dice)) # Convert CuPy scalar to Python float
    dice_scores_cpu = np.array(dice_list)
   
    brain_slice_indices = [i for i, area in enumerate(areas_cpu) if area > 100]
    if not brain_slice_indices:
        return set()
    buffer = max(5, int(len(brain_slice_indices) * 0.1))
    core_brain_indices = brain_slice_indices[buffer:-buffer]
    if not core_brain_indices:
        dynamic_dice_thresh = 0.95
    else:
        core_dice_scores = [dice_scores_cpu[i] for i in core_brain_indices]
        q1 = np.percentile(core_dice_scores, 25)
        q3 = np.percentile(core_dice_scores, 75)
        iqr = q3 - q1
        dynamic_dice_thresh = max(dice_floor, q1 - iqr_sensitivity_factor * iqr)
   
    print(f" - Automatically determined Dice Threshold: {dynamic_dice_thresh:.4f}")
    abnormal_indices = {i for i in core_brain_indices if dice_scores_cpu[i] < dynamic_dice_thresh}
    return abnormal_indices
def _correct_abnormal_blocks_gpu(volume_data_gpu, abnormal_indices):
    """GPU version of abnormal block correction."""
    print("--- Correcting abnormal slice blocks (GPU)...")
    refined_volume_gpu = volume_data_gpu.copy()
    total_slices = volume_data_gpu.shape[2]
    sorted_indices = sorted(list(abnormal_indices))
   
    selem_disk_3_gpu = _create_gpu_disk(3)
    while sorted_indices:
        start_abnormal = sorted_indices[0]
        end_abnormal = start_abnormal
        while end_abnormal + 1 in sorted_indices:
            end_abnormal += 1
        block = list(range(start_abnormal, end_abnormal + 1))
        print(f" - Correcting block: slices {start_abnormal} to {end_abnormal}")
        sorted_indices = [i for i in sorted_indices if i not in block]
        temp_vol_left_gpu = refined_volume_gpu.copy()
        temp_vol_right_gpu = refined_volume_gpu.copy()
        # Left-to-right pass
        anchor_idx_left = start_abnormal - 1
        if anchor_idx_left >= 0:
            for i in block:
                potential_mask = cp.logical_or(temp_vol_left_gpu[:, :, i], temp_vol_left_gpu[:, :, anchor_idx_left])
                # EXPLICITLY use GPU functions
                search_area = cupy_binary_dilation(temp_vol_left_gpu[:, :, anchor_idx_left], structure=selem_disk_3_gpu)
                corrected = cp.logical_and(potential_mask, search_area)
                temp_vol_left_gpu[:, :, i] = cupy_binary_fill_holes(corrected).astype(cp.uint8)
                anchor_idx_left = i
        # Right-to-left pass
        anchor_idx_right = end_abnormal + 1
        if anchor_idx_right < total_slices:
            for i in reversed(block):
                potential_mask = cp.logical_or(temp_vol_right_gpu[:, :, i], temp_vol_right_gpu[:, :, anchor_idx_right])
                # EXPLICITLY use GPU functions
                search_area = cupy_binary_dilation(temp_vol_right_gpu[:, :, anchor_idx_right], structure=selem_disk_3_gpu)
                corrected = cp.logical_and(potential_mask, search_area)
                temp_vol_right_gpu[:, :, i] = cupy_binary_fill_holes(corrected).astype(cp.uint8)
                anchor_idx_right = i
       
        # Combine the results
        for i in block:
            final_mask = cp.logical_and(temp_vol_left_gpu[:, :, i], temp_vol_right_gpu[:, :, i])
            refined_volume_gpu[:, :, i] = final_mask.astype(cp.uint8)
   
    return refined_volume_gpu

def export_mask_to_dicom(original_dicom_dir: str, nifti_mask_path: str, output_dir: str):
    """
    Reads the original DICOMs and the final NIfTI mask, replacing the pixel data
    in the DICOMs with the mask data, correcting headers for 8-bit format.
    """
    print(f"\nSTARTING DICOM EXPORT: {nifti_mask_path} -> {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Original DICOMs & Sort
    dicoms = [pydicom.dcmread(os.path.join(original_dicom_dir, f)) 
              for f in os.listdir(original_dicom_dir) if f.lower().endswith('.dcm')]
    # Robust sorting by Instance Number (fall back to SliceLocation if needed)
    dicoms.sort(key=lambda x: x.InstanceNumber if hasattr(x, 'InstanceNumber') else x.SliceLocation)

    # 2. Load NIfTI Mask
    nifti_img = nib.load(nifti_mask_path)
    mask_data = nifti_img.get_fdata().astype(np.uint8)

    # Check dimensions
    if len(dicoms) != mask_data.shape[2]:
        print(f"⚠️ WARNING: Slice count mismatch! DICOM: {len(dicoms)}, NIfTI: {mask_data.shape[2]}")
    
    # 3. Overwrite and Save
    for i, dcm in enumerate(dicoms):
        if i >= mask_data.shape[2]: break
        
        # Extract the slice (Transpose to match DICOM orientation)
        mask_slice = mask_data[:, :, i].T 
        
        # Ensure binary mask is visible (0 -> 0, 1 -> 255)
        if mask_slice.max() <= 1:
            mask_slice = mask_slice * 255
            
        # --- CRITICAL FIX START ---
        # Explicitly convert to uint8
        mask_slice = mask_slice.astype(np.uint8)
        
        # Update Pixel Data
        dcm.PixelData = mask_slice.tobytes()
        dcm.Rows, dcm.Columns = mask_slice.shape
        
        # Update Metadata to reflect 8-bit data
        dcm.BitsAllocated = 8
        dcm.BitsStored = 8
        dcm.HighBit = 7
        dcm.PixelRepresentation = 0  # Unsigned integer
        dcm.SamplesPerPixel = 1
        dcm.PhotometricInterpretation = "MONOCHROME2" # 0=Black, 255=White
        
        # Remove tags that might conflict with new format
        if 'SmallestImagePixelValue' in dcm: del dcm.SmallestImagePixelValue
        if 'LargestImagePixelValue' in dcm: del dcm.LargestImagePixelValue
        if 'WindowCenter' in dcm: del dcm.WindowCenter
        if 'WindowWidth' in dcm: del dcm.WindowWidth
        # --- CRITICAL FIX END ---
        
        # Update Series Description
        dcm.SeriesDescription = f"Brain Mask"
        
        # Use original DICOM filename
        original_filename = os.path.basename(dcm.filename)
        out_path = os.path.join(output_dir, original_filename)
        dcm.save_as(out_path)

    print(f"✅ DICOM Export complete: {len(dicoms)} slices saved to {output_dir}")

def process_nifti(image_path, accurate_mask_path, vague_mask_path, slice_range, coarse_params, fine_params,
                  output_path, delta=0.5, save_files=True, depth=100, use_boundary_specialist=True ,brain_definitness=0.98, kernel=5, use_crf=False, predict_all=False, train_brain_on_all=True, xgb_clf_params=None, xgb_reg_params=None, use_auto_refinement=True, iqr_sensitivity_factor=1.5, use_parallel=True):
   
   
   
    total_start_time = time.time()
   
    # print(f"🚀 Starting NFBS processing with {'PARALLEL' if use_parallel else 'SEQUENTIAL'} mode")
    print(f"📊 GPU Configuration:")
    print(f" - Available GPUs: {torch.cuda.device_count()}")
    print(f" - Current GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name()})")
    print(f" - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # print(f"📊 Parallelization Features:")
    # print(f" - Model Training: {'✅ ENABLED' if use_parallel else '❌ DISABLED'}")
    # print(f" - Feature Extraction: {'✅ ENABLED' if use_parallel else '❌ DISABLED'}")
    # print(f" - Forward/Backward Prediction: {'🚀 TRUE PARALLEL' if use_parallel and torch.cuda.device_count() > 1 else '⚡ INTERLEAVED PARALLEL' if use_parallel else '❌ SEQUENTIAL'}")
    # print(f" - Slice Processing: {'✅ ENABLED' if use_parallel else '❌ DISABLED'}")
    # print(f" - Post-processing: {'✅ ENABLED' if use_parallel else '❌ DISABLED'}")
   
    GABOR_KERNELS = [torch.tensor(cv2.getGaborKernel((31, 31), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F),
                                  device=device).unsqueeze(0).unsqueeze(0)
                       for sigma, theta, lambd, gamma, psi in itertools.product(*GABOR_PARAMS.values())]
    GABOR_KERNELS_STACKED = torch.cat(GABOR_KERNELS, dim=0)
    SOBEL_KERNEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
    SOBEL_KERNEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
    LAPLACIAN_KERNEL = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                    dtype=torch.float32, device=device).view(1, 1, 3, 3)
   
    # Load data
    image_nifti = nib.load(image_path)
    image_data_raw = image_nifti.get_fdata()
    global_min, global_max = np.min(image_data_raw), np.max(image_data_raw)
    # Z-score normalization
    mean = np.mean(image_data_raw)
    std = np.std(image_data_raw)
    image_data = (image_data_raw - mean) / (std + 1e-8)
    accurate_nifti = nib.load(accurate_mask_path)
    accurate_data = accurate_nifti.get_fdata()
    vague_nifti = nib.load(vague_mask_path)
    vague_data = vague_nifti.get_fdata()
   
    # --- CHANGE START ---
    total_slices = image_data.shape[2]
    # Parse the new slice_range format to get sets of known and unknown slices
    known_slices_set, unknown_slices_list = parse_slice_ranges(slice_range, total_slices)
    new_data_gpu = cp.zeros_like(image_data_raw, dtype=cp.uint8)
   
    # Pre-fill new_data_gpu with all known accurate masks (now on GPU)
    for slice_idx in known_slices_set:
        if 0 <= slice_idx < total_slices:
            new_data_gpu[:, :, slice_idx] = cp.asarray(accurate_data[:, :, slice_idx])
   
    # # Pre-fill new_data with all known accurate masks
    # for slice_idx in known_slices_set:
    # if 0 <= slice_idx < total_slices:
    # new_data[:, :, slice_idx] = accurate_data[:, :, slice_idx]
    # --- CHANGE END ---
   
    # Create output directory (ensure it exists)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")
    else:
        print(f"📁 Output directory exists: {output_dir} \n")
   
    metadata_cols = ['slice_idx', 'y', 'x', 'norm_x', 'norm_y', 'norm_slice', 'label', 'brain_ratio']
    # Generate texture_feature_names (common for all)
    lbp_n_points_coarse = 8 * coarse_params['lbp_radius']
    texture_feature_names_c = get_texture_feature_names(lbp_n_points_coarse)
    lbp_n_points_fine = 8 * fine_params['lbp_radius']
    texture_feature_names_f = get_texture_feature_names(lbp_n_points_fine)
    # Step 2 & 3: Train both specialists with parallel data generation
    specialist_start_time = time.time()
   
    # specialist_start_time = time.time()
   
    brain_train_data_list = []
    boundary_train_data_list = []
   
    # Prepare combined arguments for all relevant slices
    combined_args = []
    for slice_idx in range(total_slices):
        is_known_slice = slice_idx in known_slices_set
        brain_mask_slice = accurate_data[:, :, slice_idx]
        vague_mask_slice = vague_data[:, :, slice_idx] if not is_known_slice else brain_mask_slice
       
        if np.any(brain_mask_slice):
            combined_args.append((
                image_data[:, :, slice_idx],
                image_data_raw[:, :, slice_idx],
                brain_mask_slice,
                vague_mask_slice,
                coarse_params,
                fine_params,
                slice_idx,
                global_min,
                global_max,
                total_slices,
                GABOR_KERNELS_STACKED,
                SOBEL_KERNEL_X,
                SOBEL_KERNEL_Y,
                LAPLACIAN_KERNEL,
                is_known_slice,
                kernel,
                train_brain_on_all
            ))
   
    print("Aggregating combined training data for Brain and Boundary Specialists...")
    start_time = time.time()
   
    if use_parallel:
        try:
            with ThreadPoolExecutor(max_workers=min(4, len(combined_args))) as executor:
                futures = [executor.submit(generate_combined_training_data_wrapper, args) for args in combined_args]
               
                for future in as_completed(futures):
                    brain_df, boundary_df = future.result()
                    if not brain_df.empty:
                        brain_train_data_list.append(brain_df)
                    if not boundary_df.empty:
                        boundary_train_data_list.append(boundary_df)
               
                executor.shutdown(wait=True)
        except Exception as e:
            print(f"❌ Error in parallel combined training: {e}")
            # Fallback to sequential
            for args in combined_args:
                brain_df, boundary_df = generate_combined_training_data_wrapper(args)
                if not brain_df.empty:
                    brain_train_data_list.append(brain_df)
                if not boundary_df.empty:
                    boundary_train_data_list.append(boundary_df)
    else:
        for args in combined_args:
            brain_df, boundary_df = generate_combined_training_data_wrapper(args)
            if not brain_df.empty:
                brain_train_data_list.append(brain_df)
            if not boundary_df.empty:
                boundary_train_data_list.append(boundary_df)
   
    brain_train_df = pd.concat(brain_train_data_list, ignore_index=True) if brain_train_data_list else pd.DataFrame()
    boundary_train_df = pd.concat(boundary_train_data_list, ignore_index=True) if boundary_train_data_list else pd.DataFrame()
   
    print(f"Combined training data generation time: {time.time() - start_time:.2f} seconds")
   
    # GPU Pre-Training Phase: Move CPU-heavy feature selection and stats calculation to GPU
    pre_train_start_time = time.time()
    print("🔬 Starting PARALLEL GPU Pre-Training (Feature ID & Stats Calculation)...")
    # Use ThreadPoolExecutor to parallelize the *preparation* of data for training
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            brain_prep_future = executor.submit(prepare_specialist_data_gpu, brain_train_df, 'Brain')
            if use_boundary_specialist and not boundary_train_df.empty:
                boundary_prep_future = executor.submit(prepare_specialist_data_gpu, boundary_train_df, 'Boundary')
            # Collect results
            selected_features_brain, mean_brain, std_brain, redundant_brain = brain_prep_future.result()
            if use_boundary_specialist and not boundary_train_df.empty:
                selected_features_boundary, mean_boundary, std_boundary, redundant_boundary = boundary_prep_future.result()
            else:
                selected_features_boundary, mean_boundary, std_boundary = [], pd.Series(), pd.Series()
               
            # Ensure proper cleanup
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"❌ Error in parallel GPU pre-training: {e}")
        # Fallback to sequential
        selected_features_brain, mean_brain, std_brain, redundant_brain = prepare_specialist_data_gpu(brain_train_df, 'Brain')
        if use_boundary_specialist and not boundary_train_df.empty:
            selected_features_boundary, mean_boundary, std_boundary, redundant_boundary = prepare_specialist_data_gpu(boundary_train_df, 'Boundary')
        else:
            selected_features_boundary, mean_boundary, std_boundary = [], pd.Series(), pd.Series()
    print(f"🔬 Total GPU Pre-Training Time: {time.time() - pre_train_start_time:.2f} seconds")
    # --- END PARALLEL PRE-TRAINING ---
   
    # Train Brain Specialist
    if brain_train_df.empty:
        raise ValueError("No training data for Brain Specialist.")
   
    # Use GPU-prepared feature selection and normalization stats
    X_brain = brain_train_df[selected_features_brain].astype(np.float32)
    X_brain_normalized = normalize_features(X_brain, mean_brain.values, std_brain.values)
    y_brain = brain_train_df['label'].astype(np.uint8)
    # Convert to CuPy for GPU XGBoost training
    X_brain_normalized_gpu = cp.array(X_brain_normalized)
    y_brain_gpu = cp.array(y_brain.values)
   
    print("Training Brain Specialist...")
    brain_clf_start_time = time.time()
    if xgb_clf_params is not None:
        Brain_clf = xgb.XGBClassifier(**xgb_clf_params)
    else:
        Brain_clf = xgb.XGBClassifier(
            n_estimators=depth,
            random_state=42,
            tree_method='hist',
            device='cuda',
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        )
    Brain_clf.fit(X_brain_normalized_gpu, y_brain_gpu)
    print(f"Brain Specialist training time: {time.time() - brain_clf_start_time:.2f} seconds")
   
    # Train Boundary Specialist
    Boundary_reg = None
    if use_boundary_specialist and not boundary_train_df.empty:
        # Use GPU-prepared feature selection and normalization stats
        X_boundary = boundary_train_df[selected_features_boundary].astype(np.float32)
        X_boundary_normalized = normalize_features(X_boundary, mean_boundary.values, std_boundary.values)
        y_boundary = boundary_train_df['brain_ratio'].astype(np.float32)
        # Convert to CuPy for GPU XGBoost training
        X_boundary_normalized_gpu = cp.array(X_boundary_normalized)
        y_boundary_gpu = cp.array(y_boundary.values)
       
        print("Training Boundary Specialist...")
        boundary_reg_start_time = time.time()
        if xgb_reg_params is not None:
            Boundary_reg = xgb.XGBRegressor(**xgb_reg_params)
        else:
            Boundary_reg = xgb.XGBRegressor(
                n_estimators=depth,
                random_state=42,
                tree_method='hist',
                device='cuda',
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8
            )
        Boundary_reg.fit(X_boundary_normalized_gpu, y_boundary_gpu)
        print(f"Boundary Specialist training time: {time.time() - boundary_reg_start_time:.2f} seconds")
    elif use_boundary_specialist:
        print("Warning: No training data for Boundary Specialist. Disabling it for this run.")
        use_boundary_specialist = False
   
    print(f"Total specialist training time: {time.time() - specialist_start_time:.2f} seconds")
    def process_testing_slice_batch(slices_to_process, anchor_info, use_boundary_specialist,
                               brain_definitness, vague_data, use_crf=False):
        """
        Process multiple slices in batch for better GPU utilization
        """
        #
       
        batch_start_time = time.time()
        results = {}
       
        for slice_to_predict, anchor_mask, anchor_slice_idx in anchor_info:
            slice_start_time = time.time()
           
            # Image preparation
            image_slice_pred_raw = image_data_raw[:, :, slice_to_predict]
            image_slice_raw_anchor = image_data_raw[:, :, anchor_slice_idx]
            matched_slice_pred_raw = match_histograms(image_slice_pred_raw, image_slice_raw_anchor, channel_axis=None)
           
            mean_pred = np.mean(matched_slice_pred_raw)
            std_pred = np.std(matched_slice_pred_raw)
            image_slice_pred_zscored = (matched_slice_pred_raw - mean_pred) / (std_pred + 1e-8)
            # Use GPU version for head mask extraction
            image_slice_pred_raw_gpu = cp.asarray(image_slice_pred_raw)
            head_mask_gpu = extract_head_mask_gpu(image_slice_pred_raw_gpu)
            head_mask = cp.asnumpy(head_mask_gpu)
            # Create working ROI
            vague_mask_slice = (vague_data[:, :, slice_to_predict] > 0).astype(np.uint8)
            dilation_kernel = disk(5)
            dilated_vague_mask = binary_dilation(vague_mask_slice, dilation_kernel)
            working_roi = (dilated_vague_mask & head_mask).astype(np.uint8)
            if np.sum(working_roi) == 0:
                results[slice_to_predict] = np.zeros_like(image_slice_pred_zscored, dtype=np.uint8)
                continue
               
            # Feature extraction with optimized function
            brain_features_df = extract_features_mixed_optimized(
                image_slice_pred_zscored, matched_slice_pred_raw, working_roi, coarse_params,
                slice_to_predict, global_min, global_max, total_slices, GABOR_KERNELS_STACKED,
                SOBEL_KERNEL_X, SOBEL_KERNEL_Y, LAPLACIAN_KERNEL, return_df=True
            )
           
            if brain_features_df is None or brain_features_df.empty:
                results[slice_to_predict] = np.zeros_like(image_slice_pred_zscored, dtype=np.uint8)
                continue
            # Prediction
            brain_patch_positions = brain_features_df[['y', 'x']].values.tolist()
            X_brain_test = brain_features_df[selected_features_brain].astype(np.float32)
            X_brain_normalized = normalize_features(X_brain_test, mean_brain.values, std_brain.values)
            X_brain_normalized_gpu = cp.array(X_brain_normalized)
            brain_probas = Brain_clf.predict_proba(X_brain_normalized_gpu)
            p_b = cp.asnumpy(brain_probas[:, 2])
            p_b_map = prob_voting(p_b, brain_patch_positions, image_slice_pred_zscored.shape, coarse_params['patch_size'])
            # Post-processing
            definite_brain_mask = (p_b_map > brain_definitness) & working_roi.astype(bool)
            uncertain_zone = working_roi.copy()
            uncertain_zone[definite_brain_mask] = 0
            pre_refinement_mask = definite_brain_mask.copy().astype(np.uint8)
           
            if np.sum(uncertain_zone) > 0 and use_boundary_specialist and Boundary_reg is not None:
                fine_features_df = extract_features_mixed_optimized(
                    image_slice_pred_zscored, matched_slice_pred_raw, uncertain_zone, fine_params,
                    slice_to_predict, global_min, global_max, total_slices, GABOR_KERNELS_STACKED,
                    SOBEL_KERNEL_X, SOBEL_KERNEL_Y, LAPLACIAN_KERNEL, return_df=True
                )
               
                if fine_features_df is not None and not fine_features_df.empty:
                    X_fine_test = fine_features_df[selected_features_boundary].astype(np.float32)
                    X_fine_normalized = normalize_features(X_fine_test, mean_boundary.values, std_boundary.values)
                    X_fine_normalized_gpu = cp.array(X_fine_normalized)
                    boundary_ratios_gpu = Boundary_reg.predict(X_fine_normalized_gpu)
                    boundary_ratios = cp.asnumpy(boundary_ratios_gpu)
                    fine_patch_positions = fine_features_df[['y', 'x']].values.tolist()
                    boundary_ratio_map = prob_voting(boundary_ratios, fine_patch_positions,
                                                image_slice_pred_zscored.shape, fine_params['patch_size'])
                   
                    pre_refinement_mask[uncertain_zone > 0] = (boundary_ratio_map[uncertain_zone > 0] > 0.5)
            # Final refinement
            pre_refinement_mask_gpu = cp.asarray(pre_refinement_mask)
            anchor_mask_gpu = cp.asarray(anchor_mask) # If anchor_mask is numpy
            final_refined_mask_gpu = refine_brain_mask_contextual_gpu(pre_refinement_mask_gpu, anchor_mask_gpu)
            final_refined_mask_gpu = apply_temporal_consistency_gpu(final_refined_mask_gpu, anchor_mask_gpu, alpha=0.2)
            final_refined_mask = final_refined_mask_gpu.get() # To CPU for results dict
            results[slice_to_predict] = final_refined_mask
           
            print(f"Predicting slice {slice_to_predict} took {time.time() - slice_start_time:.2f} seconds")
       
        print(f"Batch of {len(slices_to_process)} slices took {time.time() - batch_start_time:.2f} seconds")
        return results
    # ... inside process_nifti function ...
    def predict_slice_without_refinement(slice_to_predict, matched_slice_pred_raw):
        """
        Performs model prediction on a PRE-MATCHED slice without any refinement.
        Returns the raw, pre-refinement mask and data for plotting.
        """
       
        # --- 1. Image Preparation (Histogram matching is now DONE OUTSIDE) ---
        mean_pred = np.mean(matched_slice_pred_raw)
        std_pred = np.std(matched_slice_pred_raw)
        image_slice_pred_zscored = (matched_slice_pred_raw - mean_pred) / (std_pred + 1e-8)
       
        # The rest of the function remains largely the same...
        # Use GPU version for head mask extraction
        matched_slice_pred_raw_gpu = cp.asarray(matched_slice_pred_raw)
        head_mask_gpu = extract_head_mask_gpu(matched_slice_pred_raw_gpu)
        head_mask = cp.asnumpy(head_mask_gpu)
        vague_mask_slice = (vague_data[:, :, slice_to_predict] > 0).astype(np.uint8)
        dilated_vague_mask = binary_dilation(vague_mask_slice, disk(5))
        working_roi = (dilated_vague_mask & head_mask).astype(np.uint8)
        if np.sum(working_roi) == 0:
            return np.zeros_like(working_roi, dtype=np.uint8), None
        # --- 2. Feature Extraction and Brain Specialist Prediction ---
        brain_features_df = extract_features_mixed_optimized(
            image_slice_pred_zscored, matched_slice_pred_raw, working_roi, coarse_params,
            slice_to_predict, global_min, global_max, total_slices, GABOR_KERNELS_STACKED,
            SOBEL_KERNEL_X, SOBEL_KERNEL_Y, LAPLACIAN_KERNEL, return_df=True
        )
       
        if brain_features_df is None or brain_features_df.empty:
            return np.zeros_like(working_roi, dtype=np.uint8), None
        # ... (the rest of the prediction logic for Brain and Boundary specialists is identical) ...
        brain_patch_positions = brain_features_df[['y', 'x']].values.tolist()
        X_brain_test = brain_features_df[selected_features_brain].astype(np.float32)
        X_brain_normalized = normalize_features(X_brain_test, mean_brain.values, std_brain.values)
        X_brain_normalized_gpu = cp.array(X_brain_normalized)
        brain_probas = Brain_clf.predict_proba(X_brain_normalized_gpu)
        p_b = cp.asnumpy(brain_probas[:, 2])
        p_b_map = prob_voting(p_b, brain_patch_positions, image_slice_pred_zscored.shape, coarse_params['patch_size'])
        definite_brain_mask = (p_b_map > brain_definitness) & working_roi.astype(bool)
        uncertain_zone = working_roi.copy()
        uncertain_zone[definite_brain_mask] = 0
        pre_refinement_mask = definite_brain_mask.copy().astype(np.uint8)
        boundary_ratio_map = np.zeros_like(p_b_map)
        if np.sum(uncertain_zone) > 0 and use_boundary_specialist and Boundary_reg is not None:
            fine_features_df = extract_features_mixed_optimized(
                image_slice_pred_zscored, matched_slice_pred_raw, uncertain_zone, fine_params,
                slice_to_predict, global_min, global_max, total_slices, GABOR_KERNELS_STACKED,
                SOBEL_KERNEL_X, SOBEL_KERNEL_Y, LAPLACIAN_KERNEL, return_df=True
            )
            if fine_features_df is not None and not fine_features_df.empty:
                X_fine_test = fine_features_df[selected_features_boundary].astype(np.float32)
                X_fine_normalized = normalize_features(X_fine_test, mean_boundary.values, std_boundary.values)
                X_fine_normalized_gpu = cp.array(X_fine_normalized)
                boundary_ratios_gpu = Boundary_reg.predict(X_fine_normalized_gpu)
                boundary_ratios = cp.asnumpy(boundary_ratios_gpu)
                fine_patch_positions = fine_features_df[['y', 'x']].values.tolist()
                boundary_ratio_map = prob_voting(boundary_ratios, fine_patch_positions, image_slice_pred_zscored.shape, fine_params['patch_size'])
                pre_refinement_mask[uncertain_zone > 0] = (boundary_ratio_map[uncertain_zone > 0] > 0.5)
        plot_data = {
            'image_slice_pred_raw': matched_slice_pred_raw, 'working_roi': working_roi, 'p_b_map': p_b_map,
            'uncertain_zone': uncertain_zone, 'boundary_ratio_map': boundary_ratio_map,
            'pre_refinement_mask': pre_refinement_mask.copy()
        }
       
        return pre_refinement_mask, plot_data
    print("--- Initializing new prediction strategy ---")
    # --- Start of NEW logic to find the best anchor ---
    print("--- Finding best anchor slice based on mask area ---")
    max_area = -1
    best_anchor_item = None
    # We need to iterate through the original slice_range structure
    for item in slice_range:
        current_item_slices = []
        if isinstance(item, int):
            current_item_slices.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            current_item_slices.extend(range(item[0], item[1] + 1))
       
        for slice_idx in current_item_slices:
            if 0 <= slice_idx < total_slices:
                mask = accurate_data[:, :, slice_idx]
                area = np.sum(mask)
                if area > max_area:
                    max_area = area
                    best_anchor_item = item # Store the original item, e.g., 55 or [50, 60]
    if best_anchor_item is None:
        raise ValueError("Cannot find a valid anchor slice with a brain mask in the provided slice_range.")
    # Determine start and end anchors from the best item found
    if isinstance(best_anchor_item, int):
        start_anchor = best_anchor_item
        end_anchor = best_anchor_item
    else: # It's a list or tuple
        start_anchor = best_anchor_item[0]
        end_anchor = best_anchor_item[1]
    print(f"✅ Identified best anchor block by area: Slices {start_anchor} to {end_anchor} (Area: {max_area}).")
    # --- End of NEW logic ---
    if predict_all:
        print(f"\n--- PREDICT ALL MODE: Starting from anchor slice {start_anchor} ---")
        def _match_slice(slice_to_match_idx, anchor_slice_idx):
          """Helper function to match a single slice's histogram."""
          slice_to_match_raw = image_data_raw[:, :, slice_to_match_idx]
          anchor_slice_raw = image_data_raw[:, :, anchor_slice_idx]
          return match_histograms(slice_to_match_raw, anchor_slice_raw, channel_axis=None)
        def _run_3_stage_pipeline(initial_anchor_idx, slice_order):
            """Executes the 3-stage pipeline entirely on GPU where possible."""
            if not slice_order:
                return
            # STAGE 1 & 2 remain the same...
            # ... (code for Stage 1 and Stage 2) ...
            print("Stage 1: Starting parallel histogram matching...")
            stage1_start = time.time()
            matched_images = {}
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_to_slice = {executor.submit(_match_slice, slice_idx, anchor_idx): slice_idx for slice_idx, anchor_idx in slice_order}
                for future in as_completed(future_to_slice):
                    slice_idx = future_to_slice[future]
                    matched_images[slice_idx] = future.result()
            print(f"Stage 1 finished in {time.time() - stage1_start:.2f} seconds.")
            print("Stage 2: Starting fully parallel prediction...")
            stage2_start = time.time()
            raw_predictions = {}
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_to_slice = {
                    executor.submit(predict_slice_without_refinement, slice_idx, matched_images[slice_idx]): slice_idx
                    for slice_idx, _ in slice_order
                }
                for future in as_completed(future_to_slice):
                    slice_idx = future_to_slice[future]
                    try:
                        raw_predictions[slice_idx] = future.result()
                    except Exception as e:
                        print(f"Prediction for slice {slice_idx} failed: {e}")
                        raw_predictions[slice_idx] = (np.zeros_like(image_data_raw[:,:,0]), None)
            print(f"Stage 2 finished in {time.time() - stage2_start:.2f} seconds.")
            # --- STAGE 3: GPU-ACCELERATED SEQUENTIAL REFINEMENT ---
            print("Stage 3: Starting GPU-accelerated sequential refinement...")
            stage3_start = time.time()
            current_anchor_idx = initial_anchor_idx
            for slice_idx, _ in slice_order:
                pre_refinement_mask_cpu, plot_data = raw_predictions.get(slice_idx)
                if pre_refinement_mask_cpu is None: continue
                pre_refinement_mask_gpu = cp.asarray(pre_refinement_mask_cpu)
               
                anchor_mask_gpu = new_data_gpu[:, :, current_anchor_idx]
               
                final_refined_mask_gpu = refine_brain_mask_contextual_gpu(pre_refinement_mask_gpu, anchor_mask_gpu)
               
                final_refined_mask_gpu = apply_temporal_consistency_gpu(final_refined_mask_gpu, anchor_mask_gpu, alpha=0.2)
               
                new_data_gpu[:, :, slice_idx] = final_refined_mask_gpu
               
                if save_files and plot_data:
                    # For plotting, get to CPU only here
                    final_refined_mask_cpu = final_refined_mask_gpu.get()
                    generate_plots(slice_idx, plot_data["image_slice_pred_raw"], plot_data["working_roi"],
                                plot_data["p_b_map"], plot_data["uncertain_zone"],
                                plot_data["boundary_ratio_map"], plot_data["pre_refinement_mask"],
                                final_refined_mask_cpu, output_dir)
               
                current_anchor_idx = slice_idx
            optimized_memory_cleanup()
            print(f"Stage 3 finished in {time.time() - stage3_start:.2f} seconds.")
       
        def predict_backward_parallel():
          """Sets up and runs the backward pipeline."""
          print("\n--- Phase 1: Processing backwards from anchor to slice 0 ---")
          slice_order = []
          anchor = start_anchor
          for i in range(start_anchor - 1, -1, -1):
              slice_order.append((i, anchor))
              anchor = i
          _run_3_stage_pipeline(start_anchor, slice_order)
        def predict_forward_parallel():
            """Sets up and runs the forward pipeline."""
            print("\n--- Phase 2: Processing forwards from anchor to end ---")
            slice_order = []
            anchor = end_anchor
            for i in range(end_anchor + 1, total_slices):
                slice_order.append((i, anchor))
                anchor = i
            _run_3_stage_pipeline(end_anchor, slice_order)
           
        # PURE PARALLEL EXECUTION: Run forward and backward prediction simultaneously
        prediction_start_time = time.time()
       
        if use_parallel:
            print("🚀 Running TRUE PARALLEL forward/backward prediction")
            try:
                # Use threading for true parallel execution of independent tasks
                import threading
               
                # Create threads for forward and backward prediction
                backward_thread = threading.Thread(target=predict_backward_parallel)
                forward_thread = threading.Thread(target=predict_forward_parallel)
               
                # Start both threads
                backward_thread.start()
                forward_thread.start()
               
                # Wait for both to complete
                backward_thread.join()
                forward_thread.join()
               
                print("✅ Both forward and backward predictions completed in parallel")
               
            except Exception as e:
                print(f"❌ Error in parallel prediction: {e}")
                print("🔄 Falling back to sequential execution...")
                predict_backward_parallel()
                predict_forward_parallel()
        else:
            print("⚡ Running OPTIMIZED SEQUENTIAL forward/backward prediction")
            predict_backward_parallel()
            predict_forward_parallel()
       
        print(f"🏁 Total prediction time: {time.time() - prediction_start_time:.2f} seconds")
           
    else:
        # PARALLEL PROCESSING FOR UNKNOWN SLICES ONLY
        def predict_unknown_backward():
          """Sets up and runs the backward pipeline for UNKNOWN slices only."""
          print("\n--- Phase 1: Processing unknown slices backwards from anchor ---")
         
          # This list will store tuples of (slice_to_predict, anchor_for_matching)
          slice_order = []
         
          current_anchor_idx = start_anchor
          slice_idx = start_anchor - 1
         
          # Iterate backwards and build the list of tasks
          while slice_idx >= 0:
              if slice_idx in known_slices_set:
                  # If we hit a known range, jump the anchor to the beginning of that range
                  # and continue iterating from the slice before it.
                  print(f"➡️ Encountered known slice {slice_idx}. Resetting anchor.")
                  new_anchor = slice_idx
                  while new_anchor - 1 in known_slices_set:
                      new_anchor -= 1
                  current_anchor_idx = new_anchor
                  slice_idx = new_anchor - 1
                  continue # Skip to the next iteration of the while loop
              # This is an unknown slice, so add it to our processing list
              slice_order.append((slice_idx, current_anchor_idx))
             
              # The current slice becomes the anchor for the next one in the sequence
              current_anchor_idx = slice_idx
              slice_idx -= 1
             
          # Execute the 3-stage pipeline with the generated slice order
          _run_3_stage_pipeline(start_anchor, slice_order)
        def predict_unknown_forward():
            """Sets up and runs the forward pipeline for UNKNOWN slices only."""
            print("\n--- Phase 2: Processing unknown slices forwards from anchor ---")
           
            slice_order = []
           
            current_anchor_idx = end_anchor
            slice_idx = end_anchor + 1
           
            while slice_idx < total_slices:
                if slice_idx in known_slices_set:
                    # Jump the anchor to the end of the known range
                    print(f"➡️ Encountered known slice {slice_idx}. Resetting anchor.")
                    new_anchor = slice_idx
                    while new_anchor + 1 in known_slices_set:
                        new_anchor += 1
                    current_anchor_idx = new_anchor
                    slice_idx = new_anchor + 1
                    continue
                slice_order.append((slice_idx, current_anchor_idx))
               
                current_anchor_idx = slice_idx
                slice_idx += 1
               
            # Execute the 3-stage pipeline with the generated slice order
            _run_3_stage_pipeline(end_anchor, slice_order)
            # SIMPLIFIED PARALLEL EXECUTION FOR UNKNOWN SLICES
            prediction_start_time = time.time()
           
            if use_parallel:
                print("🚀 Running TRUE PARALLEL unknown slice prediction")
                try:
                    # Use threading for true parallel execution of independent tasks
                    import threading
                   
                    # Create threads for forward and backward prediction
                    backward_thread = threading.Thread(target=predict_unknown_backward)
                    forward_thread = threading.Thread(target=predict_unknown_forward)
                   
                    # Start both threads
                    backward_thread.start()
                    forward_thread.start()
                   
                    # Wait for both to complete
                    backward_thread.join()
                    forward_thread.join()
                   
                    print("✅ Both forward and backward unknown slice predictions completed in parallel")
                   
                except Exception as e:
                    print(f"❌ Error in parallel unknown slice prediction: {e}")
                    print("🔄 Falling back to sequential execution...")
                    predict_unknown_backward()
                    predict_unknown_forward()
            else:
                print("⚡ Running OPTIMIZED SEQUENTIAL unknown slice prediction")
                predict_unknown_backward()
                predict_unknown_forward()
           
        print(f"🏁 Total unknown slice prediction time: {time.time() - prediction_start_time:.2f} seconds")
    # Save final result
    # ... (at the end of process_nifti)
    # --- START: MODIFIED FINAL SAVING LOGIC ---
    # Step 5: Apply the final automated consistency refinement
    if use_auto_refinement:
        print("\n--- Applying Final Automated Consistency Refinement (GPU) ---")
        abnormal_indices = _identify_abnormal_slices_gpu(
            volume_data_gpu=new_data_gpu,
            iqr_sensitivity_factor=iqr_sensitivity_factor,
            dice_floor=0.90
        )
        if abnormal_indices:
            new_data_gpu = _correct_abnormal_blocks_gpu(new_data_gpu, abnormal_indices)
        else:
            print("--- No significant inconsistencies found. Skipping correction. ---")
           
    if use_crf:
        print("--- Applying surgical-grade 3D volume consistency (GPU) ---")
        new_data_gpu = apply_surgical_grade_3d_consistency_gpu(new_data_gpu, slice_range)
    print(f"💾 Saving final NIfTI file to: {output_path}")
    final_data_volume = new_data_gpu.get() # Final transfer to CPU for save
    final_nifti = nib.Nifti1Image(final_data_volume.astype(np.uint8), image_nifti.affine, image_nifti.header)
    nib.save(final_nifti, output_path)
    print(f"✅ Final mask saved successfully.")
    # --- END: MODIFIED FINAL SAVING LOGIC ---
def process_testing_slice_parallel(slice_to_predict, anchor_mask, anchor_slice_idx, use_boundary_specialist, brain_definitness, vague_data, use_crf=False):
    """Wrapper to maintain compatibility - uses optimized version"""
    return process_testing_slice(slice_to_predict, anchor_mask, anchor_slice_idx, use_boundary_specialist, brain_definitness, vague_data, use_crf)

def run_ml_refinement(image_path, mask_path, slice_range, output_dir, params, base_name='output', xgb_clf_params=None, xgb_reg_params=None):
    """
    Wrapper to call Code 1's process_nifti logic.
    """
    # Unpack params for readability
    coarse = params['coarse']
    fine = params['fine']
    depth = params['rf_depth']
    brain_def = params['brain_definitness']
    use_crf = params['use_crf']
    predict_all = params['predict_all']
    train_all = params['train_brain_on_all']
    parallel = params['use_parallel']
    save_files = params.get('save_files', False)
    
    # Use _pred_brain_mask.nii.gz naming convention
    output_filename = os.path.join(output_dir, f'{base_name}_pred_brain_mask.nii.gz')
    
    print(f"🚀 Starting ML Refinement for {os.path.basename(image_path)}")
    print(f"   Slice Range: {slice_range}")
    
    # Call the main Code 1 function
    # Note: Ensure arguments match EXACTLY with your process_nifti definition
    process_nifti(
        image_path=image_path,
        accurate_mask_path=mask_path, # Using registration mask as accurate
        vague_mask_path=mask_path,    # Using registration mask as vague
        slice_range=slice_range,
        coarse_params=coarse,
        fine_params=fine,
        output_path=output_filename,
        delta=0,
        save_files=save_files,
        depth=depth,
        use_boundary_specialist=True,
        brain_definitness=brain_def,
        kernel=params.get('kernel', 5),
        use_crf=use_crf,
        predict_all=predict_all,
        train_brain_on_all=train_all,
        use_parallel=parallel,
        use_auto_refinement=True, # Defaulting to True
        iqr_sensitivity_factor=params.get('iqr_sensitivity_factor', 1.5),
        xgb_clf_params=xgb_clf_params,
        xgb_reg_params=xgb_reg_params
    )
    
    return output_filename


def run_surgical_pipeline(input_target, base_output_dir, pipeline_params, xgb_clf_params=None, xgb_reg_params=None):
    start_time = time.time()
    
    # Extract patient ID (sanitize folder name)
    if os.path.isdir(input_target):
        # DICOM input: use folder name
        patient_id = os.path.basename(os.path.normpath(input_target)).replace(' ', '_')
    else:
        # NIfTI input: parse from path or use filename
        parts = input_target.split('/')
        patient_id = parts[-2] if len(parts) > 1 else os.path.basename(input_target).split('_')[0]
        patient_id = patient_id.replace(' ', '_')
    
    # --- STEP 1: PREPROCESSING (to temporary location) ---
    print("\n" + "="*50)
    print("STEP 1: PREPROCESSING & REGISTRATION")
    print("="*50)
    
    # Create temporary preprocessing directory
    import tempfile
    temp_preprocessing_dir = tempfile.mkdtemp(prefix="preprocessing_")
    
    template_path = pipeline_params.get('template_path', 'mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin.nii')
    template_mask = pipeline_params.get('template_mask', 'mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin_mask.nii')
    remove_scout = pipeline_params.get('remove_scout', False)
    
    # Run preprocessing to get detected_ranges
    detected_ranges, source_nifti, reg_mask_path = preprocessing(
        input_path=input_target,
        template_path=template_path,
        template_mask_path=template_mask,
        output_dir=temp_preprocessing_dir,
        remove_scout=remove_scout
    )
    
    if not detected_ranges:
        print("❌ CRITICAL ERROR: No valid brain slice ranges detected. Aborting.")
        shutil.rmtree(temp_preprocessing_dir, ignore_errors=True)
        return

    # --- STEP 2: CONSTRUCT FINAL OUTPUT DIRECTORY ---
    print("\n" + "="*50)
    print("STEP 2: CONSTRUCTING OUTPUT DIRECTORY STRUCTURE")
    print("="*50)
    
    # Extract parameters
    coarse = pipeline_params['coarse']
    fine = pipeline_params['fine']
    brain_def = pipeline_params['brain_definitness']
    use_crf = pipeline_params['use_crf']
    predict_all = pipeline_params['predict_all']
    train_all = pipeline_params['train_brain_on_all']
    parallel = pipeline_params['use_parallel']
    depth = pipeline_params['rf_depth']
    kernel = pipeline_params.get('kernel', 5)
    sensitivity = pipeline_params.get('iqr_sensitivity_factor', 1.5)
    
    # Build slice range string
    range_str_parts = []
    for item in detected_ranges:
        if isinstance(item, int):
            range_str_parts.append(str(item))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            range_str_parts.append(f"{item[0]}-{item[1]}")
    range_str = "_".join(range_str_parts)
    
    # Build parameter strings
    coarse_str = f"{coarse['patch_size'][0]},{coarse['stride'][0]},{coarse['lbp_radius']},{coarse['glcm_levels']},{coarse['valid_patch_threshold']}"
    fine_str = f"{fine['patch_size'][0]},{fine['stride'][0]},{fine['lbp_radius']},{fine['glcm_levels']},{fine['valid_patch_threshold']}"
    
    # Construct the FULL parameter string as a single folder name
    param_folder_name = (
        f"Param_slice_range{range_str}_BoundaryKernel_{kernel}_"
        f"Param_C_[{coarse_str}]_F[{fine_str}]_B[{brain_def}]_"
        f"predict_all_[{predict_all}]_train_brain_on_all_[{train_all}]_"
        f"CRF_[{use_crf}]_Parallel_[{parallel}]_Depth_[{depth}]_Sensitivity_[{sensitivity}]"
    )
    
    # Define Top-Level Directory and Final Output Directory
    top_level_dir = os.path.join(base_output_dir, f"{patient_id}_pred_brain_mask")
    final_output_dir = os.path.join(top_level_dir, param_folder_name)
    
    # Create the final output directory
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"📁 Final output directory: {final_output_dir}")
    
    # --- STEP 3: MOVE PREPROCESSING OUTPUTS ---
    print("\n" + "="*50)
    print("STEP 3: ORGANIZING PREPROCESSING OUTPUTS")
    print("="*50)
    
    # Define target paths in final output directory
    final_source_nifti = os.path.join(final_output_dir, os.path.basename(source_nifti))
    final_reg_mask = os.path.join(final_output_dir, os.path.basename(reg_mask_path))
    
    # Copy files from temp directory to final directory (preserve originals)
    shutil.copy(source_nifti, final_source_nifti)
    shutil.copy(reg_mask_path, final_reg_mask)
    print(f"✅ Copied {os.path.basename(source_nifti)} to {final_output_dir}")
    print(f"✅ Copied {os.path.basename(reg_mask_path)} to {final_output_dir}")
    
    # Clean up temporary directory
    shutil.rmtree(temp_preprocessing_dir, ignore_errors=True)
    print(f"🧹 Cleaned up temporary preprocessing directory")

    # Memory Cleanup before ML load
    gpu_memory_cleanup()
    
    # --- STEP 4: ML REFINEMENT ---
    print("\n" + "="*50)
    print("STEP 4: ML REFINEMENT")
    print("="*50)
    
    final_nifti_path = run_ml_refinement(
        image_path=final_source_nifti,
        mask_path=final_reg_mask,
        slice_range=detected_ranges,
        output_dir=final_output_dir,
        params=pipeline_params,
        base_name=patient_id,
        xgb_clf_params=xgb_clf_params,
        xgb_reg_params=xgb_reg_params
    )
    
    # --- STEP 5: DICOM EXPORT (Conditional) ---
    if os.path.isdir(input_target):
        print("\n" + "="*50)
        print("STEP 5: DICOM EXPORT")
        print("="*50)
        dicom_out_dir = os.path.join(final_output_dir, f"{patient_id}_pred_brain_mask")
        export_mask_to_dicom(input_target, final_nifti_path, dicom_out_dir)

    print(f"\n✅ PIPELINE COMPLETE in {time.time() - start_time:.2f}s")
    print(f"📁 All outputs saved to: {final_output_dir}")



if __name__ == "__main__":
    # 1. Multi-processing Setup
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # ==========================================
    # OPTIONAL: PARAMETER OVERRIDE
    # Paste your optimization dictionary here. 
    # If empty, defaults below will be used.
    # ==========================================
    opt_param =  {'kernel_size':105,'reg_n_estimators':4300,'iqr_sensitivity_factor':0.7,'reg_alpha':0.00007854591081542784,'reg_lambda':0.0000025207585143903743,'brain_definitness':0.6,'glcm_levels':64,'coarse_combination':'16_8','reg_gamma':0.005018537950592926,'clf_n_estimators':2300,'clf_max_depth':16,'clf_min_child_weight':13,'fine_combination':'8_4','coarse_lbp_radius':1,'fine_lbp_radius':5,'clf_learning_rate':0.0068969094258312935,'clf_subsample':0.9803717714201715,'clf_colsample_bytree':0.954291701997208,'clf_gamma':0.0019090275859725496,'reg_learning_rate':0.0026109485367941626,'reg_max_depth':14,'reg_subsample':0.5899083402916931,'reg_colsample_bytree':0.685157208237676,'reg_min_child_weight':13}
    # Parameters: {'kernel_size':105,'reg_n_estimators':4300,'iqr_sensitivity_factor':0.7,'reg_alpha':0.00007854591081542784,'reg_lambda':0.0000025207585143903743,'brain_definitness':0.6,'glcm_levels':64,'coarse_combination':'16_8','reg_gamma':0.005018537950592926,'clf_n_estimators':2300,'clf_max_depth':16,'clf_min_child_weight':13,'fine_combination':'8_4','coarse_lbp_radius':1,'fine_lbp_radius':5,'clf_learning_rate':0.0068969094258312935,'clf_subsample':0.9803717714201715,'clf_colsample_bytree':0.954291701997208,'clf_gamma':0.0019090275859725496,'reg_learning_rate':0.0026109485367941626,'reg_max_depth':14,'reg_subsample':0.5899083402916931,'reg_colsample_bytree':0.685157208237676,'reg_min_child_weight':13}
    # 2. Default Configuration (The "Magic Numbers")
    PIPELINE_CONFIG = {
        'template_path': 'mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin.nii',
        'template_mask': 'mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin_mask.nii',
        'coarse': {'patch_size': (64, 64), 'stride': (16, 16), 'lbp_radius': 2, 'glcm_levels': 32, 'valid_patch_threshold': 0.2},
        'fine': {'patch_size': (4, 4), 'stride': (2, 2), 'lbp_radius': 2, 'glcm_levels': 32, 'valid_patch_threshold': 0.2},
        'rf_depth': 700,
        'brain_definitness': 0.80,
        'kernel': 5,
        'use_crf': False,
        'predict_all': True,
        'train_brain_on_all': True,
        'use_parallel': True,
        'iqr_sensitivity_factor': 1.5,
        'save_files': False,
        'remove_scout': False  # Set to True to remove scout slice from DICOM input
    }

    xgb_clf_params = None
    xgb_reg_params = None

    # 3. Apply Overrides if opt_param is present
    if opt_param:
        print(f"⚙️ Applying Optimization Parameters...")
        
        # --- A. Update Pipeline Config ---
        if 'brain_definitness' in opt_param:
            PIPELINE_CONFIG['brain_definitness'] = opt_param['brain_definitness']
        if 'iqr_sensitivity_factor' in opt_param:
            PIPELINE_CONFIG['iqr_sensitivity_factor'] = opt_param['iqr_sensitivity_factor']
        if 'kernel_size' in opt_param:
            PIPELINE_CONFIG['kernel'] = int(opt_param['kernel_size']) # Mapping kernel_size -> kernel
        if 'glcm_levels' in opt_param:
            PIPELINE_CONFIG['coarse']['glcm_levels'] = int(opt_param['glcm_levels'])
            PIPELINE_CONFIG['fine']['glcm_levels'] = int(opt_param['glcm_levels'])
        if 'coarse_lbp_radius' in opt_param:
            PIPELINE_CONFIG['coarse']['lbp_radius'] = int(opt_param['coarse_lbp_radius'])
        if 'fine_lbp_radius' in opt_param:
            PIPELINE_CONFIG['fine']['lbp_radius'] = int(opt_param['fine_lbp_radius'])

        # Parse 'combination' strings (e.g., '8_4' -> patch(8,8), stride(4,4))
        if 'coarse_combination' in opt_param:
            p, s = map(int, opt_param['coarse_combination'].split('_'))
            PIPELINE_CONFIG['coarse']['patch_size'] = (p, p)
            PIPELINE_CONFIG['coarse']['stride'] = (s, s)
        
        if 'fine_combination' in opt_param:
            p, s = map(int, opt_param['fine_combination'].split('_'))
            PIPELINE_CONFIG['fine']['patch_size'] = (p, p)
            PIPELINE_CONFIG['fine']['stride'] = (s, s)

        # --- B. Build XGBoost Params ---
        xgb_clf_params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42
        }
        xgb_reg_params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42
        }

        # Separate clf_ and reg_ params
        for key, value in opt_param.items():
            if key.startswith('clf_'):
                clean_key = key.replace('clf_', '')
                # Ensure integer types for specific params
                if clean_key in ['n_estimators', 'max_depth', 'min_child_weight']:
                    xgb_clf_params[clean_key] = int(value)
                else:
                    xgb_clf_params[clean_key] = value
            elif key.startswith('reg_'):
                clean_key = key.replace('reg_', '')
                if clean_key in ['n_estimators', 'max_depth', 'min_child_weight']:
                    xgb_reg_params[clean_key] = int(value)
                else:
                    xgb_reg_params[clean_key] = value

        # Update rf_depth in config to match classifier n_estimators for consistency
        if 'n_estimators' in xgb_clf_params:
            PIPELINE_CONFIG['rf_depth'] = xgb_clf_params['n_estimators']
    
    # for i in range(1, 65):
    # for i in range(32, 65):
    # for i in range(65, 97):
    for i in range(100, 126): 
        print(f"\n\n================ Running Test Case {i} ================\n")
        # 4. Input Definitions
        INPUT_TARGET = f"NFBS_Dataset/{i}/{i}_T1w_sagittal.nii.gz"
        OUTPUT_BASE = "optuna_opt_trial/Hyper_Space_TPE_v1_H95_Only/19338"

            # 5. Run
        run_surgical_pipeline(INPUT_TARGET, OUTPUT_BASE, PIPELINE_CONFIG, xgb_clf_params, xgb_reg_params)