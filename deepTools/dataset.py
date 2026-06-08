#!/usr/bin/env python

import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
from .utils import load_mrc_file, save_mrc_image , log_to_file
from scipy.ndimage import (
    gaussian_laplace,
    rotate
)
import random
from .map_processing import inline_map_processing

def compute_abs_laplacian_of_gaussian(volume, sigma=0.0001):
    return np.abs(gaussian_laplace(volume, sigma=sigma))

def rotate_3d_image(image, phi, theta, psi):
    rotated = rotate(image, phi, axes=(1, 0), reshape=False, order=1)
    rotated = rotate(rotated, theta, axes=(2, 0), reshape=False, order=1)
    rotated = rotate(rotated, psi, axes=(2, 1), reshape=False, order=1)
    return rotated
def generate_random_angles():
    phi = random.uniform(0, 360)  # Rotation around Z-axis
    theta = random.uniform(0, 180)  # Rotation around Y-axis (restricted to 0-180 degrees)
    psi = random.uniform(0, 360)  # Rotation around X-axis
    return phi, theta, psi
def crop_to_max_size(volume, max_size):
    """
    Crops a 3D volume to the specified max size along all dimensions.
    If the volume is smaller than max size, it remains unchanged.
    """
    D, H, W = volume.shape
    if D <= max_size and H <= max_size and W <= max_size:
        return volume

    start_D = max(0, (D - max_size) // 2)
    start_H = max(0, (H - max_size) // 2)
    start_W = max(0, (W - max_size) // 2)

    end_D = start_D + max_size
    end_H = start_H + max_size
    end_W = start_W + max_size

    return volume[start_D:end_D, start_H:end_H, start_W:end_W]
def crop_to_mask(image, mask):
    non_zero_indices = np.argwhere(mask > 0)
    min_coords = non_zero_indices.min(axis=0)
    max_coords = non_zero_indices.max(axis=0) + 1
    cropped_image = image[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
    return cropped_image


def standard_normalize(volume, epsilon=1e-8):
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    if std_val < epsilon:
        return volume - mean_val
    return (volume - mean_val) / std_val

def robust_normalize_patch(volume, epsilon=1e-8, fallback_scaling=1.0, smoothing=0.1, dataset_obj=None):
    """
    Normalises the volume using its median and MAD.
    If MAD is too small (i.e. volume is nearly homogeneous), then use a global scaling factor
    (stored in dataset_obj.global_robust_scaling) updated over time.
    If no global value exists yet, fallback to fallback_scaling.
    """
    med = np.median(volume)
    mad = np.median(np.abs(volume - med))
    # If the patch has enough variation, use a default factor (1.4826) and update the global value.
    if mad >= epsilon:
        local_scaling = 1.4826
        if dataset_obj is not None:
            if dataset_obj.global_robust_scaling is None:
                dataset_obj.global_robust_scaling = local_scaling
            else:
                # Exponential moving average update.
                dataset_obj.global_robust_scaling = ((1 - smoothing) * dataset_obj.global_robust_scaling +
                                                     smoothing * local_scaling)
        robust_std = mad * local_scaling
    else:
        # Patch is nearly homogeneous. Use the global robust scaling if available.
        if dataset_obj is not None and dataset_obj.global_robust_scaling is not None:
            local_scaling = dataset_obj.global_robust_scaling
        else:
            local_scaling = fallback_scaling
        robust_std = mad * local_scaling

    if robust_std < epsilon:
        return volume - med
    return (volume - med) / robust_std

def blended_normalize_patch(volume, alpha=0.5, epsilon=1e-8, fallback_scaling=1.0, smoothing=0.1, dataset_obj=None):
    robust_norm = robust_normalize_patch(volume, epsilon, fallback_scaling, smoothing, dataset_obj)
    standard_norm = standard_normalize(volume, epsilon)
    return alpha * robust_norm + (1 - alpha) * standard_norm

def random_patch(cryo_em, target, mask, patch_size, min_mask_ratio=0.3, max_attempts=60, flat_signal=0.1):
    """
    Extracts a random cubic patch from the input volumes.
    If a mask is provided, the function repeatedly samples random patches until
    a patch is found in which the mask contains at least min_mask_ratio of positive voxels.
    In addition, if a patch exhibits flat homogeneous signal over at least a fraction
    (flat_signal) of its voxels, that patch is skipped.
    The function terminates early when a valid patch is encountered.
    A maximum number of attempts is enforced; if no valid patch is found within these attempts,
    the patch with the highest mask coverage is returned.
    Returns patched cryo_em, target, and mask (mask is None if input mask is None).
    """
    D, H, W = cryo_em.shape
    # If volume dimensions are smaller than the patch size, return the original volumes (after cleaning NaN)
    if D < patch_size or H < patch_size or W < patch_size:
        cryo_em = np.nan_to_num(cryo_em, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)
        if mask is not None:
            mask = np.nan_to_num(mask, nan=0.0)
        return cryo_em, target, mask

    best_mask_sum = -1
    best_patch = None
    attempts = 0
    eps = 1e-6  # tolerance for detecting flat (homogeneous) signal

    while attempts < max_attempts:
        d_start = random.randint(0, D - patch_size)
        h_start = random.randint(0, H - patch_size)
        w_start = random.randint(0, W - patch_size)
        patch_cryo = cryo_em[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size]
        patch_target = target[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size]
        
        # Check for flat homogeneous signal in the patch
        flat_count = (abs(patch_cryo - np.median(patch_cryo)) < eps).sum()
        if flat_count >= flat_signal * (patch_size ** 3):
            attempts += 1
            continue

        if mask is not None:
            patch_mask = mask[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size]
            if patch_mask.sum() >= min_mask_ratio * (patch_size ** 3):
                # Replace NaN values with zero before returning.
                patch_cryo = np.nan_to_num(patch_cryo, nan=0.0)
                patch_target = np.nan_to_num(patch_target, nan=0.0)
                patch_mask = np.nan_to_num(patch_mask, nan=0.0)
                return patch_cryo, patch_target, patch_mask
            if patch_mask.sum() > best_mask_sum:
                best_mask_sum = patch_mask.sum()
                best_patch = (patch_cryo, patch_target, patch_mask)
        else:
            # No mask provided; clean patches for NaN and return.
            patch_cryo = np.nan_to_num(patch_cryo, nan=0.0)
            patch_target = np.nan_to_num(patch_target, nan=0.0)
            return patch_cryo, patch_target, None
        attempts += 1

    if best_patch is not None:
        patch_cryo, patch_target, patch_mask = best_patch
        patch_cryo = np.nan_to_num(patch_cryo, nan=0.0)
        patch_target = np.nan_to_num(patch_target, nan=0.0)
        patch_mask = np.nan_to_num(patch_mask, nan=0.0)
        return patch_cryo, patch_target, patch_mask

    # Fallback: return a random patch if no valid patch was found within the maximum attempts.
    d_start = random.randint(0, D - patch_size)
    h_start = random.randint(0, H - patch_size)
    w_start = random.randint(0, W - patch_size)
    patch_cryo = cryo_em[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size]
    patch_target = target[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size]
    if mask is not None:
        patch_mask = mask[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size]
        patch_mask = np.nan_to_num(patch_mask, nan=0.0)
    else:
        patch_mask = None
    patch_cryo = np.nan_to_num(patch_cryo, nan=0.0)
    patch_target = np.nan_to_num(patch_target, nan=0.0)
    return patch_cryo, patch_target, patch_mask



class CryoEMDataset(Dataset):
    def __init__(self, data_list, cryo_em_keys, target_keys, mask_keys=None,
                 transform=None, scale_factor=0.3, pixel_size_default=1.0,
                 normalize_input=True, use_robust_normalization=False, robust_normalization_alpha=0.5, robust_fallback_scaling=1.0,robust_smoothing=0.1,
                 normalize_reference=False, logfile=None, random_rotation=False, max_volume_side_length=-1, masked_crop=False, 
                 use_random_patch_training=False,random_patch_training_size=64,
                 inline_input_preProcessing="", inline_input_preProcessing_command="--threshold 0.01 --dilate 3 --close 5", 
                 inline_target_preProcessing="", inline_target_preProcessing_command="--threshold 0.01 --dilate 3 --close 5", 
                 masked_learning=False):
        """
        Args:
            normalize_input (bool): Apply normalization to cryo-EM input data.
            normalize_reference (bool): Apply normalization to the target data.
            random_rotation (bool): Apply random rotations to the cryo-EM and target data.
        """
        self.transform = transform
        self.logfile = logfile
        self.scale_factor = scale_factor
        self.pixel_size_default = pixel_size_default
        self.normalize_input = normalize_input
        self.use_robust_normalization=use_robust_normalization
        self.robust_normalization_alpha=robust_normalization_alpha
        self.robust_fallback_scaling=robust_fallback_scaling
        self.robust_smoothing = robust_smoothing
        self.global_robust_scaling = None
        self.normalize_reference = normalize_reference
        self.use_random_patch_training=use_random_patch_training
        self.random_patch_training_size=random_patch_training_size
        self.inline_input_preProcessing = inline_input_preProcessing
        self.inline_input_preProcessing_command= inline_input_preProcessing_command
        self.inline_target_preProcessing = inline_target_preProcessing
        self.inline_target_preProcessing_command= inline_target_preProcessing_command
        self.max_volume_side_length = max_volume_side_length
        self.masked_crop=masked_crop
        self.masked_learning=masked_learning
        self.random_rotation = random_rotation  # New parameter to control random rotation
        self.samples = []
        for entry in data_list:
            #log_to_file("logfile0.log", f"entry: entry={entry}")
            pixel_size = entry.get('pixel_size', self.pixel_size_default)
            for i in range(len(cryo_em_keys)):
                cryo_key = cryo_em_keys[i]
                target_key = target_keys[i]
                #mask_key = mask_keys[i] if mask_keys and i < len(mask_keys) else None
                mask_key = mask_keys[i] if mask_keys and i < len(mask_keys) and mask_keys[i] else None
                #log_to_file("logfile0.log", f"sample: cryo_key={cryo_key}, target_key={target_key}, mask_key={mask_key}")
                if cryo_key not in entry or target_key not in entry:
                    continue
                #log_to_file("logfile.log", f"sample: map_path={entry[cryo_key]}, target_path={entry[target_key]}, mask_path={mask_key}")
                self.samples.append({
                    "map_path": entry[cryo_key],
                    "target_path": entry[target_key],
                    "mask_path": entry[mask_key] if mask_key else None,
                    "pixel_size": pixel_size
                })

        if len(self.samples) == 0:
            raise ValueError("No valid samples found. Check your data or keys.")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        map_path = sample["map_path"]
        target_path = sample["target_path"]
        mask_path = sample["mask_path"]
        pixel_size = sample["pixel_size"]

        # Logging
        if self.logfile:
            log_to_file(self.logfile, f"Loading sample: map={map_path}, target={target_path}, mask={mask_path}")

        # Load cryo-EM map and target
        cryo_em = load_mrc_file(map_path)
        target = load_mrc_file(target_path)


        mask = None
        if mask_path:
            #print("mask_path")
            mask = load_mrc_file(mask_path)
            if self.masked_crop:
                cryo_em=crop_to_mask(cryo_em, mask)
                target=crop_to_mask(target, mask)


        maskLearning=None
        mask_tensor = None
        if self.masked_learning and not self.use_random_patch_training:
            if mask_path:
                maskLearning = load_mrc_file(mask_path)
                if self.masked_crop:
                    maskLearning=crop_to_mask(maskLearning, mask)
                mask_tensor = torch.from_numpy(maskLearning).bool().unsqueeze(0)


        if self.use_random_patch_training:
            cryo_em, target, mask = random_patch(cryo_em, target, mask, patch_size=self.random_patch_training_size)
            if self.masked_learning:
                mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0)


        if cryo_em is None or target is None:
            raise ValueError(f"Failed to load map or target file at {map_path} or {target_path}")

        # Crop if max_volume_side_length is set and positive
        if self.max_volume_side_length > 0:
            cryo_em = crop_to_max_size(cryo_em, self.max_volume_side_length)
            target = crop_to_max_size(target, self.max_volume_side_length)
            if mask_tensor is not None:
                mask_to_crop = maskLearning if maskLearning is not None else mask
                mask_tensor = torch.from_numpy(crop_to_max_size(mask_to_crop, self.max_volume_side_length)).bool().unsqueeze(0)


        #save_mrc_image("target_after_processing.mrc", cryo_em)
        #print("save and exit")
        #exit()


        if self.inline_input_preProcessing:
            cryo_em = inline_map_processing(target, self.inline_input_preProcessing_command, pixel_spacing=pixel_size)


        # Apply inline target pre-processing if enabled.
        #save_mrc_image("target_map.mrc", cryo_em)
        #save_mrc_image("target_before_processing.mrc", target)
        if self.inline_target_preProcessing: # and self.inline_target_preProcessing_command:
            #target = inline_map_processing(target, self.inline_target_preProcessing_command)
            target = inline_map_processing(target, self.inline_target_preProcessing_command, pixel_spacing=pixel_size)
            #print ("processing map")
        #save_mrc_image("target_after_processing.mrc", target)
        #print("save and exit")
        #exit()

        # Normalize cryo-EM map if flag is True
        if self.normalize_input:
            if self.use_robust_normalization:
                cryo_em = blended_normalize_patch(cryo_em, 
                                            alpha=self.robust_normalization_alpha,
                                            epsilon=1e-8,
                                            fallback_scaling=self.robust_fallback_scaling,
                                            smoothing=self.robust_smoothing,
                                            dataset_obj=self)
            else:
                std_val = np.std(cryo_em)
                cryo_em = (cryo_em - np.mean(cryo_em)) / ((std_val + 1e-8) if std_val != 0 else 1.0)

        # Normalize target if flag is True
        if self.normalize_reference:
            target = (target - np.mean(target)) / ((np.std(target) + 1e-8) if np.std(target) != 0 else 1.0)

        # Apply random rotation if enabled
        if self.random_rotation:
            phi, theta, psi = generate_random_angles()
            cryo_em = rotate_3d_image(cryo_em, phi, theta, psi)
            target = rotate_3d_image(target, phi, theta, psi)
            if self.logfile:
                log_to_file(self.logfile, f"Rotated sample: phi={phi}, theta={theta}, psi={psi}")

        cryo_em_tensor = torch.from_numpy(cryo_em).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target).float().unsqueeze(0)

        # Existing padding and transformations remain unchanged
        D, H, W = cryo_em_tensor.shape[1:]
        target_D = ((D - 1) // 16 + 1) * 16
        target_H = ((H - 1) // 16 + 1) * 16
        target_W = ((W - 1) // 16 + 1) * 16

        pad_D = target_D - D
        pad_H = target_H - H
        pad_W = target_W - W

        padding = (0, pad_W, 0, pad_H, 0, pad_D)
        cryo_em_tensor = F.pad(cryo_em_tensor, padding)
        target_tensor = F.pad(target_tensor, padding)
        if mask_tensor is not None:
            mask_tensor = F.pad(mask_tensor, padding)

        # Ensure mask_tensor is always a valid tensor
        if mask_tensor is None:
            mask_tensor = torch.zeros_like(target_tensor, dtype=torch.bool)  # Placeholder tensor with same shape


        # Downsampling if needed
        if self.scale_factor < 1.0:
            cryo_em_tensor = F.interpolate(cryo_em_tensor.unsqueeze(0), scale_factor=self.scale_factor, mode='trilinear', align_corners=False).squeeze(0)
            target_tensor = F.interpolate(target_tensor.unsqueeze(0), scale_factor=self.scale_factor, mode='trilinear', align_corners=False).squeeze(0)
            if mask_tensor is not None:
                mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).float(), scale_factor=self.scale_factor, mode='nearest').squeeze(0).bool()


        # Apply optional transformations
        if self.transform:
            cryo_em_tensor = self.transform(cryo_em_tensor)
            target_tensor = self.transform(target_tensor)
            if mask_tensor is not None:
                mask_tensor = self.transform(mask_tensor)

        # Ensure mask_tensor is handled according to masked_learning flag
        if self.masked_learning:
            if mask_tensor is None:
                mask_tensor = torch.ones_like(target_tensor, dtype=torch.bool)
        else:
            mask_tensor = torch.ones_like(target_tensor, dtype=torch.bool)

        return cryo_em_tensor, target_tensor, mask_tensor
