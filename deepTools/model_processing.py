#!/usr/bin/env python

import numpy as np
import mrcfile
import argparse
import os
import json
import sys
from scipy.ndimage import (
    gaussian_laplace,
    binary_erosion,
    binary_dilation,
    binary_closing,
    binary_opening,
    rotate,
    label,
    generate_binary_structure,
    gaussian_filter
)
from scipy.fftpack import fftn, ifftn
import struct
from .utils import read_mrc_header, save_mrc_image, load_mrc_file, get_pixel_spacing



# Core functions for processing
def load_mrc_image(file_path):
    with mrcfile.open(file_path, mode="r") as mrc:
        return mrc.data.copy()


def save_mrc_image2(file_path, data, original_header_file_to_copy=None):
    with mrcfile.new(file_path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.update_header_from_data()

    if original_header_file_to_copy is not None:
        header_size = 1024
        with open(original_header_file_to_copy, "rb") as ref_file:
            header = ref_file.read(header_size)
        with open(file_path, "r+b") as output_file:
            output_file.write(header)





def replace_fourier_amplitudes(input_image, amplitude_image):
    input_fft = fftn(input_image)
    amplitude_fft = np.abs(fftn(amplitude_image))
    phase = np.angle(input_fft)
    new_fft = amplitude_fft * np.exp(1j * phase)
    modified_image = np.real(ifftn(new_fft))
    return modified_image

def flat_fourier_amplitudes(input_image):
    input_fft = fftn(input_image)
    phase = np.angle(input_fft)
    new_fft = np.exp(1j * phase)
    modified_image = np.real(ifftn(new_fft))
    return modified_image


def compute_masked_mean(input_image, mask):
    unique_values = np.unique(mask)
    output_image = np.zeros_like(input_image)

    for value in unique_values:
        region_mask = (mask == value)
        mean_value = np.mean(input_image[region_mask])
        output_image[region_mask] = mean_value

    return output_image


def create_sphere_mask(volume_shape):
    center = tuple(dim // 2 for dim in volume_shape)
    radius = min(volume_shape) / 2
    x = np.arange(0, volume_shape[0])
    y = np.arange(0, volume_shape[1])
    z = np.arange(0, volume_shape[2])
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    distance = np.sqrt((xv - center[0]) ** 2 + (yv - center[1]) ** 2 + (zv - center[2]) ** 2)
    sphere_mask = (distance <= radius).astype(np.float32)
    return sphere_mask

def sdlevel_threshold(input_image, sdlevel):
    mean_density = np.mean(input_image)
    std_density = np.std(input_image)
    threshold_value = mean_density + sdlevel * std_density
    thresholded_volume = np.where(input_image >= threshold_value, 1, 0)
    return thresholded_volume


def add_masked_noise(image, mask):
    mask = np.clip(mask, 0, 1)
    masked_region = image * mask
    min_value = np.min(masked_region[mask > 0])
    random_factor = np.random.uniform(0.9, 2.0)
    std_value = np.std(masked_region[mask > 0]) * random_factor
    noise = np.random.normal(min_value, std_value, image.shape)
    noisy_image = noise * (1 - mask) + image * mask
    return noisy_image


def apply_masked_resolution(image, mask, worst_resolution_value=60.0):
    mask = np.clip(mask, 0, 1)
    modified_image = (mask * image) + ((1 - mask) * worst_resolution_value)
    return modified_image


def apply_masked_crop(image, mask):
    non_zero_indices = np.argwhere(mask > 0)
    min_coords = non_zero_indices.min(axis=0)
    max_coords = non_zero_indices.max(axis=0) + 1
    cropped_image = image[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
    return cropped_image


def delete_dust(image, min_size):
    structure = generate_binary_structure(3, 1)
    labeled_array, num_features = label(image > 0, structure=structure)
    result_image = np.zeros_like(image)

    for feature_id in range(1, num_features + 1):
        component_size = np.sum(labeled_array == feature_id)
        if component_size >= min_size:
            result_image[labeled_array == feature_id] = image[labeled_array == feature_id]

    return result_image


def rotate_3d_image(image, phi, theta, psi):
    rotated = rotate(image, phi, axes=(1, 0), reshape=False, order=1)
    rotated = rotate(rotated, theta, axes=(2, 0), reshape=False, order=1)
    rotated = rotate(rotated, psi, axes=(2, 1), reshape=False, order=1)
    return rotated

def process_map(input_map, args, argv_list=None):
    """
    Processes the input map based on the parsed arguments.
    If argv_list is provided, it is used instead of sys.argv[1:].
    """
    if argv_list is None:
        argv_order = sys.argv[1:]  # Default to the command-line arguments
    else:
        argv_order = argv_list

    operation_list = []
    args_dict = vars(args)

    i = 0
    while i < len(argv_order):
        arg = argv_order[i]
        if arg.startswith("--"):
            key = arg.lstrip("--")
            if key in args_dict:
                values = args_dict[key]
                if key in ["mask_threshold_mean", "stretch_mask", "smooth_binary_mask"]:
                    # Package the next 3 arguments into a tuple
                    operation_list.append((key, (argv_order[i + 1], argv_order[i + 2], argv_order[i + 3])))
                    i += 4  # Skip the next three arguments
                    continue
                elif values is not None:
                    if isinstance(values, bool) and values:
                        operation_list.append((key, None))
                    elif isinstance(values, list):
                        for value in values:
                            operation_list.append((key, value))
                    elif i + 1 < len(argv_order) and not argv_order[i + 1].startswith("--"):
                        operation_list.append((key, argv_order[i + 1]))
                        i += 2  # Skip the next argument
                        continue
        i += 1

    # Process operations in the captured order.
    for operation, value in operation_list:
        if operation == "amplFlat" and value is None:
            input_map = process_operation(input_map, operation, None, args)
        else:
            print(f"Applying operation: {operation} with value: {value}")
            input_map = process_operation(input_map, operation, value, args)
    return input_map


def process_operation(input_map, operation, value, args):
    if operation == "erode":
        input_map = binary_erosion(input_map > 0, generate_binary_structure(3, value)).astype(np.float32)
    elif operation == "dilate":
        input_map = binary_dilation(input_map > 0, generate_binary_structure(3, value)).astype(np.float32)
    elif operation == "close":
        input_map = binary_closing(input_map > 0, generate_binary_structure(3, value)).astype(np.float32)
    elif operation == "open":
        input_map = binary_opening(input_map > 0, generate_binary_structure(3, value)).astype(np.float32)
    elif operation == "threshold":
        input_map = (input_map >= value).astype(np.float32)
    elif operation == "amplReplace":
        amplitude_image = load_mrc_image(value)
        input_map = replace_fourier_amplitudes(input_map, amplitude_image)
    elif operation == "amplFlat":
        input_map = flat_fourier_amplitudes(input_map)        
    elif operation == "masked_resolution":
        mask, worst_res = value
        input_map = apply_masked_resolution(input_map, mask, worst_res)
    elif operation == "masked_crop":
        mask = load_mrc_image(value)
        input_map = apply_masked_crop(input_map, mask)
    elif operation == "add_masked_noise":
        mask = load_mrc_image(value)
        input_map = add_masked_noise(input_map, mask)
    elif operation == "delete_dust":
        input_map = delete_dust(input_map, value)
    elif operation == "rotate":
        phi, theta, psi = value
        input_map = rotate_3d_image(input_map, phi, theta, psi)
    elif operation == "create_sphere_mask":
        input_map = create_sphere_mask(input_map.shape)
    elif operation == "compute_masked_mean":
        mask = load_mrc_image(value)
        input_map = compute_masked_mean(input_map, mask)
    elif operation == "smooth":
        input_map = gaussian_filter(input_map, sigma=float(value))
    elif operation == "sdlevel":
        input_map = sdlevel_threshold(input_map, value)
    elif operation == "mask_threshold_mean":
        mask_file, lower_thresh, upper_thresh = value
        mask = load_mrc_image(mask_file)
        input_map = apply_mask_threshold_mean(input_map, mask, float(lower_thresh), float(upper_thresh))
    elif operation == "smooth_binary_mask":
        mask_file, lower_thresh, upper_thresh = value
        mask = load_mrc_image(mask_file)
        input_map = apply_smooth_binary_mask(mask, float(lower_thresh), float(upper_thresh))
    elif operation == "divide_by_pixel_spacing":
        if not hasattr(args, "pixel_spacing"):
            raise ValueError("Pixel spacing not defined. Ensure the MRC header is read and pixel_spacing is set.")
        input_map = input_map / args.pixel_spacing
    elif operation == "multiply_by_pixel_spacing":
        if not hasattr(args, "pixel_spacing"):
            raise ValueError("Pixel spacing not defined. Ensure the MRC header is read and pixel_spacing is set.")
        input_map = input_map * args.pixel_spacing
    elif operation == "invert_density":
        input_map = -input_map
    elif operation == "invert_resolution":
        # Invert each pixel value: if value < 0.1 then set to 0, else compute 1/value.
        with np.errstate(divide='ignore', invalid='ignore'):
            input_map = np.where(input_map < 0.1, 0, 1.0 / input_map)
    return input_map

def apply_mask_threshold_mean(image, mask, lower_thresh, upper_thresh):
    mask = np.clip(mask, 0, 1)
    below_mask = mask < lower_thresh
    above_mask = mask > upper_thresh

    mean_below = np.mean(image[below_mask]) if np.any(below_mask) else 0
    mean_above = np.mean(image[above_mask]) if np.any(above_mask) else 0

    output_image = np.copy(image)
    output_image[below_mask] = mean_below
    output_image[above_mask] = mean_above

    scale = (mask - lower_thresh) / (upper_thresh - lower_thresh)
    scale = np.clip(scale, 0, 1)
    middle_mask = (mask >= lower_thresh) & (mask <= upper_thresh)
    output_image[middle_mask] = (
        scale[middle_mask] * mean_above + (1 - scale[middle_mask]) * mean_below
    )

    return output_image


def apply_stretch_mask(mask, lower_thresh, upper_thresh):
    """
    Stretches the mask values to range [0, 1] based on lower and upper thresholds.

    Args:
        mask (numpy.ndarray): Input mask with variable intensities.
        lower_thresh (float): Lower threshold to map to 0.
        upper_thresh (float): Upper threshold to map to 1.

    Returns:
        numpy.ndarray: Stretched mask with values in range [0, 1].
    """
    mask = np.clip(mask, 0, 1)
    stretched_mask = (mask - lower_thresh) / (upper_thresh - lower_thresh)
    stretched_mask = np.clip(stretched_mask, 0, 1)
    return stretched_mask

def apply_smooth_binary_mask(mask, lower_thresh, upper_thresh):
    """
    Applies thresholding, dust removal, and smoothing to produce a binary mask with smooth contours.

    Args:
        mask (numpy.ndarray): Input mask with variable intensities.
        lower_thresh (float): Lower threshold to map to 0.
        upper_thresh (float): Upper threshold to map to 1.

    Returns:
        numpy.ndarray: Smoothed binary mask.
    """
    # Clip mask values
    mask = np.clip(mask, 0, 1)

    # Stretch mask values
    stretched_mask = (mask - lower_thresh) / (upper_thresh - lower_thresh)
    stretched_mask = np.clip(stretched_mask, 0, 1)

    # Binarize mask
    binary_mask = (stretched_mask > 0.5).astype(np.float32)

    # Remove small dust particles
    binary_mask = delete_dust(binary_mask, min_size=8)

    # Smooth the mask
    smoothed_mask = gaussian_filter(binary_mask, sigma=1.0)

    # Re-binarize after smoothing
    final_mask = (smoothed_mask > 0.5).astype(np.float32)

    return final_mask

def setup_parser(require_io=False):
    parser = argparse.ArgumentParser(description="Process a 3D MRC density map with various options.")
    parser.add_argument("--i", help="Input MRC file path.", required=require_io)
    parser.add_argument("--o", help="Output file path.", required=require_io)
    parser.add_argument("--i_json", nargs=3, help="JSON input: json_file json_key file_suffix.")
    parser.add_argument("--smooth", type=float, help="Sigma for Gaussian smoothing.")
    parser.add_argument("--erode", type=int, action="append", help="Radius for morphological erosion.")
    parser.add_argument("--dilate", type=int, action="append", help="Radius for morphological dilation.")
    parser.add_argument("--close", type=int, action="append", help="Radius for morphological closing.")
    parser.add_argument("--threshold", type=float, action="append", help="Threshold value for binary map conversion.")
    parser.add_argument("--sdlevel", type=float, action="append", help="Threshold based on sdlevel.")
    parser.add_argument("--amplReplace", metavar="AMPLITUDE_FILE", help="Replace Fourier amplitudes.")
    parser.add_argument("--amplFlat", action="store_true", help="Erase Fourier amplitudes.")
    parser.add_argument("--masked_resolution", nargs="?", const=60.0, type=float, help="Replace values outside the mask with the worst resolution value.")
    parser.add_argument("--masked_crop", metavar="MASK_FILE", help="Crop input image to fit within non-zero mask regions.")
    parser.add_argument("--add_masked_noise", metavar="MASK_FILE", help="Add noise to regions outside the mask.")
    parser.add_argument("--delete_dust", type=int, action="append", help="Remove small features below this size.")
    parser.add_argument("--rotate", nargs=3, type=float, action="append", help="Rotate the image by phi, theta, psi angles.")
    parser.add_argument("--create_sphere_mask", action="store_true", help="Create a binary sphere mask.")
    parser.add_argument("--compute_masked_mean", metavar="MASK_FILE", help="Compute mean values for regions defined by the mask.")
    parser.add_argument("--soft_masked_mean", metavar="MASK_FILE", help="truncate a mask by min/max threshold, and place intermediate image mean values between min and max mask values")
    parser.add_argument("--invert_density", action="store_true", help="Multiply input map by -1.")   
    parser.add_argument("--divide_by_pixel_spacing", action="store_true", help="Divide all voxel values by the pixel spacing extracted from the MRC header.")
    parser.add_argument("--multiply_by_pixel_spacing", action="store_true", help="Multiply all voxel values by the pixel spacing extracted from the MRC header.")
    parser.add_argument("--invert_resolution", action="store_true", help="Invert the input image: for each pixel, if the value is less than 0.1, set it to 0; otherwise compute 1/pixel value.")
    parser.add_argument(
        "--mask_threshold_mean",
        nargs=3,
        metavar=("MASK_FILE", "LOWER_THRESH", "UPPER_THRESH"),
        help="Apply mask thresholding and compute mean values."
    )
    parser.add_argument(
        "--stretch_mask",
        nargs=3,
        metavar=("MASK_FILE", "LOWER_THRESH", "UPPER_THRESH"),
        help="Stretch mask values to [0, 1] based on thresholds."
    )
    parser.add_argument(
        "--smooth_binary_mask",
        nargs=3,
        metavar=("MASK_FILE", "LOWER_THRESH", "UPPER_THRESH"),
        help="Apply thresholding, remove dust, and smooth the binary mask for smoother contours."
    )
    return parser



def inline_map_processing(input_map, arg_string, pixel_spacing=None):
    parser = setup_parser(require_io=False)
    arg_list = arg_string.split()  # Split the argument string
    args = parser.parse_args(arg_list)
    if pixel_spacing is not None:
        args.pixel_spacing = pixel_spacing
    processed_map = process_map(input_map, args, argv_list=arg_list)
    return processed_map



# #################
# Main script
def main():

    parser = setup_parser(require_io=False)
    args = parser.parse_args()

    if args.i:
        input_map = load_mrc_image(args.i)
        #with mrcfile.open(args.i, mode="r") as mrc:
        #    args.pixel_spacing = mrc.header.cella.x / mrc.header.nx
        args.pixel_spacing = get_pixel_spacing(args.i)
        result_map = process_map(input_map, args)
        save_mrc_image(args.o, result_map, args.i)

    elif args.i_json:
        json_file, json_key, file_suffix = args.i_json
        with open(json_file, "r") as f:
            data = json.load(f)

        for entry in data:
            if json_key not in entry:
                raise ValueError(f"Key '{json_key}' not found in JSON entry.")
            input_file = entry[json_key]
            if not os.path.exists(input_file):
                raise ValueError(f"Input file {input_file} does not exist.")

            input_map = load_mrc_image(input_file)
            result_map = process_map(input_map, args)

            if input_file.lower().endswith(".mrc"):
                base_name = input_file[:-4]
                output_file = f"{base_name}_{file_suffix}.mrc"
            else:
                output_file = f"{input_file}_{file_suffix}.mrc"

            save_mrc_image(output_file, result_map,input_file)

            if json_key.endswith("_file"):
                prefix = json_key.rsplit("_file", 1)[0]
                new_key = f"{prefix}_{file_suffix}_file"
            else:
                new_key = f"{json_key}_{file_suffix}"

            entry[new_key] = output_file

        with open(args.o, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Processed files and updated JSON saved to {args.o}")


if __name__ == "__main__":
    main()

