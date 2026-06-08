#!/usr/bin/env python

import mrcfile
import numpy as np
import os
import struct


def get_pixel_spacing(file_path):
    """
    Reads the first 1024 bytes of an MRC file to extract the header information,
    then computes the pixel spacing using the x length divided by the number of x voxels.
    
    Returns:
        The pixel spacing as a float.
    """
    with open(file_path, "rb") as f:
        header = f.read(1024)
        # Unpack nx, ny, nz from the first 12 bytes (3 integers)
        nx, ny, nz = struct.unpack('iii', header[0:12])
        # Unpack the cell dimensions (x_length, y_length, z_length) from bytes 40 to 51 (3 floats)
        x_length, y_length, z_length = struct.unpack('3f', header[40:52])
    if nx == 0:
        raise ValueError("Invalid header: nx is zero.")
    return x_length / nx


def load_mrc_file(file_path):
    """
    Load a .mrc file and return the data as a NumPy array.
    Includes checks for file existence and data integrity.
    Returns None if the file cannot be loaded, does not exist, or fails integrity checks.
    """
    if not os.path.exists(file_path):
        print(f"Warning: file does not exist: {file_path}")
        return None

    try:
        with mrcfile.open(file_path, permissive=True) as mrc:
            data = mrc.data.copy()

            # Check if data is empty
            if data.size < 2:
                print(f"Warning: {file_path} is empty.")
                return None

            # Check if data contains invalid values (NaN, Inf)
            if not np.isfinite(data).all():
                print(f"Warning: {file_path} contains NaN or Inf values.")
                return None

        return data

    except Exception as e:
        print(f"Warning: could not load {file_path}, error: {e}")
        return None


def read_mrc_header(file_path):
    """
    Reads the MRC header to extract metadata, including nx, ny, nz, and mode.
    """
    with open(file_path, "rb") as f:
        header = f.read(1024)
        nx, ny, nz = struct.unpack('iii', header[0:12])
        mode = struct.unpack('i', header[12:16])[0]

        mode_dict = {
            0: "uint8",
            1: "int16",
            2: "float32",
            3: "complex64",
            4: "float16",
            6: "uint16",
        }
        data_type = mode_dict.get(mode, "unknown")
        return {"nx": nx, "ny": ny, "nz": nz, "mode": mode, "data_type": data_type}

def save_mrc_image(file_path, data, original_header_file_to_copy=None):
    """
    Saves the data to an MRC file, copying and updating the header directly using struct.

    Args:
        file_path (str): Path to save the output MRC file.
        data (numpy.ndarray): The data array to save.
        original_header_file_to_copy (str, optional): Path to the original MRC file for copying header information.
    """
    # Ensure the data is float32 for MRC compatibility
    data = data.astype(np.float32)

    # Save the data to the output file
    with open(file_path, "wb") as f:
        # Write placeholder header (1024 bytes, will be updated later)
        f.write(b"\x00" * 1024)
        # Write the actual data
        f.write(data.tobytes())

    # Read and modify the header
    if original_header_file_to_copy:
        with open(original_header_file_to_copy, "rb") as ref_file:
            header = ref_file.read(1024)
    else:
        header = b"\x00" * 1024  # Create a blank header if none exists

    # Update header fields to reflect new data dimensions
    nz, ny, nx = data.shape  # Reorder dimensions for MRC convention
    header = bytearray(header)  # Convert to mutable byte array
    struct.pack_into("iii", header, 0, nx, ny, nz)  # Update nx, ny, nz (offset 0-11)

    # Update mode (data type)
    mode_dict = {np.uint8: 0, np.int16: 1, np.float32: 2}
    mode = mode_dict.get(data.dtype.type, 2)  # Default to float32
    struct.pack_into("i", header, 12, mode)  # Update mode (offset 12-15)

    # Update data statistics (dmin, dmax, dmean)
    dmin, dmax, dmean = np.min(data), np.max(data), np.mean(data)
    struct.pack_into("fff", header, 76, dmin, dmax, dmean)  # Update stats (offset 76-87)

    # Write the updated header back to the output file
    with open(file_path, "r+b") as f:
        f.write(header)  # Overwrite the placeholder header with updated values

    #print(f"Header updated: nx={nx}, ny={ny}, nz={nz}, mode={mode}, dmin={dmin}, dmax={dmax}, dmean={dmean}")


def log_to_file(log_file, message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
        
