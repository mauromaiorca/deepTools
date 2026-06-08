#!/usr/bin/env python
import numpy as np
import struct
import mrcfile
from collections import namedtuple
from dataclasses import dataclass
from Bio.PDB import MMCIFParser, PDBParser
#from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
import argparse


##############################################
# Provided MRC functions
##############################################

MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin"])

def load_mrc(mrc_fn: str, multiply_global_origin: bool = True) -> MRCObject:
    mrc_file_handle = mrcfile.open(mrc_fn, "r")
    voxel_size = float(mrc_file_handle.voxel_size.x)
    if voxel_size <= 0:
        raise RuntimeError(f"Seems like the MRC file: {mrc_fn} does not have a header.")
    c = mrc_file_handle.header["mapc"]
    r = mrc_file_handle.header["mapr"]
    s = mrc_file_handle.header["maps"]
    global_origin = mrc_file_handle.header["origin"]
    global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
    global_origin[0] += mrc_file_handle.header["nxstart"]
    global_origin[1] += mrc_file_handle.header["nystart"]
    global_origin[2] += mrc_file_handle.header["nzstart"]
    if multiply_global_origin:
        global_origin *= mrc_file_handle.voxel_size.x
    if c == 1 and r == 2 and s == 3:
        grid = mrc_file_handle.data
    elif c == 3 and r == 2 and s == 1:
        grid = np.moveaxis(mrc_file_handle.data, [0, 1, 2], [2, 1, 0])
    elif c == 2 and r == 1 and s == 3:
        grid = np.moveaxis(mrc_file_handle.data, [1, 2, 0], [2, 1, 0])
    else:
        raise RuntimeError("MRC file axis arrangement not supported!")
    return MRCObject(grid, voxel_size, global_origin)

def save_mrc_image(file_path, data, original_header_file_to_copy=None):
    """
    Saves the data to an MRC file, copying and updating the header from an
    original MRC file if provided.
    """
    data = data.astype(np.float32)
    with open(file_path, "wb") as f:
        f.write(b"\x00" * 1024)
        f.write(data.tobytes())
    if original_header_file_to_copy:
        with open(original_header_file_to_copy, "rb") as ref_file:
            header = ref_file.read(1024)
    else:
        header = b"\x00" * 1024
    # MRC convention: data shape is (nz, ny, nx)
    nz, ny, nx = data.shape
    header = bytearray(header)
    struct.pack_into("iii", header, 0, nx, ny, nz)
    mode_dict = {np.uint8: 0, np.int16: 1, np.float32: 2}
    mode = mode_dict.get(data.dtype.type, 2)
    struct.pack_into("i", header, 12, mode)
    dmin, dmax, dmean = float(np.min(data)), float(np.max(data)), float(np.mean(data))
    struct.pack_into("fff", header, 76, dmin, dmax, dmean)
    with open(file_path, "r+b") as f:
        f.write(header)
    #print(f"Header updated: nx={nx}, ny={ny}, nz={nz}, mode={mode}, dmin={dmin}, dmax={dmax}, dmean={dmean}")

def gaussian_kernel_3d(sigma_voxel):
    kernel_radius = int(np.ceil(3 * sigma_voxel))
    kernel_size = 2 * kernel_radius + 1
    # Create a coordinate grid centred at 0.
    ax = np.arange(-kernel_radius, kernel_radius + 1)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma_voxel**2))
    kernel /= np.sum(kernel)  # Normalize the kernel to sum to 1.
    return kernel

def gaussian_kernel_3d_preserve_max(sigma_voxel):
    """
    Create a centred 3D Gaussian kernel with standard deviation sigma_voxel.
    Instead of normalising the kernel to sum to 1, it is normalised so that
    the central (peak) value is 1. This means that convolving a delta
    (i.e. a single-voxel spike) will yield a Gaussian with the same maximum value.
    The kernel size covers ±3σ.
    """
    kernel_radius = int(np.ceil(3 * sigma_voxel))
    ax = np.arange(-kernel_radius, kernel_radius + 1)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma_voxel**2))
    # Normalize so that the centre (at index [kernel_radius, kernel_radius, kernel_radius]) equals 1.
    kernel /= kernel[kernel_radius, kernel_radius, kernel_radius]
    return kernel

##############################################
# Modified Protein class and parser function
##############################################

@dataclass
class Protein:
    # Each atom is stored as an entry.
    # Shape: (num_atoms, 1, 3)
    atom_positions: np.ndarray
    # Mask: (num_atoms, 1) with 1 if the atom is present.
    atom_mask: np.ndarray

def get_protein_from_file_path(file_path: str, chain_id: str = None) -> Protein:
    """
    Parses a CIF (or PDB) file using Bio.PDB and returns a Protein object.
    This version includes all atoms.
    """
    ext = file_path.split(".")[-1].lower()
    if ext == "pdb":
        parser = PDBParser(QUIET=True)
    elif ext == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise RuntimeError("Unsupported file extension: " + ext)
    structure = parser.get_structure("protein", file_path)
    coords = []
    # Iterate over models, chains, residues and atoms.
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for residue in chain:
                # Do not filter out any residues.
                for atom in residue:
                    coords.append(atom.get_coord())
    coords = np.array(coords)
    if coords.size == 0:
        raise ValueError("No atom coordinates were found in the file.")
    # Reshape so that each atom is one entry.
    atom_positions = coords.reshape(-1, 1, 3)
    atom_mask = np.ones((coords.shape[0], 1))
    return Protein(atom_positions=atom_positions, atom_mask=atom_mask)

##############################################
# Comparison map generation code
##############################################

def generate_comparison_atom_map(automatic_model_file: str, manual_model_file: str,locres_file: str,
                                 ref_mrc_file: str, output_mrc_file: str,
                                 search_range_angstrom: float,
                                 sigma_blur_A: float = 0.5,
                                 exact_match_tol: float = 0.5):
    """
    For each atom in the manual model, compute the minimum distance to any atom in the
    automatic model. If this distance is greater than search_range_angstrom, assign 0.
    If it is less than or equal to exact_match_tol, assign 2, and otherwise assign 1.
    The resulting comparison map is then blurred using a Gaussian filter with sigma
    equal to search_range_angstrom (converted to voxel units).
    """
    # Load the reference MRC file.
    mrc_obj = load_mrc(ref_mrc_file)
    grid_shape = mrc_obj.grid.shape  # (nz, ny, nx)
    voxel_size = mrc_obj.voxel_size    # Assumed isotropic (scalar)
    global_origin = mrc_obj.global_origin
    locres_map=load_mrc(locres_file)

    # Flip the global origin for voxel index calculation.
    flipped_origin = np.flip(global_origin, axis=-1)

    # Load protein models.
    auto_protein = get_protein_from_file_path(automatic_model_file)
    manual_protein = get_protein_from_file_path(manual_model_file)

    # Flatten atom coordinate arrays.
    auto_coords = auto_protein.atom_positions.reshape(-1, 3)
    manual_coords = manual_protein.atom_positions.reshape(-1, 3)

    # Create an empty comparison map using a float grid.
    comparison_map = np.zeros(grid_shape, dtype=np.float32)

    # Loop over each manual atom.
    for manual_atom in manual_coords:
        # Compute distances from the manual atom to every automatic atom.
        distances = np.linalg.norm(auto_coords - manual_atom, axis=1)
        min_distance = np.min(distances)
        # Apply thresholds:
        # - If min_distance > search_range_angstrom, mark as 0.
        # - If min_distance <= exact_match_tol, mark as 2.
        # - Otherwise, mark as 1.
        if min_distance > search_range_angstrom:
            value = 0.15
        elif min_distance <= exact_match_tol:
            #value = 1.0
            # Interpolate: 1.0 at d = 0 and 0.9 at d = exact_match_tol.
            value = 1.0 - (min_distance / exact_match_tol) * 0.1
        else:
            #value = 0.5
            # Interpolate: 0.8 at d = exact_match_tol and 0.4 at d = search_range_angstrom.
            value = 0.8 - ((min_distance - exact_match_tol) / (search_range_angstrom - exact_match_tol)) * 0.4

        # Compute the voxel index from the manual atom coordinate.
        flipped_atom = np.flip(manual_atom, axis=-1)
        idx = np.floor((flipped_atom - flipped_origin) / voxel_size).astype(int)
        if np.all(idx >= 0) and np.all(idx < grid_shape):
            # If multiple manual atoms fall into the same voxel, retain the highest value.
            current_value = comparison_map[tuple(idx)]
            if value > current_value:
                comparison_map[tuple(idx)] = value

    # Convert the search range from angstroms to voxel units.
    sigma_voxel = sigma_blur_A / voxel_size
    #kernel = gaussian_kernel_3d(sigma_voxel)
    
    if sigma_voxel>0:
        #kernel = gaussian_kernel_3d_preserve_max(sigma_voxel)
        kernel = gaussian_kernel_3d(sigma_voxel)
        # Apply a Gaussian blur to the comparison map.
        #blurred_map = gaussian_filter(comparison_map, sigma=sigma_voxel)
        #blurred_map = convolve(comparison_map, kernel, mode='same')
        orig_max = np.max(comparison_map)
        orig_min = np.min(comparison_map)
        comparison_map = convolve(comparison_map, kernel, mode='same')
        result_max = np.max(comparison_map)
        result_min = np.min(comparison_map)
        if orig_min < orig_max and result_min < result_max:
            comparison_map = (comparison_map-result_min)*(orig_max-orig_min)/(result_max-result_min)
            comparison_map = np.clip(comparison_map, orig_min, orig_max)
            #print("normalize")
    save_mrc_image(output_mrc_file, comparison_map, original_header_file_to_copy=ref_mrc_file)
    #print(f"Comparison atom map saved to {output_mrc_file}")

##############################################
# Main execution block
##############################################

def main():
    parser = argparse.ArgumentParser(
        description="This software computes a modelability score map by comparing atomic positions between an automatically generated model and a manually curated reference model. For each atom in the reference model, the minimum distance to any corresponding atom in the automatic model is determined and converted into a score via piecewise linear interpolation. In the resulting MRC image, a score of 1 indicates close correspondence between the models, implying high agreement and accuracy, whereas lower scores reflect larger discrepancies. These discrepancies may result from increased structural flexibility, where atoms deviate more due to inherent variability, or from insufficient resolution, as high resolution is required not only to predict atomic positions but also to achieve precise placement. Consequently, the modelability score map indirectly indicates regions where the data supports accurate predictions and areas where either flexibility or limited resolution may hinder precise modelling."
    )
    parser.add_argument(
        '--map',
        required=True,
        help="Path to the reference MRC file (map)"
    )
    parser.add_argument(
        '--ref_model',
        required=True,
        help="Path to the manual model file"
    )
    parser.add_argument(
        '--auto_model',
        required=True,
        help="Path to the automatic model file"
    )
    parser.add_argument(
        '--o',
        required=True,
        help="Output file name for the modelability score map"
    )
    parser.add_argument(
        '--search_range',
        type=float,
        default=5.0,
        help="Search range in angstroms (default: 5.0)"
    )
    parser.add_argument(
        '--smooth',
        type=float,
        default=1.0,
        help="Sigma blur value (default: 1.0)"
    )
    parser.add_argument(
        '--locres',
        required=True,
        help="Path to the locres map"
    args = parser.parse_args()
    
    generate_comparison_atom_map(
        automatic_model_file=args.auto_model,
        manual_model_file=args.ref_model,
        locres_file=args.locres,
        ref_mrc_file=args.map,
        output_mrc_file=args.o,
        search_range_angstrom=args.search_range,
        sigma_blur_A=args.smooth
    )

if __name__ == "__main__":
    main()
