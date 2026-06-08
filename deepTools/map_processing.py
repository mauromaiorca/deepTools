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
    return np.real(ifftn(new_fft))


def flat_fourier_amplitudes(input_image):
    input_fft = fftn(input_image)
    phase = np.angle(input_fft)
    return np.real(ifftn(np.exp(1j * phase)))


def compute_masked_mean(input_image, mask):
    output = np.zeros_like(input_image)
    for val in np.unique(mask):
        region = mask == val
        output[region] = np.mean(input_image[region])
    return output


def create_sphere_mask(shape):
    center = np.array(shape) // 2
    grid = np.stack(np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij"
    ), axis=-1)
    dist = np.linalg.norm(grid - center, axis=-1)
    radius = min(shape) / 2
    return (dist <= radius).astype(np.float32)


def sdlevel_threshold(vol, level):
    mean, std = np.mean(vol), np.std(vol)
    return (vol >= mean + level * std).astype(np.float32)


def add_masked_noise(image, mask):
    mask = np.clip(mask, 0, 1)
    region = image * mask
    minv = np.min(region[mask > 0])
    stdv = np.std(region[mask > 0]) * np.random.uniform(0.9, 2.0)
    noise = np.random.normal(minv, stdv, image.shape)
    return noise * (1 - mask) + image * mask


def add_gaussian_noise(image, sigma=0.1):
    noise = np.random.normal(0, sigma, image.shape)
    return image + noise


def standard_normalize(vol, epsilon=1e-8):
    mean, std = np.mean(vol), np.std(vol)
    if std < epsilon:
        return vol - mean
    return (vol - mean) / std


def apply_masked_resolution(image, mask, worst=60.0):
    mask = np.clip(mask, 0, 1)
    return mask * image + (1 - mask) * worst


def apply_masked_crop(image, mask):
    idx = np.argwhere(mask > 0)
    mins, maxs = idx.min(0), idx.max(0) + 1
    return image[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]


def delete_dust(image, min_size):
    struct = generate_binary_structure(3, 1)
    labeled, count = label(image > 0, structure=struct)
    out = np.zeros_like(image)
    for i in range(1, count + 1):
        comp = labeled == i
        if comp.sum() >= min_size:
            out[comp] = image[comp]
    return out


def rotate_3d_image(image, phi, theta, psi):
    out = rotate(image, phi, axes=(1,0), reshape=False, order=1)
    out = rotate(out, theta, axes=(2,0), reshape=False, order=1)
    return rotate(out, psi, axes=(2,1), reshape=False, order=1)


def process_operation(vol, op, val, args):
    if op == "erode":
        return binary_erosion(vol>0, generate_binary_structure(3,val)).astype(np.float32)
    if op == "dilate":
        return binary_dilation(vol>0, generate_binary_structure(3,val)).astype(np.float32)
    if op == "close":
        return binary_closing(vol>0, generate_binary_structure(3,val)).astype(np.float32)
    if op == "open":
        return binary_opening(vol>0, generate_binary_structure(3,val)).astype(np.float32)
    if op == "threshold":
        return (vol>=val).astype(np.float32)
    if op == "amplReplace":
        amp = load_mrc_image(val)
        return replace_fourier_amplitudes(vol, amp)
    if op == "amplFlat":
        return flat_fourier_amplitudes(vol)
    if op == "masked_resolution":
        return apply_masked_resolution(vol, *val)
    if op == "masked_crop":
        return apply_masked_crop(vol, load_mrc_image(val))
    if op == "add_masked_noise":
        return add_masked_noise(vol, load_mrc_image(val))
    if op == "add_gaussian_noise":
        sigma = float(val) if val is not None else 0.1
        return add_gaussian_noise(vol, sigma)
    if op == "normalize":
        return standard_normalize(vol)
    if op == "delete_dust":
        return delete_dust(vol, val)
    if op == "rotate":
        return rotate_3d_image(vol, *val)
    if op == "create_sphere_mask":
        return create_sphere_mask(vol.shape)
    if op == "compute_masked_mean":
        return compute_masked_mean(vol, load_mrc_image(val))
    if op == "smooth":
        return gaussian_filter(vol, sigma=float(val))
    if op == "sdlevel":
        return sdlevel_threshold(vol, val)
    if op == "divide_by_pixel_spacing":
        return vol / args.pixel_spacing
    if op == "multiply_by_pixel_spacing":
        return vol * args.pixel_spacing
    if op == "invert_density":
        return -vol
    if op == "reciprocal_density":
        max_value=60.0
        min_density = 1.0 / max_value
        safe_vol = np.where(vol < min_density, min_density, vol)
        return 1.0 / safe_vol
    if op == "invert_resolution":
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(vol<0.1,0,1.0/vol)
    if op == "replace_density_range":
        # val will now be a list [min_target, max_target, new_value]
        min_target, max_target, new_value = val
        out = vol.copy()
        mask = (out >= min_target) & (out <= max_target)
        out[mask] = new_value
        return out
    return vol


def process_map(vol, args, argv=None):
    argv = argv or sys.argv[1:]
    ops=[]
    ad=vars(args)
    i=0
    while i<len(argv):
        a=argv[i]
        if a.startswith("--"):
            k=a.lstrip("--")
            v=ad.get(k)
            if v is not None:
                if isinstance(v,bool) and v:
                    ops.append((k,None))
                elif isinstance(v,list):
                    for x in v: ops.append((k,x))
                elif i+1<len(argv) and not argv[i+1].startswith("--"):
                    ops.append((k,argv[i+1]));i+=1
        i+=1
    for op,val in ops:
        print(f"Applying: {op} val={val}")
        vol=process_operation(vol,op,val,args)
    return vol


def setup_parser(require_io=False):
    p=argparse.ArgumentParser(description="3D MRC processor")
    p.add_argument("--i",required=require_io)
    p.add_argument("--o",required=require_io)
    p.add_argument("--i_json",nargs=3)
    p.add_argument("--smooth",type=float,action='append')
    p.add_argument("--erode",type=int,action='append')
    p.add_argument("--dilate",type=int,action='append')
    p.add_argument("--close",type=int,action='append')
    p.add_argument("--threshold",type=float,action='append')
    p.add_argument("--sdlevel",type=float,action='append')
    p.add_argument("--amplReplace",metavar="AMPL_FILE")
    p.add_argument("--amplFlat",action='store_true')
    p.add_argument("--masked_resolution",nargs=2)
    p.add_argument("--masked_crop")
    p.add_argument("--add_masked_noise")
    p.add_argument("--add_gaussian_noise",nargs='?',const=0.1,type=float)
    p.add_argument("--normalize",action='store_true')
    p.add_argument("--delete_dust",type=int,action='append')
    p.add_argument("--rotate",nargs=3,type=float,action='append')
    p.add_argument("--create_sphere_mask",action='store_true')
    p.add_argument("--compute_masked_mean")
    p.add_argument("--invert_density",action='store_true')
    p.add_argument("--reciprocal_density",action='store_true')
    p.add_argument("--divide_by_pixel_spacing",action='store_true')
    p.add_argument("--multiply_by_pixel_spacing",action='store_true')
    p.add_argument("--invert_resolution",action='store_true')
    p.add_argument(
        "--replace_density_range",
        nargs=3,
        type=float,
        action='append',
        metavar=("MIN", "MAX", "NEW"),
        help="Replace voxel values in [MIN,MAX] with NEW"
    )
    return p

# Inline map processing for API usage

def inline_map_processing(input_map, arg_string, pixel_spacing=None):
    parser = setup_parser(require_io=False)
    arg_list = arg_string.split()
    args = parser.parse_args(arg_list)
    if pixel_spacing is not None:
        args.pixel_spacing = pixel_spacing
    return process_map(input_map, args, argv=arg_list)

# Main entry-point

def main():
    parser = setup_parser(require_io=False)
    args = parser.parse_args()
    if args.i:
        vol = load_mrc_image(args.i)
        args.pixel_spacing = get_pixel_spacing(args.i)
        out = process_map(vol, args)
        save_mrc_image(args.o, out, args.i)
    elif args.i_json:
        jf, key, suf = args.i_json
        data = json.load(open(jf))
        for entry in data:
            f = entry[key]
            vol = load_mrc_image(f)
            out = process_map(vol, args)
            base = f[:-4] if f.lower().endswith('.mrc') else f
            outp = f"{base}_{suf}.mrc"
            save_mrc_image(outp, out, f)
            newk = f"{key}_{suf}" if not key.endswith('_file') else key.replace('_file', f"_{suf}_file")
            entry[newk] = outp
        json.dump(data, open(args.o, 'w'), indent=4)
        print(f"Saved {args.o}")

if __name__ == "__main__":
    main()
