#!/usr/bin/env python

import argparse
import torch
import numpy as np
import mrcfile
import os
import json
import torch.nn.functional as F  # For pooling and interpolation
from .utils import load_mrc_file
from .model import UNetWithAttention
from .map_processing import inline_map_processing
from .utils import read_mrc_header, save_mrc_image, load_mrc_file, get_pixel_spacing


DEFAULT_RESTORE_WEDGE_MODEL = os.path.join("mapSharp_47", "learning_model_mapSharp_47.pth")
USER_CONFIG_PATH = os.path.expanduser("~/.deeptools_config.json")


def get_models_dir(cli_models_dir=None):
    """Return the models base directory from CLI, env var, or user config."""
    if cli_models_dir:
        return os.path.abspath(os.path.expanduser(cli_models_dir))
    env = os.environ.get("DEEPTOOLS_MODELS_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if os.path.exists(USER_CONFIG_PATH):
        try:
            with open(USER_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            d = cfg.get("models_dir")
            if d:
                return os.path.abspath(os.path.expanduser(d))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def resolve_existing_path(path, config_path=None, models_dir=None):
    """Resolve a model/config path from common runtime locations.

    Search order for relative paths:
      1. models_dir (from --models_dir, $DEEPTOOLS_MODELS_DIR, or ~/.deeptools_config.json)
      2. current working directory
      3. config file directory
      4. package directory
      5. parent of the package directory
    The first existing path is returned. If none exists, the original path is
    returned so the caller can report a clear error.
    """
    if not path:
        return path

    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        if models_dir:
            basename = os.path.basename(path)
            candidates.append(os.path.abspath(os.path.join(models_dir, path)))
            candidates.append(os.path.abspath(os.path.join(models_dir, basename)))
        candidates.append(os.path.abspath(path))
        if config_path:
            candidates.append(os.path.abspath(os.path.join(os.path.dirname(config_path), path)))
        package_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.abspath(os.path.join(package_dir, path)))
        candidates.append(os.path.abspath(os.path.join(package_dir, "..", path)))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return path


def load_state_dict_robust(model, model_path, device):
    """Load a PyTorch state dict, handling DataParallel 'module.' prefixes
    and checkpoints wrapped in 'state_dict' or 'model_state_dict' keys."""
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    model.load_state_dict(cleaned, strict=True)



def patch_inference_random(model, input_tensor, patch_size, minimal_count_per_pixel=1):
    """
    Perform inference on the input_tensor by randomly sampling patches.

    For each random patch, the model is applied and the inferred values are accumulated 
    in result_sum, while a count tensor is updated to record the number of times each voxel 
    has been processed. The loop continues until every voxel has been covered at least 
    'minimal_count_per_pixel' times. The final result is computed as result_sum/count.

    Parameters:
      model: The neural network model.
      input_tensor: a 5D tensor of shape [1, 1, D, H, W].
      patch_size: integer specifying the side-length of the cubic patch.
      minimal_count_per_pixel: minimum number of times each voxel must be processed.

    Returns:
      A tensor with the same shape as input_tensor, representing the averaged inference.
    """
    device = input_tensor.device
    _, _, D, H, W = input_tensor.shape
    result_sum = torch.zeros_like(input_tensor)
    count = torch.zeros_like(input_tensor)
    iteration = 0
    # Continue until every voxel has been processed at least minimal_count_per_pixel times.
    while (count < minimal_count_per_pixel).any():
        # Randomly choose starting indices for the patch.
        z = torch.randint(0, D - patch_size + 1, (1,)).item()
        y = torch.randint(0, H - patch_size + 1, (1,)).item()
        x = torch.randint(0, W - patch_size + 1, (1,)).item()
        # Extract patch.
        patch = input_tensor[:, :, z:z+patch_size, y:y+patch_size, x:x+patch_size]
        # Run inference on the patch.
        out_patch = model(patch)
        # Accumulate the predicted values.
        result_sum[:, :, z:z+patch_size, y:y+patch_size, x:x+patch_size] += out_patch
        # Increment the count for these voxels.
        count[:, :, z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1
        iteration += 1
        if iteration % 100 == 0:
            remaining = (count < minimal_count_per_pixel).sum().item()
            print(f"Iteration {iteration}: {remaining} voxels have counts below {minimal_count_per_pixel}")
    # Compute the average over the overlapping patches.
    averaged_result = result_sum / count
    return averaged_result


def main():
    parser = argparse.ArgumentParser(
        description="Infer local resolution map from a cryo-EM map using a pre-trained model configured via infer_config.json."
    )
    parser.add_argument(
        "--mode", 
        required=False, 
        default="locres",
        choices=["locres", "locres_old", "mask", "maskT", "maskOksh", "denoise", "sharp", "tomoSharp", "tomoSharpOk", "simulate", "restoreWedge"],
        help="Select the inference mode to use (e.g. 'locres', 'mask' or 'restoreWedge')."
    )
    parser.add_argument(
        "--restoreWedge",
        action="store_true",
        help=(
            "Shortcut for --mode restoreWedge. By default this uses "
            "mapSharp_47/learning_model_mapSharp_47.pth unless --model is supplied."
        )
    )
    parser.add_argument(
        "--model", "--model_path",
        dest="model_path",
        default=None,
        help="Optional path to a .pth model file. This overrides the model selected by --mode/config."
    )
    parser.add_argument(
        "--map", 
        required=True,
        help="Path to the input cryo-EM map (MRC file)."
    )
    parser.add_argument(
        "--o", 
        required=True,
        help="Path to save the output local resolution map (MRC file)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the inference configuration file (defaults to infer_config.json in the package directory)."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if a GPU is available."
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=1,
        help="Binning factor. If >1, the input is downsampled by this factor before inference and then upsampled back."
    )
    parser.add_argument(
        "--patch",
        action="store_true",
        help="Enable patch-based inference: the input is processed in patches to reduce memory usage."
    )
    parser.add_argument(
        "--models_dir",
        default=None,
        help="Base directory for model files. Overrides DEEPTOOLS_MODELS_DIR and ~/.deeptools_config.json."
    )
    args = parser.parse_args()

    if args.restoreWedge:
        args.mode = "restoreWedge"

    models_dir = get_models_dir(args.models_dir)

    # Determine configuration file location.
    if args.config:
        config_path = args.config
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "infer_config.json")

    # Built-in fallback for the missing-wedge restoration model.
    # Other modes continue to use infer_config.json unless --model is provided.
    config_json = {
        "restoreWedge": {
            "model_path": DEFAULT_RESTORE_WEDGE_MODEL,
            "preprocessing": "",
            "postprocessing": ""
        }
    }

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_config = json.load(f)
        # Values in infer_config.json override built-in defaults when present.
        config_json.update(file_config)
    elif args.mode != "restoreWedge" and args.model_path is None:
        print(f"Configuration file not found: {config_path}")
        print("Use --config, choose --restoreWedge, or provide a model with --model / --model_path.")
        return

    if args.mode not in config_json:
        if args.model_path is None:
            print(f"Mode '{args.mode}' not found in configuration file.")
            return
        # Allow ad-hoc testing of any mode name with a user-supplied .pth.
        # No pre/post-processing is applied unless provided by infer_config.json.
        config_json[args.mode] = {
            "model_path": args.model_path,
            "preprocessing": "",
            "postprocessing": ""
        }

    mode_config = config_json[args.mode]
    model_path = args.model_path or mode_config.get("model_path")
    postprocessing = mode_config.get("postprocessing", "").strip()
    preprocessing = mode_config.get("preprocessing", "").strip()

    model_path = resolve_existing_path(model_path, config_path=config_path, models_dir=models_dir)

    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        if models_dir:
            print(f"Models directory: {models_dir}")
        print("To configure the models directory, run: deepTools_setup")
        print("Or use: --models_dir /path/to/dir  or  --model /path/to/model.pth")
        return

    # Set up device.
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetWithAttention(in_channels=1, out_channels=1)
    try:
        print("model path=",model_path)
        load_state_dict_robust(model, model_path, device)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    model.to(device)
    model.eval()

    # Load and preprocess the input map.
    cryo_em = load_mrc_file(args.map)
    if cryo_em is None:
        print(f"Error loading input map from {args.map}")
        return

    # Apply preprocessing if specified in the configuration.
    pixel_spacing = get_pixel_spacing(args.map)
    if preprocessing:
        print("Preprocessing:", preprocessing)
        cryo_em = inline_map_processing(cryo_em, preprocessing, pixel_spacing=pixel_spacing)

    # Normalize the data.
    cryo_em = (cryo_em - np.mean(cryo_em)) / np.std(cryo_em)
    cryo_em_tensor = torch.from_numpy(cryo_em).float().unsqueeze(0).unsqueeze(0).to(device)
    
    original_shape = cryo_em_tensor.shape  # [1, 1, D, H, W]
    _, _, D, H, W = original_shape

    # If binning is requested, downsample the input.
    bin_factor = args.bin
    if bin_factor > 1:
        new_shape = (max(1, D // bin_factor), max(1, H // bin_factor), max(1, W // bin_factor))
        cryo_em_tensor = F.adaptive_avg_pool3d(cryo_em_tensor, output_size=new_shape)
        print(f"Binned input shape from {(D, H, W)} to {new_shape}")

    # Perform inference.
    with torch.no_grad():
        if args.patch:
            patch_size = 64  # Default patch size; adjust as needed.
            predicted_tensor = patch_inference_random(model, cryo_em_tensor, patch_size)
        else:
            predicted_tensor = model(cryo_em_tensor)

    # If binning was applied, upsample the prediction back to the original dimensions.
    if bin_factor > 1:
        predicted_tensor = F.interpolate(predicted_tensor, size=(D, H, W), mode='trilinear', align_corners=False)
        print(f"Upsampled prediction back to original shape {(D, H, W)}")

    predicted_map = predicted_tensor.cpu().numpy()[0, 0]
    
    #with mrcfile.open(args.map, mode="r") as mrc:
    #	pixel_spacing = mrc.header.cella.x / mrc.header.nx


    # Apply postprocessing if specified.
    if len(postprocessing) > 2:
        print("postprocessing:", postprocessing)
        predicted_map = inline_map_processing(predicted_map, postprocessing, pixel_spacing=pixel_spacing)

    # Save the output map.
    try:
        save_mrc_image(args.o, predicted_map, args.map)
        print(f"Saved inferred map to {args.o}")
    except Exception as e:
        print(f"Error saving map to {args.o}: {e}")

if __name__ == "__main__":
    main()

