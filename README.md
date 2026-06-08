# deepTools

A Python package for cryo-EM map processing with deep learning. Includes tools for local resolution estimation, denoising, sharpening, mask prediction, missing-wedge restoration, and more.

## Installation

### Requirements

- Python >= 3.8
- PyTorch 2.0
- CUDA-capable GPU (recommended; CPU fallback available)

### Install from source

```bash
git clone https://github.com/mauromaiorca/deepTools.git
cd deepTools
pip install -e .
```

This installs the package in editable mode along with all dependencies listed in `setup.py`.

### Install from GitHub directly

```bash
pip install git+https://github.com/mauromaiorca/deepTools.git
```

### Verify the installation

```bash
python -m deepTools.infer --help
```

## Usage

### Inference

The main inference entry point supports multiple modes configured via `infer_config.json`.

```bash
# Local resolution estimation (default mode)
python -m deepTools.infer --mode locres --map input.mrc --o output_locres.mrc

# Mask prediction
python -m deepTools.infer --mode mask --map input.mrc --o output_mask.mrc

# Denoising
python -m deepTools.infer --mode denoise --map input.mrc --o output_denoised.mrc

# Sharpening
python -m deepTools.infer --mode sharp --map input.mrc --o output_sharp.mrc
```

#### Missing-wedge restoration

Restore incomplete angular sampling artefacts from cryo-ET maps using a pre-trained model.

```bash
# Using the default restoreWedge model (mapSharp_47/learning_model_mapSharp_47.pth)
python -m deepTools.infer --restoreWedge --map input_missing_wedge.mrc --o restored.mrc

# Equivalent using --mode
python -m deepTools.infer --mode restoreWedge --map input_missing_wedge.mrc --o restored.mrc

# Using a custom model file
python -m deepTools.infer --restoreWedge --model /path/to/test_model.pth --map input_missing_wedge.mrc --o restored.mrc
```

#### Additional inference options

| Option | Description |
|---|---|
| `--cpu` | Force CPU inference even if a GPU is available |
| `--bin N` | Downsample input by factor N before inference, then upsample back |
| `--patch` | Enable patch-based inference to reduce memory usage |
| `--config PATH` | Use a custom inference configuration file |
| `--model PATH` | Override the model file for any mode |

### Map processing

Apply various operations to MRC density maps (smoothing, thresholding, erosion, dilation, rotation, masking, etc.).

```bash
python -m deepTools.map_processing --i input.mrc --o output.mrc --smooth 2.0
python -m deepTools.map_processing --i input.mrc --o output.mrc --threshold 0.5
```

### Model processing

Process 3D MRC density maps with smoothing, morphological operations, Fourier amplitude manipulation, masking, and more.

```bash
python -m deepTools.model_processing --i input.mrc --o output.mrc --smooth 1.5
```

### Training

Train models using a configuration file and training data.

```bash
python -m deepTools.train \
    --config train_config.json \
    --model_file model.pth \
    --disc_file discriminator.pth \
    --training_data training_data.json
```

## CLI entry points

After installation, the following commands are available:

| Command | Description |
|---|---|
| `deepTools` | Run inference (`deepTools.infer:main`) |
| `deepTools_train` | Train a model (`deepTools.train:main`) |
| `deepTools_map_process` | Map processing utilities (`deepTools.map_processing:main`) |
| `deepTools_model_process` | Model processing utilities (`deepTools.model_processing:main`) |

## Configuration

Inference modes are configured via `infer_config.json`, which maps mode names to model paths and pre/post-processing steps. Place this file in the package directory or specify its location with `--config`.

## License

See repository for license information.
