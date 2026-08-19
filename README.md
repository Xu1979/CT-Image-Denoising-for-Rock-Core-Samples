# Micro-CT Image Denoising for Rock Core Samples

Deep learning and traditional methods for denoising CT scan images of geological rock samples (sandstone and carbonate). Handles three noise types: Salt-and-Pepper (SAP), ring artifacts, and beam hardening (BH), individually and in combination.

## Paper

> [Micro-CT Image Denoising of Rock Sample Using Deep Learning]  
> Jiang  
> *[Journal Name]*, 2026  
> [DOI / Link — to be added upon acceptance]

## Models

| Model | Type | File |
|---|---|---|
| U-Net | Deep learning | `unet_complete.py` |
| FCN-ResNet50 / ResNet | Deep learning (pretrained) | `FCN_complete.py` |
| Hybrid Swin Transformer | Deep learning (pretrained) | `SwinUNet_complete.py` |
| Gaussian / Median / Mean / NLM / SNN | Traditional (SAP) | `traditional_denoising_pipeline.py` |
| Ring artifact correction (RHC) | Traditional | `traditional_denoising_pipeline.py` |
| K-means-assisted beam hardening correction (BHC) | Traditional | `traditional_denoising_pipeline.py` |

> **Hybrid Swin Transformer** is a custom architecture combining a pretrained Swin Transformer encoder (`swin_base_patch4_window7_224`) with a lightweight CNN decoder. It is distinct from the original SwinUNet (Cao et al., 2021), which uses a pure Transformer for both encoder and decoder. In this repository, "SwinUNet" refers to this custom hybrid implementation rather than the original pure-Transformer Swin-Unet architecture.

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

> **PyTorch**: Install the CUDA-compatible version for your system from  
> https://pytorch.org/get-started/locally/  
> then install the remaining packages via `pip install -r requirements.txt`.

**Pretrained weights for SwinUNet** (`swin_base_patch4_window7_224.pth`):  
Download from the [Swin Transformer repository](https://github.com/microsoft/Swin-Transformer) and place at:
```
pretrained/swin_base_patch4_window7_224.pth
```

## Repository Structure

```
├── dataset_pre.py                   # Dataset generation and DataLoader
├── unet_complete.py                 # U-Net training
├── FCN_complete.py                  # FCN-ResNet training
├── SwinUNet_complete.py             # SwinUNet training
├── traditional_denoising_pipeline.py# Traditional denoising pipeline
├── history_visual_prediction.py     # Training curve plots + patch-wise inference
├── Generate_multiversion_noise.py   # Generate multi-version noisy images
├── direct_real_image_prediction.py  # Deep-learning/traditional inference on real images
├── minkowski_functionals_2d.py      # Minkowski analysis for synthetic-noise experiments
├── downsample_tomo.py               # ×2 downsampling and segmentation comparison
├── requirements.txt
└── LICENSE
```

## Usage

### Step 1 — Prepare dataset

Place raw CT scan files (`.nc` / `.tiff`) in an `input/` directory, then run:

```bash
python dataset_pre.py
```

This generates a `dataset/` directory with `train/` and `val/` splits (80/20), containing noisy/clean image pairs.

The noise-generation script supports two experiment modes. Select the mode by
editing `RUN_MODE` in `Generate_multiversion_noise.py`:

```python
RUN_MODE = "fixed_slice"
```

`fixed_slice` generates multiple noise types and intensity versions from one
fixed clean slice. Configure `INPUT_PATH` and `OUTPUT_DIR` before running.

```python
RUN_MODE = "generalization"
```

`generalization` applies the experiment to multiple clean slices. Configure
`GENERALIZATION_INPUT_DIR` and `GENERALIZATION_OUTPUT_DIR` before running. For
each slice and intensity version, the recorded parameters for an individual
noise type are reused when that same noise appears in a combined-noise image.

Run the selected mode with:

```bash
python Generate_multiversion_noise.py
```

### Step 2 — Train a model

```bash
# U-Net
python unet_complete.py

# FCN-ResNet50
python FCN_complete.py

# SwinUNet (requires pretrained Swin weights)
python SwinUNet_complete.py

```

Training automatically resumes from `*_latest_checkpoint.pth` if it exists.  
Best model is saved as `*_best_model.pth`.

### Step 3 — Run inference and plot training curves

```bash
python history_visual_prediction.py
```

Edit the `__main__` block to select the model, input directory, and output directory.  
Outputs denoised images and a CSV of PSNR / SSIM results.

### Step 4 — Run traditional denoising

```bash
python traditional_denoising_pipeline.py
```

Edit `SANDSTONE_ROOT` and `CARBONATE_ROOT` at the top of the file to point to your data.

### Step 5 — Denoise real images

Real images can be processed using either the deep-learning models or the
traditional correction methods:

```bash
python direct_real_image_prediction.py
```

Before running, edit the configuration section in
`direct_real_image_prediction.py`:

- `INPUT_DIR` can point to either a directory or one specific image.
- Set `PREDICTION_MODE = "deep_learning"` to run U-Net, FCN-ResNet50 and the
  Hybrid Swin Transformer.
- Set `PREDICTION_MODE = "traditional"` and configure
  `TRADITIONAL_NOISE_TYPES` to run the corresponding traditional method.

Supported input formats include PNG, TIFF, JPEG, BMP and WebP. WebP inputs and
their processed outputs are saved as PNG. Results are written under
`real_image_direct_denoised/`, with one subdirectory per method.

### Step 6 — Segment the real-image results in ImageJ

Segment the original and denoised images using the same ImageJ workflow. The
current experiments use ImageJ with Advanced/Trainable Weka Segmentation.
Export the segmentation results while retaining the original image name, for
example:

```text
BHG0_z0368_tomo_Simple Segmentation_visual.tif
```

Place the original-image segmentations in:

```text
real_image/imageJ_segmentation_result/
```

Place the model segmentations in the corresponding method directories:

```text
real_image_direct_denoised/
├── fcn_resnet50/imageJ_segmentation_result/
├── swinunet/imageJ_segmentation_result/
└── unet/imageJ_segmentation_result/
```

Traditional-method segmentations remain in `traditional_bh/` or
`traditional_ring/`. For binary segmentation images, black (`0`) represents
the pore phase and white (`255`) represents the solid matrix.

### Step 7 — Calculate 2D Minkowski functionals

For the synthetic-noise/multi-slice experiments, arrange the segmented TIFF
files under `output_test/minkowski_functionals/`, then run:

```bash
python minkowski_functionals_2d.py
```

The script calculates three structural descriptors:

- porosity;
- pore–matrix perimeter density (`pixel^-1`);
- Euler density (`pixel^-2`).

Sandstone images (cylindrical samples) use the configured oval specimen ROI, while carbonate images (rectangular samples)
are evaluated over the full image. Per-image results, mean and standard
deviation, relative errors against the clean segmentation, and comparison
figures are saved in:

```text
output_test/minkowski_functionals/results/
```

### Step 8 — Run the ×2 downsampling experiment

The downsampling experiment compares segmentation performed on a high-
resolution image with segmentation performed after ×2 downsampling and then
upsampled to the HR grid. First, prepare the aligned HR and ×2 images and the
segmentation directories:

```bash
python downsample_tomo.py
```

Next, segment both the HR and ×2 images using Otsu, MidGrey and Weka in ImageJ, and place
the label images in the corresponding directories prepared by the script.
Then upsample the completed ×2 label maps to the aligned HR grid:

```bash
python downsample_tomo.py --upsample-labels
```

Finally, calculate the pixel-wise agreement and structural deviations:

```bash
python downsample_tomo.py --compare-pixelwise --compare-structural
```

The output includes Dice, IoU, precision, recall, pixel accuracy, and the
deviations in porosity, perimeter density and Euler density. The HR
segmentation is a method-specific resolution-consistency reference, rather
than an independent manually annotated ground truth.


## Noise Types

| Code | Description |
|---|---|
| `sap_only` | Salt-and-Pepper noise only |
| `ring_only` | Ring artifact only |
| `bh_only` | Beam hardening only |
| `sap_ring` | SAP + Ring |
| `sap_bh` | SAP + Beam hardening |
| `ring_bh` | Ring + Beam hardening |
| `sap_ring_bh` | All three combined |

## Quick Test

Download the sample test images and pretrained models:

Place the downloaded files in the project root, then run:

```bash
python quick_test.py
```

By default this runs U-Net. To switch models, open `quick_test.py` and uncomment the desired model block (Hybrid Swin Transformer or FCN-ResNet50).

Expected output:
```
Model : unet  (unet_best_model.pth)
Input : test_data/noisy
Output: test_data/denoised/unet

Average PSNR: XX.XX ± X.XX dB
Average SSIM: 0.XXXX ± 0.XXXX
✓ all the denoised images have been saved: test_data/denoised/unet
```

## Metrics

All models are evaluated with **PSNR** (dB) and **SSIM** on the validation set.  
Results are saved automatically as JSON (training history) and CSV (per-image inference).

Structural analysis of the segmented images additionally reports:

- **Porosity** (dimensionless pore-area fraction)
- **Perimeter density** (`pixel^-1`)
- **Euler density** (`pixel^-2`)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
