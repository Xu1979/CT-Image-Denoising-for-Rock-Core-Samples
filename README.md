# Micro-CT Image Denoising for Rock Core Samples

Deep learning and traditional methods for denoising CT scan images of geological rock samples (sandstone and carbonate). Handles three noise types: Salt-and-Pepper (SAP), ring artifacts, and beam hardening (BH), individually and in combination.

## Paper

> [Paper Title]  
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

## Requirements

- Python 3.9+
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
├── requirements.txt
└── LICENSE
```

## Usage

### Step 1 — Prepare dataset

Place raw CT scan files (`.nc` / `.tiff`) in an `input/` directory, then run:

```bash
python dataset_pre.py
```

This generates a `dataset/` directory with `train/` and `val/` splits (80/20), containing noisy/clean image patch pairs.

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

## Metrics

All models are evaluated with **PSNR** (dB) and **SSIM** on the validation set.  
Results are saved automatically as JSON (training history) and CSV (per-image inference).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
