"""
Quick Test Script
-----------------
Tests a pretrained denoising model on sample images across multiple noise types.

Directory structure required:
    test_data/
        sap_only/      <- 30 noisy images (v00.png ... v29.png)
        ring_only/
        bh_only/
        sap_ring/
        sap_bh/
        ring_bh/
        sap_ring_bh/
        clean/
            clean.png  <- single clean reference image (for PSNR/SSIM)

Usage:
    1. Select a model by uncommenting the corresponding block below.
    2. Run:  python quick_test.py

Denoised images are saved to test_data/denoised/{model_name}/{noise_type}/.
A summary CSV is saved to test_data/denoised/{model_name}_summary.csv.
"""

import os
import pandas as pd
from history_visual_prediction import test_model_on_images

# ============================================================
# Configuration — uncomment ONE model block to test
# ============================================================

# --- U-Net ---
MODEL_NAME = 'unet'
MODEL_PATH = 'unet_best_model.pth'
PATCH_SIZE = 256
STRIDE     = 64

# --- Hybrid Swin Transformer ---
# MODEL_NAME = 'swinunet'
# MODEL_PATH = 'swinunet_best_denoising_model.pth'
# PATCH_SIZE = 224
# STRIDE     = 56

# --- FCN-ResNet50 ---
# MODEL_NAME = 'fcn_resnet50'
# MODEL_PATH = 'fcn_best_model.pth'
# PATCH_SIZE = 256
# STRIDE     = 64

# ============================================================
# Paths
# ============================================================

TEST_ROOT    = 'test_data'
CLEAN_DIR    = os.path.join(TEST_ROOT, 'clean')
MARGIN_RATIO = 0.3
NORMALIZE    = True

NOISE_FOLDERS = [
    'sap_only',
    'ring_only',
    'bh_only',
    'sap_ring',
    'sap_bh',
    'ring_bh',
    'sap_ring_bh',
]

# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    # Sanity checks
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            f"Download pretrained models from the Releases page and place "
            f"them in the project root."
        )
    if not os.path.isdir(TEST_ROOT):
        raise FileNotFoundError(
            f"Test data directory not found: {TEST_ROOT}\n"
            f"Download test_data/ from the Releases page and place it in "
            f"the project root."
        )

    print(f"Model : {MODEL_NAME}  ({MODEL_PATH})\n")

    all_records = []

    for folder in NOISE_FOLDERS:
        noisy_dir  = os.path.join(TEST_ROOT, folder)
        output_dir = os.path.join(TEST_ROOT, 'denoised', MODEL_NAME, folder)

        if not os.path.isdir(noisy_dir):
            print(f"[SKIP] Not found: {noisy_dir}")
            continue

        print(f"{'=' * 60}")
        print(f"Noise type: {folder}")
        print(f"{'=' * 60}")

        psnr_list, ssim_list = test_model_on_images(
            model_path=MODEL_PATH,
            test_noisy_dir=noisy_dir,
            test_clean_dir=CLEAN_DIR,
            output_dir=output_dir,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            margin_ratio=MARGIN_RATIO,
            model_name=MODEL_NAME,
            normalize=NORMALIZE,
            show_progress=True
        )

        # Collect per-image records
        img_files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))
        ])
        for fname, psnr, ssim in zip(img_files, psnr_list, ssim_list):
            all_records.append({
                'noise_type': folder,
                'image':      fname,
                'PSNR':       round(psnr, 4),
                'SSIM':       round(ssim, 4),
            })

    # Save summary CSV
    if all_records:
        df = pd.DataFrame(all_records)

        summary_rows = []
        for folder in NOISE_FOLDERS:
            sub = df[df['noise_type'] == folder]
            if len(sub) > 0:
                summary_rows.append({
                    'noise_type': folder,
                    'image':      'MEAN ± STD',
                    'PSNR':       f"{sub['PSNR'].mean():.2f} ± {sub['PSNR'].std():.2f}",
                    'SSIM':       f"{sub['SSIM'].mean():.4f} ± {sub['SSIM'].std():.4f}",
                })

        df_summary = pd.DataFrame(summary_rows)
        df_all = pd.concat([df, df_summary], ignore_index=True)

        csv_path = os.path.join(TEST_ROOT, 'denoised', f'{MODEL_NAME}_results.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_all.to_csv(csv_path, index=False)

        print(f"\n{'=' * 60}")
        print(f"Summary — {MODEL_NAME}")
        print(f"{'=' * 60}")
        print(df_summary.to_string(index=False))
        print(f"\nFull results saved: {csv_path}")
        print("Done!")