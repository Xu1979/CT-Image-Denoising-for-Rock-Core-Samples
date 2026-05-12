"""
Batch traditional denoising pipeline.
- Sandstone: all 7 noise folders
- Carbonate: sap_ring_bh only
- For SAP-containing combos: run 5 SAP filters (SNN, Gaussian, Median, Mean, NLM)
- BH and Ring: full-image processing with intensity calibration
- Results saved in-place + CSV output
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from skimage import exposure, morphology, measure
from sklearn.cluster import KMeans
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from skimage.restoration import denoise_nl_means, estimate_sigma

# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

SANDSTONE_ROOT = r"E:\sandstone"
CARBONATE_ROOT = r"E:\carbonate"

SANDSTONE_CLEAN = os.path.join(SANDSTONE_ROOT, "clean", "clean.png")
CARBONATE_CLEAN = os.path.join(CARBONATE_ROOT, "clean", "clean.png")

SANDSTONE_FOLDERS = [
    "sap_only", "ring_only", "bh_only",
    "sap_ring", "sap_bh", "ring_bh", "sap_ring_bh",
]
CARBONATE_FOLDERS = ["sap_ring_bh"]

PATCH_SIZE = 256
MARGIN_RATIO = 0.2
STRIDE = 128


# ══════════════════════════════════════════════════════════════════════
# Traditional Denoising Functions
# ══════════════════════════════════════════════════════════════════════

def beam_hardening_correction(img, cx=-1, cy=-1):
    img_initial = img.copy()
    rows, col = img.shape
    if cx < 0:
        cx = int(np.round(rows / 2))
    if cy < 0:
        cy = int(np.round(col / 2))

    imgt = np.ones((rows, col), dtype='u1')
    imgt[cx][cy] = 0
    img_edt = ndimage.distance_transform_edt(imgt)
    img_edt = np.round(img_edt).astype('u4')

    dist = list(np.unique(img_edt))
    ring = []

    num_b = np.min(np.array([cx, cy, rows - cx, col - cy]))
    num_b = int(0.414 * num_b * 0.2)
    boundary_std = 100000 * np.ones((num_b,), dtype='f8')
    boundary_num = 0
    for radius in dist:
        if np.sum(boundary_std) > 0:
            idx = np.flatnonzero(img_edt == radius)
            rcvalue = img.flat[idx]
            ring.append([radius, np.mean(rcvalue), np.std(rcvalue)])
            boundary_std[boundary_num % num_b] = np.std(rcvalue)
            boundary_num = boundary_num + 1
        else:
            ring = ring[:-num_b]
            break
    ring = np.array(ring)

    ring_feature = ring[:, 1:]
    kmeans = KMeans(n_clusters=2, n_init=10).fit(ring_feature)
    classes = kmeans.labels_

    radius = ring[:, 0]
    radius_img = radius.copy()
    radius_0 = radius[classes == 0]
    radius_1 = radius[classes == 1]
    if np.sum(radius_0[:100]) < np.sum(radius_1[:100]):
        center_round_radius = radius_0
    else:
        center_round_radius = radius_1
    center_round_radius = center_round_radius.astype('u4')
    imgt = np.zeros((9, np.max(center_round_radius) + 1), dtype='u1')
    imgt[4, center_round_radius] = 1
    imgt = morphology.binary_dilation(imgt, np.ones((3, 3), dtype='u1'))
    labels = measure.label(imgt, connectivity=2)
    ele = np.unique(labels)
    ele = list(ele[1:])
    size = 0
    for elec in ele:
        idx = np.flatnonzero(labels == elec)
        if len(idx) > size:
            size = len(idx)
            idx0 = idx
    coordi = np.unravel_index(idx0, imgt.shape)
    radius = list(set(coordi[1]))
    lx = len(radius)
    radius_band = radius[int(0.4 * lx):int(0.6 * lx)]

    idx0 = []
    for radius_ele in list(radius_band):
        idx0 = idx0 + list(np.flatnonzero(img_edt == radius_ele))
    idx0 = np.array(idx0)
    refer = img.flat[idx0]

    for radius_beam in list(radius_img[np.max(radius_band):]):
        idx = np.flatnonzero(img_edt == radius_beam)
        current_circle = img.flat[idx]
        current_circle = exposure.match_histograms(current_circle, refer)
        img_initial.flat[idx] = current_circle

    return img_initial


def ring_artifact_correction(img, cx=-1, cy=-1):
    img_initial = img.copy()
    rows, col = img.shape
    if cx < 0:
        cx = int(np.round(rows / 2))
    if cy < 0:
        cy = int(np.round(col / 2))

    imgt = np.ones((rows, col), dtype='u1')
    imgt[cx][cy] = 0
    img_edt = ndimage.distance_transform_edt(imgt)
    img_edt = np.round(img_edt).astype('u4')

    dist = list(np.unique(img_edt))
    num_b = np.min(np.array([cx, cy, rows - cx, col - cy]))
    idx_inner = np.flatnonzero(img_edt <= num_b)

    refer_radius = int(num_b / 2)
    mask = (img_edt <= refer_radius + 2) & (img_edt >= refer_radius - 2)
    refer_ring_idx = np.flatnonzero(mask)
    refer = img.flat[refer_ring_idx]

    for radius_beam in dist:
        idx = np.flatnonzero(img_edt == radius_beam)
        idx = np.intersect1d(idx, idx_inner)
        current_circle = img.flat[idx]
        current_circle = exposure.match_histograms(current_circle, refer)
        img_initial.flat[idx] = current_circle

    return img_initial


def intensity_calibration(corrected, reference):
    mask = reference > 0
    if mask.sum() == 0:
        return corrected
    ref_float = reference.astype(np.float64)
    cor_float = corrected.astype(np.float64)
    ref_mean = ref_float[mask].mean()
    ref_std = ref_float[mask].std()
    cor_mean = cor_float[mask].mean()
    cor_std = cor_float[mask].std()
    if cor_std < 1e-8:
        return corrected
    calibrated = cor_float.copy()
    calibrated[mask] = (cor_float[mask] - cor_mean) / cor_std * ref_std + ref_mean
    return np.clip(calibrated, 0, 255).astype(np.uint8)


# ── SAP filters ──

def snn_filter(img: np.ndarray, radius=2) -> np.ndarray:
    img_f = img.astype(np.float64)
    pad = np.pad(img_f, radius, mode='reflect')
    h, w = img_f.shape
    result = np.zeros_like(img_f)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = pad[radius + dy: radius + dy + h, radius + dx: radius + dx + w]
            sym_neighbor = pad[radius - dy: radius - dy + h, radius - dx: radius - dx + w]
            closer = np.where(
                np.abs(neighbor - img_f) <= np.abs(sym_neighbor - img_f),
                neighbor, sym_neighbor
            )
            result += closer
            count += 1
    result = (result + img_f) / (count + 1)
    return np.clip(result, 0, 255).astype(np.uint8)


def denoise_gaussian(img: np.ndarray, ksize=5, sigma=1.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def denoise_median(img: np.ndarray, ksize=5) -> np.ndarray:
    return cv2.medianBlur(img, ksize)


def denoise_mean(img: np.ndarray, ksize=5) -> np.ndarray:
    return cv2.blur(img, (ksize, ksize))


def denoise_nlm(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    sigma_est = float(np.mean(estimate_sigma(img_f, channel_axis=None)))
    denoised = denoise_nl_means(
        img_f, h=0.8 * sigma_est, sigma=sigma_est,
        patch_size=5, patch_distance=7,
        channel_axis=None, fast_mode=True
    )
    return (np.clip(denoised, 0, 1) * 255).astype(np.uint8)


# All SAP filter methods
SAP_FILTERS = {
    "snn": snn_filter,
    "gaussian": denoise_gaussian,
    "median": denoise_median,
    "mean": denoise_mean,
    "nlm": denoise_nlm,
}


# ══════════════════════════════════════════════════════════════════════
# Patch-wise Processing
# ══════════════════════════════════════════════════════════════════════

def split_into_patches(img: Image.Image, patch_size=256, stride=128):
    w, h = img.size
    patches, positions = [], []
    y_positions = list(range(0, h - patch_size + 1, stride))
    x_positions = list(range(0, w - patch_size + 1, stride))
    if len(y_positions) == 0 or y_positions[-1] + patch_size < h:
        y_positions.append(max(0, h - patch_size))
    if len(x_positions) == 0 or x_positions[-1] + patch_size < w:
        x_positions.append(max(0, w - patch_size))
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))
    for top in y_positions:
        for left in x_positions:
            right = min(left + patch_size, w)
            bottom = min(top + patch_size, h)
            actual_w = right - left
            actual_h = bottom - top
            patch = img.crop((left, top, right, bottom))
            padded = Image.new("L", (patch_size, patch_size), color=0)
            padded.paste(patch, (0, 0))
            patches.append(padded)
            positions.append((left, top, actual_w, actual_h))
    return patches, positions, (w, h)


def create_blend_weights(h, w, margin_ratio=0.2):
    margin_h = max(int(h * margin_ratio), 10)
    margin_w = max(int(w * margin_ratio), 10)
    margin_h = min(margin_h, h // 2)
    margin_w = min(margin_w, w // 2)
    weight = np.ones((h, w), dtype=np.float32)
    for i in range(margin_h):
        alpha = (i + 1) / (margin_h + 1)
        weight[i, :] *= alpha
        weight[-(i + 1), :] *= alpha
    for j in range(margin_w):
        alpha = (j + 1) / (margin_w + 1)
        weight[:, j] *= alpha
        weight[:, -(j + 1)] *= alpha
    return weight


def merge_patches(patches, positions, full_size,
                  patch_size=256, stride=128, margin_ratio=0.2):
    full_w, full_h = full_size
    canvas = np.zeros((full_h, full_w), dtype=np.float32)
    weight_sum = np.zeros((full_h, full_w), dtype=np.float32)
    for patch, pos in zip(patches, positions):
        x, y, actual_w, actual_h = pos
        patch_np = np.array(patch).squeeze().astype(np.float32)
        if patch_np.max() > 1.5:
            patch_np = patch_np / 255.0
        patch_valid = patch_np[:actual_h, :actual_w]
        patch_weight = create_blend_weights(actual_h, actual_w, margin_ratio)
        y_end = min(y + actual_h, full_h)
        x_end = min(x + actual_w, full_w)
        valid_h = y_end - y
        valid_w = x_end - x
        patch_valid = patch_valid[:valid_h, :valid_w]
        patch_weight = patch_weight[:valid_h, :valid_w]
        canvas[y:y_end, x:x_end] += patch_valid * patch_weight
        weight_sum[y:y_end, x:x_end] += patch_weight
    weight_sum = np.maximum(weight_sum, 1e-8)
    return np.clip(canvas / weight_sum, 0, 1)


def denoise_image_patchwise(noisy_pil, denoise_fn, stride=128):
    patches, positions, full_size = split_into_patches(
        noisy_pil, patch_size=PATCH_SIZE, stride=stride
    )
    denoised_patches = []
    for patch in patches:
        patch_np = np.array(patch).astype(np.uint8)
        denoised_np = denoise_fn(patch_np)
        denoised_patches.append(denoised_np)
    result = merge_patches(
        denoised_patches, positions, full_size,
        patch_size=PATCH_SIZE, stride=stride, margin_ratio=MARGIN_RATIO
    )
    return result


# ══════════════════════════════════════════════════════════════════════
# Core: denoise one folder with one SAP filter
# ══════════════════════════════════════════════════════════════════════

def detect_noise_types(folder_name):
    name = folder_name.lower()
    noise_types = set()
    if 'sap' in name:
        noise_types.add('sap')
    if 'ring' in name:
        noise_types.add('ring')
    if 'bh' in name:
        noise_types.add('bh')
    return noise_types


def denoise_one_image(img_uint8, noise_types, sap_fn):
    """
    Apply traditional pipeline to one image.
    BH → Ring (full image + calibration) → SAP (patch-wise)
    sap_fn: which SAP filter to use
    """
    img = img_uint8.copy()
    img_before = img.copy()

    if 'bh' in noise_types:
        img = beam_hardening_correction(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = intensity_calibration(img, img_before)

    if 'ring' in noise_types:
        img_before_ring = img.copy()
        img = ring_artifact_correction(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = intensity_calibration(img, img_before_ring)

    if 'sap' in noise_types:
        img_pil = Image.fromarray(img)
        result = denoise_image_patchwise(img_pil, sap_fn, stride=STRIDE)
        return result

    return img.astype(np.float32) / 255.0


def process_folder(sample_root, folder_name, clean_path, records):
    """
    Process one noise folder.
    - If folder contains SAP noise: run all 5 SAP filters
    - If no SAP: run BH/Ring pipeline once (method_name = 'traditional')
    """
    folder_path = os.path.join(sample_root, folder_name)
    if not os.path.isdir(folder_path):
        print(f"  [SKIP] Not found: {folder_path}")
        return

    noise_types = detect_noise_types(folder_name)
    has_sap = 'sap' in noise_types
    sample_name = os.path.basename(sample_root)  # 'sandstone' or 'carbonate'

    # Load clean image
    clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
    if clean_img is None:
        print(f"  [ERROR] Cannot read clean image: {clean_path}")
        return
    clean_np = clean_img.astype(np.float32) / 255.0

    # Get noisy image files
    img_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))
        and not f.startswith('.')
        and '_denoised' not in f.lower()
    ])
    # Exclude any existing denoised subfolders' files
    img_files = [f for f in img_files if os.path.isfile(os.path.join(folder_path, f))]

    if not img_files:
        print(f"  [SKIP] No images in {folder_path}")
        return

    # Determine which SAP filters to run
    if folder_name == "sap_only":
        sap_methods = SAP_FILTERS  # all 5 filters for comparison
    elif has_sap:
        sap_methods = {"snn": snn_filter}  # only SNN for combined noise
    else:
        sap_methods = {"traditional": None}  # no SAP, just BH/Ring

    for method_name, sap_fn in sap_methods.items():
        # Output directory
        out_dir = os.path.join(folder_path, f"{method_name}_denoised")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  --- {folder_name} / {method_name} ({len(img_files)} images) ---")

        for img_file in tqdm(img_files, desc=f"    {method_name}", leave=False):
            img_path = os.path.join(folder_path, img_file)
            img_uint8 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_uint8 is None:
                continue

            # Noisy metrics
            noisy_np = img_uint8.astype(np.float32) / 255.0
            if noisy_np.shape != clean_np.shape:
                print(f"    [WARN] Shape mismatch: {img_file}")
                continue

            try:
                if has_sap:
                    denoised_np = denoise_one_image(img_uint8, noise_types, sap_fn)
                else:
                    denoised_np = denoise_one_image(img_uint8, noise_types, None)

                # Metrics
                psnr_val = calc_psnr(clean_np, denoised_np, data_range=1.0)
                ssim_val = calc_ssim(clean_np, denoised_np, data_range=1.0)

                # Save
                out_img = Image.fromarray((denoised_np * 255).astype(np.uint8))
                out_img.save(os.path.join(out_dir, img_file))

                records.append({
                    "sample": sample_name,
                    "noise_type": folder_name,
                    "method": method_name,
                    "image": img_file,
                    "PSNR": round(psnr_val, 4),
                    "SSIM": round(ssim_val, 4),
                })

            except Exception as e:
                print(f"    [ERROR] {method_name} on {img_file}: {e}")
                records.append({
                    "sample": sample_name,
                    "noise_type": folder_name,
                    "method": method_name,
                    "image": img_file,
                    "PSNR": None,
                    "SSIM": None,
                })


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    records = []

    # ── Sandstone ──
    print("=" * 70)
    print("SANDSTONE")
    print("=" * 70)
    for folder in SANDSTONE_FOLDERS:
        print(f"\n{'=' * 60}")
        print(f"Processing: sandstone / {folder}")
        process_folder(SANDSTONE_ROOT, folder, SANDSTONE_CLEAN, records)

    # ── Carbonate ──
    print("\n" + "=" * 70)
    print("CARBONATE")
    print("=" * 70)
    for folder in CARBONATE_FOLDERS:
        print(f"\n{'=' * 60}")
        print(f"Processing: carbonate / {folder}")
        process_folder(CARBONATE_ROOT, folder, CARBONATE_CLEAN, records)

    # ── Save CSV ──
    df = pd.DataFrame(records)
    csv_path = os.path.join(SANDSTONE_ROOT, "traditional_all_results.csv")
    df.to_csv(csv_path, index=False)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    summary_rows = []
    for (sample, noise, method), grp in df.groupby(["sample", "noise_type", "method"]):
        psnr_vals = grp["PSNR"].dropna()
        ssim_vals = grp["SSIM"].dropna()
        if len(psnr_vals) > 0:
            summary_rows.append({
                "sample": sample,
                "noise_type": noise,
                "method": method,
                "PSNR": f"{psnr_vals.mean():.2f} ± {psnr_vals.std():.2f}",
                "SSIM": f"{ssim_vals.mean():.4f} ± {ssim_vals.std():.4f}",
                "n": len(psnr_vals),
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(SANDSTONE_ROOT, "traditional_summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    print(df_summary.to_string(index=False))
    print(f"\nDetailed CSV: {csv_path}")
    print(f"Summary CSV:  {summary_csv}")
    print("Done!")


if __name__ == "__main__":
    main()