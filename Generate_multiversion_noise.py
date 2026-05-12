"""
Generate N versions of each noise type/combination on a fixed clean image.
Each version uses different random noise parameters.
Within each version, combined noise reuses the SAME params as single noise.
All parameters are recorded in a JSON log for reproducibility.

Output structure:
  output_dir/
    sap_only/          -> v00.png, v01.png, ..., v29.png
    ring_only/         -> v00.png, v01.png, ..., v29.png
    bh_only/           -> v00.png, v01.png, ..., v29.png
    sap_ring/          -> v00.png, v01.png, ..., v29.png
    sap_bh/            -> v00.png, v01.png, ..., v29.png
    ring_bh/           -> v00.png, v01.png, ..., v29.png
    sap_ring_bh/       -> v00.png, v01.png, ..., v29.png
    clean/             -> clean.png (single copy)
    noise_params.json  -> all parameters for all versions
"""

import os
import cv2
import numpy as np
from scipy.special import sph_harm_y
import json
from tqdm import tqdm

# ========================================================================
# Noise Functions (original from dataset_pre.py, NO modifications)
# ========================================================================

def add_salt_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5):
    noisy = np.copy(image)
    num_salt = int(np.ceil(amount * image.size * salt_vs_pepper))
    num_pepper = int(np.ceil(amount * image.size * (1 - salt_vs_pepper)))
    coords_salt = [np.random.randint(0, i, num_salt) for i in image.shape]
    coords_pepper = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[tuple(coords_salt)] = 255
    noisy[tuple(coords_pepper)] = 0
    return noisy


def add_ring_artifact(image, num_rings=10, intensity=30):
    rows, cols = image.shape
    center = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    ring_img = image.astype(np.float32)
    for i in range(num_rings):
        ring_radius = (i + 1) * (min(rows, cols) / (2 * num_rings))
        ring_width = np.random.randint(10, 30)
        inner = ring_radius - ring_width / 2
        outer = ring_radius + ring_width / 2
        ring_mask = np.logical_and(dist >= inner, dist <= outer)
        ring_img[ring_mask] += intensity * (np.random.rand() - 0.5)
    return np.clip(ring_img, 0, 255).astype(np.uint8)


def sph_harmonic_shape(image, lm_list, coeffs, base_radius=1, scale_factor=3, n_points=500):
    img_height, img_width = image.shape
    image_center_x = img_width // 2
    image_center_y = img_height // 2

    x_std_dev = img_width * 0.1
    y_std_dev = img_height * 0.1
    x_offset = np.random.normal(loc=0, scale=x_std_dev)
    y_offset = np.random.normal(loc=0, scale=y_std_dev)

    center_x = int(np.clip(image_center_x + x_offset, 0, img_width - 1))
    center_y = int(np.clip(image_center_y + y_offset, 0, img_height - 1))
    phi = np.linspace(0, 2 * np.pi, n_points)
    theta = np.pi / 2
    radii = np.zeros(n_points)

    for i, phi_value in enumerate(phi):
        total_Y_lm_sum = 0.0
        for (l, m), c in zip(lm_list, coeffs):
            Y_lm = sph_harm_y(l, m, theta, phi_value)
            total_Y_lm_sum += c * Y_lm
        radii[i] = base_radius + scale_factor * np.abs(np.real(total_Y_lm_sum))

    vertices = []
    for i in range(n_points):
        x = int(center_x + radii[i] * np.cos(phi[i]))
        y = int(center_y + radii[i] * np.sin(phi[i]))
        vertices.append((x, y))

    pts = np.array(vertices, np.int32)
    pts = pts.reshape((-1, 1, 2))

    image_with_shape = np.zeros(image.shape, dtype='f4')
    cv2.fillPoly(image_with_shape, [pts], 128)
    index = np.flatnonzero(image_with_shape == 128)

    imgt = np.zeros(image.shape, dtype='f4')
    mean = 30
    variance = 10
    mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
    sigma = np.sqrt(np.log(1 + variance / mean ** 2))
    random_value = np.random.lognormal(mean=mu, sigma=sigma)
    random_value *= np.random.choice([-1, 1])
    imgt.flat[index] += random_value
    kernel_height = np.random.randint(img_height / 30, img_height / 10)
    kernel_width = np.random.randint(img_width / 30, img_width / 10)
    imgt = cv2.blur(imgt, (int(kernel_height / 2) * 2 + 1, int(kernel_width / 2) * 2 + 1))
    image_with_shape = image + imgt
    idx = np.flatnonzero(image_with_shape >= 255)
    image_with_shape.flat[idx] = 255

    return image_with_shape, vertices, (center_x, center_y), index


def draw_limited_by_mask(image, center=None, radius=None):
    img_height, img_width = image.shape
    if center is None:
        center = (img_width // 2, img_height // 2)
    if radius is None:
        radius = min(img_height, img_width) // 2
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    result_image = cv2.bitwise_and(image, image, mask=mask)
    return result_image


# ========================================================================
# Config
# ========================================================================

INPUT_PATH = r"E:\243Summer session\image denoising_Jiang\new_coding\output_test\random_noise_tensity_on_fixed_slice\00062_s0020_v02.png"
OUTPUT_DIR = r"E:\243Summer session\image denoising_Jiang\new_coding\output_test\random_noise_tensity_on_fixed_slice\carbonate"
SANDSTONE_PARAMS_PATH = r"E:\243Summer session\image denoising_Jiang\new_coding\output_test\random_noise_tensity_on_fixed_slice\sandstone\all_noise_params.json"
NUM_VERSIONS = 30
USE_MASK = False  # True for sandstone, False for carbonate
SANDSTONE_WIDTH = 2007  # sandstone image width, for BH scaling
# Noise parameter ranges (same as your dataset_pre.py)
SAP_AMOUNT_RANGE = (0.01, 0.05)
SAP_RATIO_RANGE = (0.3, 0.7)
RING_NUM_RANGE = (10, 20)
RING_INTENSITY_RANGE = (30, 60)
BH_LM_LIST = [(0, 0), (1, 0), (1, 1), (2, -2), (3, 1), (4, 2), (5, 0), (6, -3), (7, 4)]

# All 7 noise combinations
NOISE_COMBOS = [
    "sap_only",
    "ring_only",
    "bh_only",
    "sap_ring",
    "sap_bh",
    "ring_bh",
    "sap_ring_bh",
]


# ========================================================================
# Parameter generation (one set per version, reused across combos)
# ========================================================================

def generate_version_params(img_w, img_h, version_seed):
    """Generate one set of noise parameters for a version."""
    np.random.seed(version_seed)

    # SAP params
    sap_amount = round(np.random.uniform(*SAP_AMOUNT_RANGE), 3)
    sap_ratio = round(np.random.uniform(*SAP_RATIO_RANGE), 2)
    sap_seed = np.random.randint(0, 100000)

    # Ring params
    ring_num = int(np.random.randint(*RING_NUM_RANGE))
    ring_intensity = int(np.random.randint(*RING_INTENSITY_RANGE))
    ring_seed = np.random.randint(0, 100000)

    # BH params
    bh_seed = np.random.randint(0, 100000)
    np.random.seed(bh_seed)
    tail_coeffs = np.random.uniform(0, 2, size=7)
    tail_max = np.max(tail_coeffs)
    head_coeffs = np.random.uniform(tail_max + 1, tail_max + 3, size=2)
    bh_coeffs = list(head_coeffs) + list(tail_coeffs)
    bh_base_radius = int(np.random.randint(img_w / 6, img_w / 3))
    bh_scale_factor = int(np.random.randint(bh_base_radius / 3, bh_base_radius))

    return {
        "version_seed": version_seed,
        "salt_pepper": {
            "amount": sap_amount,
            "salt_vs_pepper": sap_ratio,
            "seed": int(sap_seed),
        },
        "ring_artifact": {
            "num_rings": ring_num,
            "intensity": ring_intensity,
            "seed": int(ring_seed),
        },
        "beam_hardening": {
            "coeffs": [float(c) for c in bh_coeffs],
            "base_radius": bh_base_radius,
            "scale_factor": bh_scale_factor,
            "seed": int(bh_seed),
        },
    }


# ========================================================================
# Noise application helpers (reuse params via seed)
# ========================================================================

def apply_sap(image, params):
    p = params["salt_pepper"]
    np.random.seed(p["seed"])
    return add_salt_pepper_noise(image, amount=p["amount"], salt_vs_pepper=p["salt_vs_pepper"])


def apply_ring(image, params):
    p = params["ring_artifact"]
    np.random.seed(p["seed"])
    return add_ring_artifact(image, num_rings=p["num_rings"], intensity=p["intensity"])

def apply_bh(image, params):
    p = params["beam_hardening"]
    np.random.seed(p["seed"])
    tc = np.random.uniform(0, 2, size=7)
    tm = np.max(tc)
    hc = np.random.uniform(tm + 1, tm + 3, size=2)
    coeffs = list(hc) + list(tc)
    # Use carbonate's own image size for radius range
    img_w = image.shape[1]
    br = int(np.random.randint(img_w / 6, img_w / 3))
    sf = int(np.random.randint(br / 3, br))
    result, _, _, _ = sph_harmonic_shape(image, BH_LM_LIST, coeffs, br, sf, 500)
    return result

def finalize(image, use_mask=True):
    """Clip to uint8, optionally apply circular mask."""
    image = np.clip(image, 0, 255).astype(np.uint8)
    if use_mask:
        image = draw_limited_by_mask(image)
    return image


def generate_noisy_image(clean, combo, params):
    """Generate a noisy image for a given combination using given params."""
    img = clean.copy()

    # Pipeline: always apply in order BH → Ring → SAP
    if "bh" in combo:
        img = apply_bh(img, params)

    if "ring" in combo:
        img = apply_ring(img, params)

    if "sap" in combo:
        img = apply_sap(img, params)

    return finalize(img, use_mask=USE_MASK)


# Mapping from combo name to noise types
COMBO_TO_TYPES = {
    "sap_only":     {"sap"},
    "ring_only":    {"ring"},
    "bh_only":      {"bh"},
    "sap_ring":     {"sap", "ring"},
    "sap_bh":       {"sap", "bh"},
    "ring_bh":      {"ring", "bh"},
    "sap_ring_bh":  {"sap", "ring", "bh"},
}


# ========================================================================
# Main
# ========================================================================

def main():
    # Read clean image
    clean = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)
    if clean is None:
        raise FileNotFoundError(f"Cannot read image: {INPUT_PATH}")
    img_h, img_w = clean.shape
    print(f"Clean image loaded: {clean.shape}")
    print(f"Generating {NUM_VERSIONS} versions x {len(NOISE_COMBOS)} combinations")
    print(f"Use mask: {USE_MASK}\n")

    # Create output directories
    clean_dir = os.path.join(OUTPUT_DIR, "clean")
    os.makedirs(clean_dir, exist_ok=True)
    cv2.imwrite(os.path.join(clean_dir, "clean.png"), clean)

    for combo in NOISE_COMBOS:
        os.makedirs(os.path.join(OUTPUT_DIR, combo), exist_ok=True)

    # Generate all versions
    all_params = {}

    for v in tqdm(range(NUM_VERSIONS), desc="Versions"):
        version_seed = 1000 + v  # Deterministic but different per version
        params = generate_version_params(img_w, img_h, version_seed)
        all_params[f"v{v:02d}"] = params

        for combo in NOISE_COMBOS:
            noise_types = COMBO_TO_TYPES[combo]
            noisy_img = generate_noisy_image(clean, noise_types, params)

            filename = f"v{v:02d}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, combo, filename), noisy_img)

    # Save all parameters
    meta = {
        "input_image": os.path.basename(INPUT_PATH),
        "num_versions": NUM_VERSIONS,
        "use_mask": USE_MASK,
        "image_shape": [img_h, img_w],
        "noise_param_ranges": {
            "sap_amount": list(SAP_AMOUNT_RANGE),
            "sap_ratio": list(SAP_RATIO_RANGE),
            "ring_num": list(RING_NUM_RANGE),
            "ring_intensity": list(RING_INTENSITY_RANGE),
            "bh_lm_list": BH_LM_LIST,
        },
        "versions": all_params,
    }

    params_path = os.path.join(OUTPUT_DIR, "all_noise_params.json")
    with open(params_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"{'=' * 60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total images: {NUM_VERSIONS * len(NOISE_COMBOS)}")
    print(f"\nSubdirectories:")
    for combo in NOISE_COMBOS:
        print(f"  {combo}/  -> {NUM_VERSIONS} images")
    print(f"  clean/  -> 1 image")
    print(f"\nParameter log: {params_path}")

    # Print a sample version's params
    sample = all_params["v00"]
    print(f"\nSample (v00) params:")
    print(f"  SAP:  amount={sample['salt_pepper']['amount']}, ratio={sample['salt_pepper']['salt_vs_pepper']}")
    print(f"  Ring: num_rings={sample['ring_artifact']['num_rings']}, intensity={sample['ring_artifact']['intensity']}")
    print(f"  BH:   base_radius={sample['beam_hardening']['base_radius']}, scale_factor={sample['beam_hardening']['scale_factor']}")


# ========================================================================
# Cross-rock variant: apply sandstone noise params to a carbonate image.
# Purpose: generate carbonate noisy images with the same noise intensity
# as sandstone v00..v29, enabling fair cross-rock qualitative comparison.
# Only sap_ring_bh is activated in NOISE_COMBOS for this purpose.
# Run this variant manually when carbonate noisy images need to be rebuilt.
# ========================================================================

# def main_cross_rock():
#     clean = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)
#     if clean is None:
#         raise FileNotFoundError(f"Cannot read image: {INPUT_PATH}")
#     img_h, img_w = clean.shape
#     print(f"Clean image loaded: {clean.shape}")
#
#     with open(SANDSTONE_PARAMS_PATH, 'r') as f:
#         sandstone_meta = json.load(f)
#     all_params = sandstone_meta["versions"]
#     num_versions = len(all_params)
#     print(f"Loaded {num_versions} versions from sandstone params")
#
#     clean_dir = os.path.join(OUTPUT_DIR, "clean")
#     os.makedirs(clean_dir, exist_ok=True)
#     cv2.imwrite(os.path.join(clean_dir, "clean.png"), clean)
#
#     for combo in NOISE_COMBOS:
#         os.makedirs(os.path.join(OUTPUT_DIR, combo), exist_ok=True)
#
#     for v_key, params in tqdm(all_params.items(), desc="Versions"):
#         for combo in NOISE_COMBOS:
#             noise_types = COMBO_TO_TYPES[combo]
#             noisy_img = generate_noisy_image(clean, noise_types, params)
#             cv2.imwrite(os.path.join(OUTPUT_DIR, combo, f"{v_key}.png"), noisy_img)
#
#     meta = {
#         "input_image": os.path.basename(INPUT_PATH),
#         "source_params": SANDSTONE_PARAMS_PATH,
#         "use_mask": USE_MASK,
#         "image_shape": [img_h, img_w],
#         "bh_scale_ratio": img_w / SANDSTONE_WIDTH,
#     }
#     with open(os.path.join(OUTPUT_DIR, "carbonate_meta.json"), 'w') as f:
#         json.dump(meta, f, indent=2)
#
#     print(f"\nDone! {num_versions} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()