import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import euler_number, perimeter_crofton
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tqdm import tqdm

try:
    font_manager.findfont('Arial', fallback_to_default=False)
    matplotlib.rcParams['font.family'] = 'Arial'
except Exception:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')

# Current paper experiment layout.
MINKOWSKI_ROOT = Path("output_test/minkowski_functionals")
RESULTS_ROOT = MINKOWSKI_ROOT / "results"
SANDSTONE_ROI = (8, 32, 1994, 1984)  # ImageJ: makeOval(x, y, width, height)


# ========================================================================
# 1. Binarization (segmentation -> pore/solid mask)
# ========================================================================

def binarize_image(img_array, dark_is_pore=True, fixed_threshold=None):
    """
    Convert a grayscale image into a boolean pore mask (True = pore).

    - If the image already has <=2 unique gray levels, it is treated as an
      already-segmented mask (no thresholding applied).
    - Otherwise it is binarized with a fixed threshold or Otsu's method.
    """
    unique_vals = np.unique(img_array)

    if len(unique_vals) <= 2:
        low = unique_vals.min()
        high = unique_vals.max() if len(unique_vals) == 2 else low
        pore_val = low if dark_is_pore else high
        return img_array == pore_val

    threshold = fixed_threshold if fixed_threshold is not None else threshold_otsu(img_array)
    return img_array < threshold if dark_is_pore else img_array > threshold


def build_masks(img_array, dark_is_pore=True, fixed_threshold=None,
                pore_label=None, background_label=None):
    """
    Build (pore_mask, valid_mask) from a segmented image.

    Two modes:
    - Label mode (pore_label is not None): img_array holds integer class
      labels (e.g. an Ilastik "Simple Segmentation" export with values
      1/2/3). pore_mask = (img_array == pore_label); pixels equal to
      background_label are excluded from valid_mask so they never enter
      the area/perimeter/Euler-number calculation.
    - Grayscale mode (pore_label is None): falls back to binarize_image()
      (Otsu or fixed threshold); the whole image is valid.
    """
    if pore_label is not None:
        pore_mask = img_array == pore_label
        if background_label is not None:
            valid_mask = img_array != background_label
        else:
            valid_mask = np.ones_like(img_array, dtype=bool)
        return pore_mask, valid_mask

    pore_mask = binarize_image(img_array, dark_is_pore=dark_is_pore, fixed_threshold=fixed_threshold)
    return pore_mask, np.ones_like(pore_mask, dtype=bool)


def oval_mask(shape, roi_oval):
    """Return an ImageJ-compatible oval ROI mask using pixel centres.

    ``roi_oval`` follows ImageJ's ``makeOval(x, y, width, height)`` convention.
    """
    height, width = shape
    x, y, roi_width, roi_height = roi_oval
    if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0:
        raise ValueError(f"Invalid oval ROI: {roi_oval}")
    if x + roi_width > width or y + roi_height > height:
        raise ValueError(
            f"Oval ROI {roi_oval} is outside image bounds {width}x{height}"
        )

    yy, xx = np.ogrid[:height, :width]
    center_x = x + roi_width / 2.0
    center_y = y + roi_height / 2.0
    return (
        ((xx + 0.5 - center_x) / (roi_width / 2.0)) ** 2
        + ((yy + 0.5 - center_y) / (roi_height / 2.0)) ** 2
        <= 1.0
    )


def _masked_perimeter(pore_mask, valid_mask):
    """
    Digital pore-matrix boundary length (4-connectivity edge count),
    counting only transitions where both neighboring pixels are inside
    the valid (non-background) domain.
    """
    pore = pore_mask & valid_mask
    total = 0
    h_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    total += np.count_nonzero((pore[:, :-1] != pore[:, 1:]) & h_valid)
    v_valid = valid_mask[:-1, :] & valid_mask[1:, :]
    total += np.count_nonzero((pore[:-1, :] != pore[1:, :]) & v_valid)
    return float(total)


# ========================================================================
# 2. 2D Minkowski functionals
# ========================================================================

def compute_minkowski_functionals_2d(pore_mask, valid_mask=None):
    """
    Compute the three 2D Minkowski functionals of a binary pore/solid image.

    valid_mask restricts the domain (e.g. excludes a "background" label);
    pixels outside it don't count toward area, perimeter, or connectivity.
    When valid_mask covers the whole image, perimeter uses the Crofton
    estimator (less digitization bias); otherwise it falls back to a
    masked digital edge count that correctly ignores background pixels.

    porosity          : pore area fraction within the valid domain (dimensionless)
    perimeter_total    : pore-matrix boundary length (px)
    perimeter_density : perimeter_total per unit valid area         (1/px)
    euler_number       : Euler characteristic of the pore phase      (#components - #holes)
    euler_density      : euler_number per unit valid area            (1/px^2)
    """
    if valid_mask is None:
        valid_mask = np.ones_like(pore_mask, dtype=bool)

    area = valid_mask.sum()
    pore_in_domain = pore_mask & valid_mask
    porosity = pore_in_domain.sum() / area

    if valid_mask.all():
        perimeter_total = perimeter_crofton(pore_mask, directions=4)
    else:
        perimeter_total = _masked_perimeter(pore_mask, valid_mask)

    euler_total = euler_number(pore_in_domain, connectivity=2)

    return {
        "porosity": porosity,
        "perimeter_total": perimeter_total,
        "perimeter_density": perimeter_total / area,
        "euler_number": euler_total,
        "euler_density": euler_total / area,
    }


# ========================================================================
# 3. Per-folder / cross-method analysis
# ========================================================================

def image_pair_sort_key(filename):
    """Natural pairing order: unnumbered image first, then numeric suffixes.

    ImageJ exports use inconsistent prefixes (``Classified image`` versus
    ``Classification result``), so alphabetical order is not reliable. The
    suffix is nevertheless stable enough to put base, _01/_011/... and _0786
    into the intended positional order without requiring identical filenames.
    """
    stem = Path(filename).stem
    match = re.search(r"_(\d+)$", stem)
    if match is None:
        return (0, 0, stem.casefold())
    return (1, int(match.group(1)), stem.casefold())


def analyze_folder(folder_path, dark_is_pore=True, fixed_threshold=None,
                   pore_label=None, background_label=None, roi_oval=None,
                   show_progress=True):
    """
    Compute Minkowski functionals for every image in a folder.

    Pass pore_label (and optionally background_label) to treat images as
    integer label maps (e.g. Ilastik Simple Segmentation exports) instead
    of grayscale images to be thresholded.
    """
    files = sorted(
        (f for f in os.listdir(folder_path) if f.lower().endswith(IMG_EXTENSIONS)),
        key=image_pair_sort_key,
    )
    iterator = tqdm(files, desc=os.path.basename(folder_path)) if show_progress else files

    records = []
    for sample_index, fname in enumerate(iterator):
        pil_img = Image.open(os.path.join(folder_path, fname))
        img = np.array(pil_img) if pore_label is not None else np.array(pil_img.convert('L'))
        pore_mask, valid_mask = build_masks(
            img, dark_is_pore=dark_is_pore, fixed_threshold=fixed_threshold,
            pore_label=pore_label, background_label=background_label
        )
        if roi_oval is not None:
            valid_mask &= oval_mask(img.shape[:2], roi_oval)
        metrics = compute_minkowski_functionals_2d(pore_mask, valid_mask)
        metrics["image"] = fname
        # Pair corresponding images by their sorted position, not filename.
        metrics["sample_index"] = sample_index
        records.append(metrics)

    return pd.DataFrame(records)


def compare_methods(method_folders, output_dir, dark_is_pore=True, fixed_threshold=None,
                    pore_label=None, background_label=None,
                    baseline=None, roi_oval=None, show_progress=True):
    """
    Compute and compare Minkowski functionals across several segmentation
    results (e.g. one folder per denoising model).

    method_folders : dict {method_name: folder_path}
    pore_label / background_label : see analyze_folder(); set pore_label to
                                     switch every folder to label-map mode.
    baseline        : method_name to compute relative differences against
                       (e.g. the un-denoised / original segmentation). Pass
                       None to skip relative-difference comparison.

    Saves under output_dir:
      minkowski_per_image.csv          per-image, per-method functionals
      minkowski_summary.csv            mean +/- std per method
      minkowski_relative_to_<name>.csv relative diff (%) vs baseline (if set)
      minkowski_comparison.png         bar chart comparing methods
    """
    os.makedirs(output_dir, exist_ok=True)

    if pore_label is not None:
        print(f"Label-map mode: pore_label={pore_label}, background_label={background_label} "
              f"(everything else is treated as solid matrix)")

    per_method = {}
    for name, path in method_folders.items():
        if not os.path.isdir(path):
            print(f"[SKIP] Folder not found: {path}")
            continue
        df = analyze_folder(path, dark_is_pore=dark_is_pore, fixed_threshold=fixed_threshold,
                            pore_label=pore_label, background_label=background_label,
                            roi_oval=roi_oval,
                            show_progress=show_progress)
        if df.empty:
            print(f"[SKIP] No segmentation images found directly in: {path}")
            continue
        df["method"] = name
        per_method[name] = df

    if not per_method:
        raise ValueError("No segmentation images were found in any configured folder")

    combined = pd.concat(per_method.values(), ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "minkowski_per_image.csv"), index=False)

    summary = combined.groupby("method")[["porosity", "perimeter_density", "euler_density"]] \
                       .agg(["mean", "std"])
    summary.to_csv(os.path.join(output_dir, "minkowski_summary.csv"))
    print("\n=== Summary (mean ± std per method) ===")
    print(summary)

    if baseline is not None and baseline in per_method:
        baseline_df = per_method[baseline].set_index("sample_index")
        diff_records = []
        for name, df in per_method.items():
            if name == baseline:
                continue
            if len(df) != len(baseline_df):
                raise ValueError(
                    f"Cannot pair '{name}' with baseline '{baseline}': "
                    f"found {len(df)} versus {len(baseline_df)} images. "
                    "Each folder must contain the same number of corresponding "
                    "images in the same sorted order."
                )
            merged = df.set_index("sample_index").join(
                baseline_df, rsuffix="_baseline", validate="one_to_one"
            )
            for metric in ["porosity", "perimeter_density", "euler_density"]:
                merged[f"{metric}_abs_diff"] = (
                    merged[metric] - merged[f"{metric}_baseline"]
                ).abs()
                merged[f"{metric}_rel_diff_%"] = (
                    (merged[metric] - merged[f"{metric}_baseline"])
                    / (np.abs(merged[f"{metric}_baseline"]) + 1e-12) * 100
                )
                merged[f"{metric}_abs_rel_error_%"] = merged[
                    f"{metric}_rel_diff_%"
                ].abs()
            merged["method"] = name
            diff_records.append(merged.reset_index())

        if diff_records:
            diff_df = pd.concat(diff_records, ignore_index=True)
            diff_path = os.path.join(output_dir, f"minkowski_relative_to_{baseline}.csv")
            diff_df.to_csv(diff_path, index=False)
            print(f"\nPosition-paired relative-difference table saved: {diff_path}")

    plot_comparison_bars(combined, output_dir)
    return combined, summary


def plot_comparison_bars(combined_df, output_dir):
    metrics = ["porosity", "perimeter_density", "euler_density"]
    titles = ["Porosity", "Perimeter density (1/px)", "Euler density (1/px$^2$)"]
    methods = list(combined_df["method"].unique())
    colors = plt.get_cmap("Set2").colors

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, metrics, titles):
        means = [combined_df.loc[combined_df.method == m, metric].mean() for m in methods]
        stds = [combined_df.loc[combined_df.method == m, metric].std() for m in methods]
        ax.bar(methods, means, yerr=stds, capsize=4, color=colors[:len(methods)])
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "minkowski_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    plt.close(fig)


def canonical_method_name(folder_name):
    """Normalize spelling differences such as traditional method/_method."""
    normalized = folder_name.strip().lower().replace(" ", "_")
    return "traditional" if normalized == "traditional_method" else normalized


def method_folders_for_case(clean_dir, noisy_dir):
    """Build clean/noisy/denoising method mapping for one noise case."""
    folders = {
        "clean": str(clean_dir),
        "noisy": str(noisy_dir),
    }
    for child in sorted(noisy_dir.iterdir(), key=lambda p: p.name.casefold()):
        if child.is_dir() and child.resolve() != clean_dir.resolve():
            name = canonical_method_name(child.name)
            if name in folders:
                raise ValueError(f"Duplicate method name '{name}' under {noisy_dir}")
            folders[name] = str(child)
    return folders


def run_current_experiment(root=MINKOWSKI_ROOT, results_root=RESULTS_ROOT):
    """Run all sandstone noise cases plus the current carbonate case."""
    root = Path(root).resolve()
    results_root = Path(results_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Minkowski input root not found: {root}")

    print("Binary segmentation mode: 0=pores, 255=matrix")
    print(f"Sandstone ROI: makeOval{SANDSTONE_ROI}")
    print("Carbonate ROI: full image")

    sandstone_root = root / "sandstone"
    sandstone_clean = sandstone_root / "clean image"
    if not sandstone_clean.is_dir():
        raise FileNotFoundError(f"Sandstone clean folder not found: {sandstone_clean}")

    excluded = {"clean image", "results"}
    noise_dirs = sorted(
        (p for p in sandstone_root.iterdir()
         if p.is_dir() and p.name.casefold() not in excluded),
        key=lambda p: p.name.casefold(),
    )
    for noise_dir in noise_dirs:
        print(f"\n{'=' * 72}\nSANDSTONE / {noise_dir.name}\n{'=' * 72}")
        compare_methods(
            method_folders=method_folders_for_case(sandstone_clean, noise_dir),
            output_dir=results_root / "sandstone" / noise_dir.name,
            dark_is_pore=True,
            fixed_threshold=None,
            pore_label=None,
            background_label=None,
            baseline="clean",
            roi_oval=SANDSTONE_ROI,
        )

    carbonate_root = root / "carbonate"
    carbonate_clean = carbonate_root / "clean"
    if not carbonate_clean.is_dir():
        raise FileNotFoundError(f"Carbonate clean folder not found: {carbonate_clean}")
    print(f"\n{'=' * 72}\nCARBONATE\n{'=' * 72}")
    compare_methods(
        method_folders=method_folders_for_case(carbonate_clean, carbonate_root),
        output_dir=results_root / "carbonate",
        dark_is_pore=True,
        fixed_threshold=None,
        pore_label=None,
        background_label=None,
        baseline="clean",
        roi_oval=None,
    )


# ========================================================================
# 4. main
# ========================================================================

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_current_experiment()
