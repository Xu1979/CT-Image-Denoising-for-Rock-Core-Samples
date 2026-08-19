"""Prepare one clean image for the current x2 ImageJ segmentation study.

The script creates grid-aligned HR and x2 grayscale inputs and empty output
folders for Otsu, MidGrey, and Trainable Weka segmentation. It also
compares, per method, that method's own HR segmentation against that same
method's x2 segmentation upsampled back to the HR grid:

- pixel-wise agreement (Dice / IoU / precision / recall)
- structural statistics deviation (porosity / perimeter / Euler-number
  density, via minkowski_functionals_2d.py, reported as % relative to HR)

IMPORTANT — what these comparisons can and cannot answer:
HR here is a *method-specific* / resolution-consistency reference, not an
independent ground truth: HR-Otsu is only compared against LR-Otsu-upsampled,
never against HR-Weka. So this pipeline answers "how stable is method X
across resolution" for each X separately, not "which method is most
accurate" (a method could be perfectly self-consistent while being wrong in
the same way at both resolutions). Ranking accuracy across methods requires
an independent manually-annotated ground truth that all three methods are
compared against. Likewise, since this script processes one image, its
outputs are a single-slice structural *deviation*, not a systematic bias
estimate (that needs multiple independent slices, reported as mean +/- std).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import minkowski_functionals_2d as mkf  # noqa: E402 (needs SCRIPT_DIR on sys.path first)

DEFAULT_INPUT = (
    SCRIPT_DIR
    / "output_test"
    / "random_noise_tensity_on_fixed_slice"
    / "00061_s0020_v01.png"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "output_test"
    / "single_image_x2"
    / "00061_s0020_v01"
)
SEGMENTATION_METHODS = ("otsu", "midgrey", "weka")
LABEL_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}
FACTOR = 2
DEFAULT_HR_OVAL = (8, 32, 1994, 1984)
DEFAULT_LR_OVAL = (5, 16, 993, 992)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--upsample-labels", action="store_true",
        help="Upsample completed x2 ImageJ label maps to the aligned HR grid.")
    parser.add_argument(
        "--compare-pixelwise", action="store_true",
        help="Dice/IoU/precision/recall of upsampled x2 labels vs each "
             "method's own HR labels (a resolution-consistency reference, "
             "not an independent ground truth), per method.")
    parser.add_argument(
        "--compare-structural", action="store_true",
        help="Porosity/perimeter/Euler-number relative %% deviation of "
             "upsampled x2 vs HR, per method (reuses minkowski_functionals_2d).")
    parser.add_argument(
        "--background-label", action="append", default=[], metavar="METHOD=VALUE",
        help="Per-method label value to exclude from both comparisons, "
             "e.g. weka=0. Repeatable. Only the HR "
             "label image is used to decide what counts as background; the "
             "x2/upsampled prediction is scored (not excluded) even where "
             "it disagrees with HR about what's background.")
    parser.add_argument(
        "--pore-label", action="append", default=[], metavar="METHOD=VALUE",
        help="Explicit pore label for a method whose label image has more "
             "than 2 classes (e.g. weka=2). Repeatable. Used by both "
             "--compare-pixelwise (to report a pore-specific summary) and "
             "--compare-structural.")
    parser.add_argument(
        "--light-is-pore", action="store_true",
        help="For binary (2-value) label images, treat the higher value as "
             "pore instead of the default (lower value = pore). Used by "
             "both --compare-pixelwise and --compare-structural.")
    parser.add_argument(
        "--rel-diff-zero-tol", type=float, default=1e-9,
        help="If abs(HR value) for a structural metric is below this, "
             "report its relative %% deviation as NaN instead of a blown-up "
             "number (mainly matters for euler_density, which can be "
             "exactly 0). Absolute difference is always reported regardless.")
    parser.add_argument(
        "--roi-oval", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
        default=DEFAULT_HR_OVAL,
        help="HR-grid oval ROI used for metrics; default matches the current "
             "experiment: makeOval(8,32,1994,1984).")
    parser.add_argument(
        "--lr-roi-oval", type=int, nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        default=DEFAULT_LR_OVAL,
        help="x2-grid oval ROI; it is mapped to HR and intersected with the "
             "HR oval. Default: makeOval(5,16,993,992).")
    parser.add_argument(
        "--full-image", action="store_true",
        help="Disable the default oval ROI and calculate full-image metrics.")
    return parser.parse_args()


def _oval_mask(shape: tuple[int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    """Rasterize an ImageJ-style bounding-box oval at pixel centers."""
    height, width = shape
    x, y, roi_width, roi_height = roi
    if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0:
        raise ValueError(f"Invalid oval ROI: {roi}")
    if x + roi_width > width or y + roi_height > height:
        raise ValueError(
            f"Oval ROI {roi} is outside image bounds {width}x{height}"
        )
    yy, xx = np.ogrid[:height, :width]
    center_x = x + roi_width / 2.0
    center_y = y + roi_height / 2.0
    return (
        ((xx + 0.5 - center_x) / (roi_width / 2.0)) ** 2
        + ((yy + 0.5 - center_y) / (roi_height / 2.0)) ** 2
        <= 1.0
    )


def _comparison_roi_mask(
    hr_shape: tuple[int, int],
    hr_roi: tuple[int, int, int, int] | None,
    lr_roi: tuple[int, int, int, int] | None,
) -> np.ndarray | None:
    """Intersect the HR oval with the x2 oval mapped onto the HR grid."""
    if hr_roi is None:
        return None
    if lr_roi is None:
        raise ValueError("LR oval must be supplied when HR oval is enabled")
    hr_height, hr_width = hr_shape
    if hr_height % FACTOR or hr_width % FACTOR:
        raise ValueError(f"HR shape is not divisible by x{FACTOR}: {hr_shape}")
    hr_mask = _oval_mask(hr_shape, hr_roi)
    lr_shape = (hr_height // FACTOR, hr_width // FACTOR)
    lr_mask = _oval_mask(lr_shape, lr_roi)
    lr_mask_on_hr = cv2.resize(
        lr_mask.astype(np.uint8),
        (hr_width, hr_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    common = hr_mask & lr_mask_on_hr
    if not np.any(common):
        raise ValueError("HR and LR oval ROIs have no common pixels")
    return common


def _roi_suffix(
    hr_roi: tuple[int, int, int, int] | None,
    lr_roi: tuple[int, int, int, int] | None = None,
) -> str:
    if hr_roi is None:
        return ""
    hx, hy, hw, hh = hr_roi
    lx, ly, lw, lh = lr_roi if lr_roi is not None else (0, 0, 0, 0)
    return (
        f"_hrOval_x{hx}_y{hy}_w{hw}_h{hh}"
        f"_lrOval_x{lx}_y{ly}_w{lw}_h{lh}"
    )


def _parse_method_labels(entries: list[str], option_name: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"{option_name} expects METHOD=VALUE, got: {entry}")
        method, value = entry.split("=", 1)
        method = method.strip().lower()
        if method not in SEGMENTATION_METHODS:
            raise ValueError(
                f"Unknown method '{method}' in {option_name}; expected one "
                f"of {SEGMENTATION_METHODS}"
            )
        if method in labels:
            raise ValueError(f"Duplicate {option_name} for method: {method}")
        labels[method] = int(value)
    return labels


def _parse_pore_labels(entries: list[str]) -> dict[str, int]:
    return _parse_method_labels(entries, "--pore-label")


def _parse_background_labels(entries: list[str]) -> dict[str, int]:
    return _parse_method_labels(entries, "--background-label")


def _resolve_pore_label(
    hr_labels: np.ndarray,
    lr_labels: np.ndarray,
    override: int | None,
    dark_is_pore: bool,
    method: str,
    background_label: int | None = None,
) -> int:
    """Pick which label value is 'pore', resolved once from HR and reused
    for both HR and LR so the two sides use an identical class definition."""
    unique_hr = set(int(v) for v in np.unique(hr_labels))
    unique_lr = set(int(v) for v in np.unique(lr_labels))
    if unique_hr != unique_lr:
        raise ValueError(
            f"{method}: HR/LR label values differ: "
            f"HR={sorted(unique_hr)}, LR={sorted(unique_lr)}. "
            "Export both resolutions with identical label encoding."
        )
    if background_label is not None:
        if background_label not in unique_hr:
            raise ValueError(
                f"{method}: background label {background_label} does not "
                f"exist; available labels are {sorted(unique_hr)}"
            )
        phase_labels = unique_hr - {background_label}
    else:
        phase_labels = unique_hr
    if len(phase_labels) < 2:
        raise ValueError(
            f"{method}: segmentation collapsed to {sorted(phase_labels)} "
            "after background exclusion; both pore and matrix are required."
        )
    if override is not None:
        if override not in phase_labels:
            raise ValueError(
                f"{method}: pore label {override} does not exist as a valid "
                f"phase; available phase labels are {sorted(phase_labels)}"
            )
        return override
    unique = sorted(phase_labels)
    if len(unique) > 2:
        raise ValueError(
            f"{method}: label image has {len(unique)} non-background values "
            f"{unique}; pass --pore-label {method}=<value> to identify "
            "the pore class."
        )
    return unique[0] if dark_is_pore else unique[1]


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def _read_label_image(path: Path) -> np.ndarray:
    labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise FileNotFoundError(f"Cannot read label image: {path}")
    if labels.ndim != 2:
        raise ValueError(
            f"Expected a single-channel label image: {path} has shape {labels.shape}"
        )
    return labels


def _find_single_label_file(directory: Path) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    label_paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in LABEL_EXTENSIONS
    )
    if not label_paths:
        raise FileNotFoundError(f"No label image found in {directory}")
    if len(label_paths) > 1:
        raise RuntimeError(
            f"Expected exactly one label image in {directory}, found "
            f"{[p.name for p in label_paths]}"
        )
    return label_paths[0]


def prepare_single_image_x2(input_path: Path, output_dir: Path) -> None:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read input image: {input_path}")
    if image.ndim != 2:
        raise ValueError(
            f"Expected a single-channel grayscale image, found {image.shape}"
        )

    source_height, source_width = image.shape
    aligned_height = source_height - source_height % FACTOR
    aligned_width = source_width - source_width % FACTOR
    if aligned_height < FACTOR or aligned_width < FACTOR:
        raise ValueError(f"Image is too small for x{FACTOR}: {image.shape}")

    # Crop only the bottom/right remainder so every LR pixel maps exactly to a
    # 2x2 HR block. The source file itself is never modified.
    hr_image = image[:aligned_height, :aligned_width]
    lr_image = cv2.resize(
        hr_image,
        (aligned_width // FACTOR, aligned_height // FACTOR),
        interpolation=cv2.INTER_AREA,
    )

    hr_name = f"{input_path.stem}_hr_aligned.png"
    lr_name = f"{input_path.stem}_x2.png"
    hr_path = output_dir / "hr" / "imagej_input" / hr_name
    lr_path = output_dir / "x2" / "imagej_input" / lr_name
    save_image(hr_path, hr_image)
    save_image(lr_path, lr_image)

    segmentation_dirs = {}
    for resolution in ("hr", "x2"):
        segmentation_dirs[resolution] = {}
        for method in SEGMENTATION_METHODS:
            directory = output_dir / "segmentations" / resolution / method
            directory.mkdir(parents=True, exist_ok=True)
            segmentation_dirs[resolution][method] = str(directory)
    for method in SEGMENTATION_METHODS:
        (output_dir / "upsampled_to_hr" / method).mkdir(
            parents=True, exist_ok=True
        )

    protocol = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "experiment": "single_image_x2_imagej_segmentation",
        "source_image": str(input_path),
        "source_dtype": str(image.dtype),
        "source_shape_hw": [source_height, source_width],
        "aligned_hr_shape_hw": [aligned_height, aligned_width],
        "x2_shape_hw": [aligned_height // FACTOR, aligned_width // FACTOR],
        "factor": FACTOR,
        "metric_roi": {
            "shape": "oval",
            "hr_imagej": "makeOval(8, 32, 1994, 1984)",
            "hr_xywh": list(DEFAULT_HR_OVAL),
            "x2_imagej": "makeOval(5, 16, 993, 992)",
            "x2_xywh": list(DEFAULT_LR_OVAL),
            "comparison_grid": "aligned HR",
            "comparison_domain": "intersection of HR oval and x2 oval mapped to HR",
        },
        "alignment_crop": {
            "bottom_rows_removed": source_height - aligned_height,
            "right_columns_removed": source_width - aligned_width,
        },
        "downsampling_interpolation": "cv2.INTER_AREA",
        "hr_imagej_input": str(hr_path),
        "x2_imagej_input": str(lr_path),
        "segmentation_methods": list(SEGMENTATION_METHODS),
        "segmentation_output_directories": segmentation_dirs,
        "imagej_export_requirements": [
            "single-channel hard-label image",
            "no overlay, screenshot, scale bar, or annotation",
            "do not resize, crop, rotate, or pad in ImageJ",
            "use consistent class polarity between HR and x2",
        ],
        "future_label_upsampling": (
            "Use nearest-neighbor interpolation to explicit aligned HR size "
            f"({aligned_width}x{aligned_height}); do not use linear/cubic/Lanczos."
        ),
    }
    protocol_path = output_dir / "single_image_x2_protocol.json"
    with protocol_path.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)

    print(f"Source:     {input_path}")
    print(f"             {source_width}x{source_height}, {image.dtype}")
    print(f"Aligned HR: {hr_path}")
    print(f"             {aligned_width}x{aligned_height}")
    print(f"x2 image:   {lr_path}")
    print(f"             {aligned_width // FACTOR}x{aligned_height // FACTOR}")
    print(f"Methods:     {', '.join(SEGMENTATION_METHODS)}")
    print(f"Protocol:    {protocol_path}")


def upsample_x2_labels(output_dir: Path) -> None:
    """Nearest-neighbor upsample completed x2 hard-label maps to HR size."""
    output_dir = output_dir.resolve()
    protocol_path = output_dir / "single_image_x2_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Run the preparation step first; missing {protocol_path}"
        )
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    hr_height, hr_width = protocol["aligned_hr_shape_hw"]
    lr_height, lr_width = protocol["x2_shape_hw"]

    processed = 0
    for method in SEGMENTATION_METHODS:
        source_dir = output_dir / "segmentations" / "x2" / method
        destination_dir = output_dir / "upsampled_to_hr" / method
        destination_dir.mkdir(parents=True, exist_ok=True)
        label_paths = sorted(
            path for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in LABEL_EXTENSIONS
        )
        if not label_paths:
            print(f"[SKIP] No x2 {method} label image in {source_dir}")
            continue

        for label_path in label_paths:
            labels = _read_label_image(label_path)
            if labels.shape != (lr_height, lr_width):
                raise ValueError(
                    f"Wrong x2 label size for {label_path}: {labels.shape}; "
                    f"expected {(lr_height, lr_width)}"
                )

            input_labels = set(int(value) for value in np.unique(labels))
            upsampled = cv2.resize(
                labels,
                (hr_width, hr_height),
                interpolation=cv2.INTER_NEAREST,
            )
            output_labels = set(int(value) for value in np.unique(upsampled))
            if output_labels != input_labels:
                raise RuntimeError(
                    f"Label values changed during upsampling: "
                    f"{input_labels} -> {output_labels}"
                )

            destination = (
                destination_dir / f"{label_path.stem}_upsampled_to_hr.png"
            )
            save_image(destination, upsampled)
            print(
                f"{method}: {label_path.name} -> {destination.name}; "
                f"labels={sorted(input_labels)}"
            )
            processed += 1

    if processed == 0:
        raise RuntimeError(
            "No x2 ImageJ label maps found. Put them under "
            "segmentations/x2/{otsu,midgrey,weka}/ first."
        )
    print(
        f"Upsampled {processed} label image(s) to "
        f"{hr_width}x{hr_height} with cv2.INTER_NEAREST"
    )


def _hr_vs_upsampled_pairs(output_dir: Path):
    """Yield (method, hr_labels, lr_labels, hr_path, lr_path) for each method
    that has both an HR label image and an upsampled x2 label image."""
    for method in SEGMENTATION_METHODS:
        hr_dir = output_dir / "segmentations" / "hr" / method
        upsampled_dir = output_dir / "upsampled_to_hr" / method
        try:
            hr_path = _find_single_label_file(hr_dir)
            lr_path = _find_single_label_file(upsampled_dir)
        except FileNotFoundError as exc:
            print(f"[SKIP] {method}: {exc}")
            continue
        yield method, _read_label_image(hr_path), _read_label_image(lr_path), hr_path, lr_path


def _pixelwise_metrics(
    hr_labels: np.ndarray,
    lr_labels: np.ndarray,
    background_label: int | None,
    pore_label: int | None = None,
    roi_mask: np.ndarray | None = None,
) -> list[dict]:
    """Per-class Dice/IoU/precision/recall, treating hr_labels as the
    (method-specific) reference and lr_labels as the prediction to score.

    The valid domain is defined by HR alone (hr_labels != background_label).
    If lr_labels disagrees with HR about which pixels are background --
    e.g. HR says pore but the upsampled x2 result says background -- those
    pixels stay in-domain and count as errors instead of being silently
    excluded.
    """
    if hr_labels.shape != lr_labels.shape:
        raise ValueError(
            f"Shape mismatch: HR {hr_labels.shape} vs upsampled x2 {lr_labels.shape}"
        )

    if background_label is not None:
        valid = hr_labels != background_label
    else:
        valid = np.ones_like(hr_labels, dtype=bool)
    if roi_mask is not None:
        if roi_mask.shape != hr_labels.shape:
            raise ValueError(
                f"ROI mask shape {roi_mask.shape} != labels {hr_labels.shape}"
            )
        valid &= roi_mask
    if not np.any(valid):
        raise ValueError("The selected ROI contains no valid evaluation pixels")

    class_values = sorted(
        (set(int(v) for v in np.unique(hr_labels[valid]))
         | set(int(v) for v in np.unique(lr_labels[valid])))
        - ({background_label} if background_label is not None else set())
    )

    valid_count = np.count_nonzero(valid)
    pixel_accuracy = (
        np.count_nonzero((hr_labels == lr_labels) & valid) / valid_count
        if valid_count > 0 else float("nan")
    )

    records = []
    for value in class_values:
        gt = (hr_labels == value) & valid
        pred = (lr_labels == value) & valid
        intersection = np.count_nonzero(gt & pred)
        gt_count = np.count_nonzero(gt)
        pred_count = np.count_nonzero(pred)
        union = np.count_nonzero(gt | pred)
        records.append({
            "label": value,
            "is_pore": pore_label is not None and value == pore_label,
            "dice": (2 * intersection / (gt_count + pred_count)) if (gt_count + pred_count) > 0 else float("nan"),
            "iou": (intersection / union) if union > 0 else float("nan"),
            "precision": (intersection / pred_count) if pred_count > 0 else float("nan"),
            "recall": (intersection / gt_count) if gt_count > 0 else float("nan"),
            "gt_pixels": int(gt_count),
            "pred_pixels": int(pred_count),
            "pixel_accuracy": pixel_accuracy,
        })
    return records


def compare_pixelwise(
    output_dir: Path,
    background_labels: dict[str, int] | None = None,
    pore_labels: dict[str, int] | None = None,
    dark_is_pore: bool = True,
    roi: tuple[int, int, int, int] | None = DEFAULT_HR_OVAL,
    lr_roi: tuple[int, int, int, int] | None = DEFAULT_LR_OVAL,
) -> pd.DataFrame:
    """Dice/IoU/precision/recall of upsampled x2 labels vs each method's own
    HR labels (a resolution-consistency reference, not an independent
    ground truth -- see the module docstring).

    Answers "how much did downsampling-then-segmenting-then-upsampling
    change the segmentation, pixel for pixel" for each method. Reports the
    pore class specifically as the headline number, since a macro-average
    over all classes is dominated by the (usually larger) matrix class and
    can hide small-pore loss.
    """
    output_dir = output_dir.resolve()
    pore_labels = pore_labels or {}
    background_labels = background_labels or {}
    rows = []
    for method, hr_labels, lr_labels, hr_path, lr_path in _hr_vs_upsampled_pairs(output_dir):
        roi_mask = _comparison_roi_mask(hr_labels.shape, roi, lr_roi)
        background_label = background_labels.get(method)
        pore_label = _resolve_pore_label(
            hr_labels, lr_labels, pore_labels.get(method), dark_is_pore,
            method, background_label
        )

        for record in _pixelwise_metrics(
            hr_labels, lr_labels, background_label, pore_label, roi_mask
        ):
            record["method"] = method
            record["hr_file"] = hr_path.name
            record["upsampled_lr_file"] = lr_path.name
            rows.append(record)

    if not rows:
        raise RuntimeError(
            "No method had both an HR label image and an upsampled x2 label "
            "image. Segment both resolutions in ImageJ and run "
            "--upsample-labels first."
        )

    df = pd.DataFrame(rows)[
        ["method", "label", "is_pore", "dice", "iou", "precision", "recall",
         "pixel_accuracy", "gt_pixels", "pred_pixels", "hr_file", "upsampled_lr_file"]
    ]
    csv_path = output_dir / f"pixelwise_agreement{_roi_suffix(roi, lr_roi)}.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== Pixel-wise agreement: HR (method-specific reference) vs upsampled x2 ===")
    print(df.to_string(index=False))

    pore_df = df[df["is_pore"]]
    if not pore_df.empty:
        print("\n=== PRIMARY: pore-class agreement (report this for method ranking) ===")
        print(pore_df.set_index("method")[["dice", "iou", "precision", "recall"]].to_string())
    else:
        print("\n[WARN] No method had a resolvable pore label; only the "
              "macro/per-class table above is available. Pass --pore-label "
              "to enable the pore-specific summary.")

    macro_summary = df.groupby("method")[["dice", "iou"]].mean().rename(
        columns={"dice": "macro_mean_dice", "iou": "macro_mean_iou"})
    macro_summary["pixel_accuracy"] = df.groupby("method")["pixel_accuracy"].first()
    print("\n=== Supplementary: macro-average over ALL classes (includes "
          "matrix; can mask small-pore loss -- do not use this alone to rank methods) ===")
    print(macro_summary)
    print(f"\nSaved: {csv_path}")
    return df


def compare_structural_stats(
    output_dir: Path,
    pore_labels: dict[str, int] | None = None,
    background_labels: dict[str, int] | None = None,
    dark_is_pore: bool = True,
    rel_diff_zero_tol: float = 1e-9,
    roi: tuple[int, int, int, int] | None = DEFAULT_HR_OVAL,
    lr_roi: tuple[int, int, int, int] | None = DEFAULT_LR_OVAL,
) -> pd.DataFrame:
    """Porosity/perimeter/Euler-number density of upsampled x2 vs HR, per
    method, as absolute and (where well-defined) relative %% deviation
    (reuses minkowski_functionals_2d.py).

    This reports a single-slice structural *deviation*, not a systematic
    bias -- that label requires aggregating mean +/- std over multiple
    independent slices, which this single-image script does not do. It also
    does not measure pixel-exact spatial agreement (see compare_pixelwise
    for that).
    """
    output_dir = output_dir.resolve()
    pore_labels = pore_labels or {}
    background_labels = background_labels or {}
    rows = []
    for method, hr_labels, lr_labels, _hr_path, _lr_path in _hr_vs_upsampled_pairs(output_dir):
        background_label = background_labels.get(method)
        pore_label = _resolve_pore_label(
            hr_labels, lr_labels, pore_labels.get(method), dark_is_pore,
            method, background_label
        )

        # A common, HR-defined valid domain for both resolutions, so any
        # porosity/perimeter/Euler difference reflects only how the pore
        # phase itself was segmented -- not a shift in the rock/background
        # boundary between the two label images.
        common_valid = (
            hr_labels != background_label if background_label is not None
            else np.ones_like(hr_labels, dtype=bool)
        )
        roi_mask = _comparison_roi_mask(hr_labels.shape, roi, lr_roi)
        if roi_mask is not None:
            common_valid &= roi_mask
        if not np.any(common_valid):
            raise ValueError(
                f"{method}: selected ROI contains no valid structural pixels"
            )
        hr_pore, _ = mkf.build_masks(hr_labels, pore_label=pore_label, background_label=None)
        lr_pore, _ = mkf.build_masks(lr_labels, pore_label=pore_label, background_label=None)

        hr_metrics = mkf.compute_minkowski_functionals_2d(hr_pore, common_valid)
        lr_metrics = mkf.compute_minkowski_functionals_2d(lr_pore, common_valid)

        row = {"method": method}
        for key in ("porosity", "perimeter_density", "euler_density"):
            hr_value, lr_value = hr_metrics[key], lr_metrics[key]
            abs_diff = lr_value - hr_value
            row[f"hr_{key}"] = hr_value
            row[f"upsampled_lr_{key}"] = lr_value
            row[f"{key}_abs_diff"] = abs(abs_diff)
            # euler_density is a topological invariant / valid_area: when HR
            # is exactly (or near) 0, any nonzero change is a divide-by-zero
            # blowup, not a meaningful percentage -- report NaN instead.
            if abs(hr_value) < rel_diff_zero_tol:
                relative_diff = float("nan")
                absolute_relative_error = float("nan")
            else:
                relative_diff = abs_diff / abs(hr_value) * 100
                absolute_relative_error = abs(abs_diff) / abs(hr_value) * 100
            row[f"{key}_rel_diff_%"] = relative_diff
            row[f"{key}_abs_rel_error_%"] = absolute_relative_error
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No method had both an HR label image and an upsampled x2 label "
            "image. Segment both resolutions in ImageJ and run "
            "--upsample-labels first."
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"structural_stats_deviation{_roi_suffix(roi, lr_roi)}.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== Structural deviation (single slice): HR vs upsampled x2 ===")
    print(df.to_string(index=False))
    print("(Absolute diff always shown; relative %% is NaN where HR is ~0, "
          "e.g. euler_density with no holes/components. This is a single-slice "
          "deviation, not a systematic-bias estimate -- aggregate mean +/- std "
          "over multiple independent slices for that claim.)")
    print(f"\nSaved: {csv_path}")
    return df


def save_method_comparison_summary(
    output_dir: Path,
    pixel_df: pd.DataFrame | None,
    structural_df: pd.DataFrame | None,
    roi: tuple[int, int, int, int] | None = DEFAULT_HR_OVAL,
    lr_roi: tuple[int, int, int, int] | None = DEFAULT_LR_OVAL,
) -> pd.DataFrame:
    """Save headline pore agreement and absolute structural errors."""
    parts = []
    if pixel_df is not None:
        pore = pixel_df[pixel_df["is_pore"]][
            ["method", "dice", "iou", "precision", "recall"]
        ].rename(columns={
            "dice": "pore_dice",
            "iou": "pore_iou",
            "precision": "pore_precision",
            "recall": "pore_recall",
        })
        if pore["method"].duplicated().any():
            raise RuntimeError("Expected exactly one pore-class row per method")
        parts.append(pore)
    if structural_df is not None:
        structural_columns = [
            "method",
            "porosity_abs_rel_error_%",
            "perimeter_density_abs_rel_error_%",
            "euler_density_abs_diff",
            "euler_density_abs_rel_error_%",
        ]
        parts.append(structural_df[structural_columns])
    if not parts:
        raise ValueError("No comparison results supplied for summary")

    summary = parts[0]
    for part in parts[1:]:
        summary = summary.merge(part, on="method", how="outer", validate="one_to_one")
    summary = summary.sort_values("method").reset_index(drop=True)
    summary_path = (
        output_dir.resolve()
        / f"method_comparison_summary{_roi_suffix(roi, lr_roi)}.csv"
    )
    summary.to_csv(summary_path, index=False)
    print("\n=== Method comparison summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    if args.upsample_labels:
        upsample_x2_labels(args.output_dir)
    elif args.compare_pixelwise or args.compare_structural:
        pore_labels = _parse_pore_labels(args.pore_label)
        background_labels = _parse_background_labels(args.background_label)
        dark_is_pore = not args.light_is_pore
        roi = None if args.full_image else tuple(args.roi_oval)
        lr_roi = None if args.full_image else tuple(args.lr_roi_oval)
        pixel_df = None
        structural_df = None
        if args.compare_pixelwise:
            pixel_df = compare_pixelwise(
                args.output_dir,
                background_labels=background_labels,
                pore_labels=pore_labels,
                dark_is_pore=dark_is_pore,
                roi=roi,
                lr_roi=lr_roi,
            )
        if args.compare_structural:
            structural_df = compare_structural_stats(
                args.output_dir,
                pore_labels=pore_labels,
                background_labels=background_labels,
                dark_is_pore=dark_is_pore,
                rel_diff_zero_tol=args.rel_diff_zero_tol,
                roi=roi,
                lr_roi=lr_roi,
            )
        save_method_comparison_summary(
            args.output_dir, pixel_df, structural_df, roi, lr_roi
        )
    else:
        prepare_single_image_x2(args.input, args.output_dir)


if __name__ == "__main__":
    main()
