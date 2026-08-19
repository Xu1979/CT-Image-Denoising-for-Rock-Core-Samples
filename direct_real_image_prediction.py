"""Denoise real images with deep-learning models or traditional methods."""

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from FCN_complete import get_pretrained_model
from SwinUNet_complete import SwinUNet
from history_visual_prediction import merge_patches, split_into_patches
from traditional_denoising_pipeline import SAP_FILTERS, denoise_one_image
from unet_complete import UNet


SCRIPT_DIR = Path(__file__).resolve().parent
# Accepts either a directory (batch prediction) or one specific image file.
# Examples:
#   SCRIPT_DIR / "real_image"
#   SCRIPT_DIR / "real_image" / "sample.webp"
# INPUT_DIR = SCRIPT_DIR / "real_image"
INPUT_DIR = (
    SCRIPT_DIR
    / "real_image"
    / "BHG0_z0368_tomo.png"
)
OUTPUT_ROOT = SCRIPT_DIR / "real_image_direct_denoised"
ORIGINAL_PNG_DIR = OUTPUT_ROOT / "original_png"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
TIFF_LOWER_PERCENTILE = 1.0
TIFF_UPPER_PERCENTILE = 99.0

# Number of patches sent through the model per forward pass. Lower this if
# you run out of GPU memory (e.g. 16 or 8).
INFERENCE_BATCH_SIZE = 32

# Choose the prediction branch used by main():
#   "deep_learning" -> UNet, SwinUNet and FCN-ResNet50
#   "traditional"   -> beam-hardening, ring and/or SAP filters
PREDICTION_MODE = "traditional"

# Traditional branch configuration. Select the artifacts known to be present
# in the real images. Valid names are "bh", "ring" and "sap". They are applied
# in the same order as traditional_denoising_pipeline.py: BH -> Ring -> SAP.
TRADITIONAL_NOISE_TYPES = {"ring"}

# Used only when "sap" is selected above. Choose any subset of:
# "snn", "gaussian", "median", "mean", "nlm".
# Each selected filter gets a separate output directory.
TRADITIONAL_SAP_METHODS = ["snn"]

MODEL_CONFIGS = [
    {
        "name": "unet",
        "checkpoint": SCRIPT_DIR / "newest_result" / "unet_best_model.pth",
        "patch_size": 256,
        "stride": 64,
    },
    {
        "name": "swinunet",
        "checkpoint": SCRIPT_DIR / "newest_result" / "swinunet_best_denoising_model.pth",
        "patch_size": 224,
        "stride": 56,
    },
    {
        "name": "fcn_resnet50",
        "checkpoint": SCRIPT_DIR / "newest_result" / "fcn_best_model.pth",
        "patch_size": 256,
        "stride": 64,
    },
]


def build_model(model_name, device):
    if model_name == "unet":
        model = UNet(in_channels=1, out_channels=1)
    elif model_name == "swinunet":
        model = SwinUNet(pretrained=False, img_size=224, num_classes=1)
    elif model_name == "fcn_resnet50":
        model = get_pretrained_model("fcn_resnet50")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # These project checkpoints also contain NumPy training metrics, so they
    # must be loaded as complete checkpoints rather than weights-only files.
    # Only use this with checkpoints you created or otherwise trust.
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()


def load_grayscale_image(image_path):
    """Load an image as 8-bit grayscale without clipping 16-bit TIFF data."""
    with Image.open(image_path) as source:
        source.load()

        # Pillow's direct I;16 -> L conversion clips values above 255. Apply a
        # robust intensity window first, then map that window to 0..255.
        if source.mode in {"I;16", "I;16L", "I;16B", "I", "F"}:
            values = np.asarray(source, dtype=np.float32)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                raise ValueError(f"Image has no finite pixels: {image_path}")

            low, high = np.percentile(
                finite_values,
                [TIFF_LOWER_PERCENTILE, TIFF_UPPER_PERCENTILE],
            )
            if high <= low:
                low = float(finite_values.min())
                high = float(finite_values.max())

            if high > low:
                values = np.clip(values, low, high)
                values = (values - low) / (high - low)
            else:
                values = np.zeros_like(values, dtype=np.float32)

            grayscale = Image.fromarray(
                np.round(values * 255.0).astype(np.uint8), mode="L"
            )
            tqdm.write(
                f"  {image_path.name}: high-bit-depth window "
                f"[{low:.1f}, {high:.1f}] -> [0, 255]"
            )
            return grayscale

        return source.convert("L").copy()


def _aligned_size(size, patch_size, stride):
    """Smallest dimension >= size where (dim - patch_size) % stride == 0."""
    if size <= patch_size:
        return patch_size
    remainder = (size - patch_size) % stride
    if remainder == 0:
        return size
    return size + (stride - remainder)


def pad_to_grid(image, patch_size, stride):
    """Reflect-pad an image so the sliding-window grid tiles it exactly.

    Real images rarely satisfy (dim - patch_size) % stride == 0. Without
    padding, the patch splitter either produces undersized edge patches
    (which break batched inference and models with size constraints) or
    leaves an uncovered strip at the border. Padding to the aligned size
    guarantees every patch is a full patch_size x patch_size tile.

    Returns the (possibly padded) image and the original (width, height) so
    the prediction can be cropped back afterwards.
    """
    width, height = image.size
    target_w = _aligned_size(width, patch_size, stride)
    target_h = _aligned_size(height, patch_size, stride)
    if (target_w, target_h) == (width, height):
        return image, (width, height)

    pad_right = target_w - width
    pad_bottom = target_h - height

    array = np.asarray(image)
    # np.pad reflect mode requires pad < dimension size; fall back to edge
    # padding for extremely small images.
    mode = "reflect"
    if pad_bottom >= height or pad_right >= width:
        mode = "edge"
    padded = np.pad(array, ((0, pad_bottom), (0, pad_right)), mode=mode)
    return Image.fromarray(padded, mode="L"), (width, height)


def predict_with_patches(model, image, patch_size, stride, device):
    """Predict overlapping patches and blend them at the original resolution."""
    # Pad so the patch grid tiles the image exactly (every patch is full
    # size and no border strip is left uncovered). Cropped back at the end.
    image, original_size = pad_to_grid(image, patch_size, stride)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    patches, positions, full_size = split_into_patches(
        image, patch_size=patch_size, stride=stride
    )

    denoised_patches = []
    with torch.inference_mode():
        for start in range(0, len(patches), INFERENCE_BATCH_SIZE):
            batch = patches[start : start + INFERENCE_BATCH_SIZE]
            tensors = torch.stack([transform(p) for p in batch]).to(device)

            prediction = model(tensors)
            if isinstance(prediction, dict):
                prediction = prediction["out"]

            # Models end with Tanh (range [-1, 1]); map back to [0, 1].
            prediction = (prediction.float().cpu().numpy() + 1.0) / 2.0
            prediction = np.clip(prediction, 0.0, 1.0)
            for patch_pred in prediction:
                denoised_patches.append(patch_pred.squeeze(0))

    merged = merge_patches(
        denoised_patches,
        positions,
        full_size,
        patch_size=patch_size,
        stride=stride,
        margin_ratio=0.3,
    )

    # Round (not truncate) when quantizing, and clip in case blending
    # produced values slightly outside [0, 1].
    merged_u8 = np.round(np.clip(merged, 0.0, 1.0) * 255.0).astype(np.uint8)
    result = Image.fromarray(merged_u8, mode="L")

    # Crop back to the original size if the input was padded.
    if result.size != original_size:
        result = result.crop((0, 0, original_size[0], original_size[1]))

    return result, len(patches)


def save_in_original_format(image, output_path):
    """Save with the source extension and suitable quality/compression."""
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95, subsampling=0)
    elif suffix == ".webp":
        image.save(output_path, lossless=True, quality=100)
    elif suffix in {".tif", ".tiff"}:
        # Do NOT pass compression="tiff_deflate" here: it routes through
        # Pillow's libtiff encoder, which crashes the process (access
        # violation, no traceback) on some Windows Pillow builds.
        # Uncompressed TIFF uses Pillow's own safe writer.
        image.save(output_path)
    else:  # PNG
        image.save(output_path)


def prediction_output_name(image_path):
    """Use PNG output for WebP inputs; preserve other input extensions."""
    if image_path.suffix.lower() == ".webp":
        return f"{image_path.stem}.png"
    return image_path.name


def export_webp_originals_as_png(image_paths):
    """Save lossless PNG copies of WebP inputs without modifying the sources."""
    webp_paths = [path for path in image_paths if path.suffix.lower() == ".webp"]
    if not webp_paths:
        return

    ORIGINAL_PNG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(webp_paths)} original WebP image(s) to PNG")
    for image_path in tqdm(webp_paths, desc="original_webp_to_png"):
        grayscale = load_grayscale_image(image_path)
        output_path = ORIGINAL_PNG_DIR / f"{image_path.stem}.png"
        grayscale.save(output_path)
        tqdm.write(f"  Original: {image_path.name} -> {output_path}")


def run_model(config, image_paths, device):
    model_name = config["name"]
    output_dir = OUTPUT_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print(
        f"Patch size: {config['patch_size']}, "
        f"stride: {config['stride']}"
    )
    print(f"Output: {output_dir}")
    print("=" * 70)

    model = build_model(model_name, device)
    load_checkpoint(model, config["checkpoint"], device)

    for image_path in tqdm(image_paths, desc=model_name):
        grayscale = load_grayscale_image(image_path)
        denoised, patch_count = predict_with_patches(
            model,
            grayscale,
            config["patch_size"],
            config["stride"],
            device,
        )

        # Preserve the source type except for WebP, which is exported as PNG.
        # Each model has its own directory, so the source is never overwritten.
        output_path = output_dir / prediction_output_name(image_path)
        save_in_original_format(denoised, output_path)
        tqdm.write(
            f"  {image_path.name}: {patch_count} patches -> {output_path.name}"
        )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def find_input_images():
    """Return one input image or all supported images directly in a directory."""
    if INPUT_DIR.is_file():
        if INPUT_DIR.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported input image type: {INPUT_DIR.suffix}; "
                f"supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return [INPUT_DIR]

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input path not found: {INPUT_DIR}")

    image_paths = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {INPUT_DIR}")
    return image_paths


def run_deep_learning_prediction(image_paths):
    """Run all configured trained models."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Deep-learning prediction on {len(image_paths)} real images")

    for config in MODEL_CONFIGS:
        run_model(config, image_paths, device)

    print(f"\nDeep-learning prediction done. Results: {OUTPUT_ROOT}")


def validate_traditional_config():
    valid_noise_types = {"bh", "ring", "sap"}
    if not TRADITIONAL_NOISE_TYPES:
        raise ValueError(
            "TRADITIONAL_NOISE_TYPES is empty; select at least one of "
            f"{sorted(valid_noise_types)}"
        )
    unknown_noise_types = set(TRADITIONAL_NOISE_TYPES) - valid_noise_types
    if unknown_noise_types:
        raise ValueError(
            f"Unknown traditional noise types: {sorted(unknown_noise_types)}; "
            f"choose from {sorted(valid_noise_types)}"
        )

    unknown_methods = set(TRADITIONAL_SAP_METHODS) - set(SAP_FILTERS)
    if unknown_methods:
        raise ValueError(
            f"Unknown SAP methods: {sorted(unknown_methods)}; "
            f"choose from {sorted(SAP_FILTERS)}"
        )
    if "sap" in TRADITIONAL_NOISE_TYPES and not TRADITIONAL_SAP_METHODS:
        raise ValueError("Select at least one TRADITIONAL_SAP_METHOD when using SAP")


def run_traditional_method(method_name, sap_filter, image_paths):
    """Apply one traditional pipeline configuration to every input image."""
    output_dir = OUTPUT_ROOT / f"traditional_{method_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Traditional method: {method_name}")
    print(f"Noise/artifact types: {sorted(TRADITIONAL_NOISE_TYPES)}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    for image_path in tqdm(image_paths, desc=f"traditional_{method_name}"):
        grayscale = load_grayscale_image(image_path)
        image_u8 = np.asarray(grayscale, dtype=np.uint8)
        denoised = denoise_one_image(
            image_u8,
            set(TRADITIONAL_NOISE_TYPES),
            sap_filter,
        )
        # denoise_one_image returns [0, 1] float data for the completed
        # pipeline. Convert it back to an 8-bit PIL image for saving.
        if np.issubdtype(np.asarray(denoised).dtype, np.floating):
            denoised_u8 = np.round(
                np.clip(denoised, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
        else:
            denoised_u8 = np.asarray(denoised, dtype=np.uint8)

        output_path = output_dir / prediction_output_name(image_path)
        save_in_original_format(Image.fromarray(denoised_u8, mode="L"), output_path)
        tqdm.write(f"  {image_path.name} -> {output_path}")


def run_traditional_prediction(image_paths):
    """Run the selected traditional artifact-removal/filter pipelines."""
    validate_traditional_config()
    print(f"Traditional prediction on {len(image_paths)} real images")

    if "sap" in TRADITIONAL_NOISE_TYPES:
        for method_name in TRADITIONAL_SAP_METHODS:
            run_traditional_method(method_name, SAP_FILTERS[method_name], image_paths)
    else:
        # BH and/or Ring do not need an SAP filter.
        method_name = "_".join(
            noise_type
            for noise_type in ("bh", "ring")
            if noise_type in TRADITIONAL_NOISE_TYPES
        )
        run_traditional_method(method_name, None, image_paths)

    print(f"\nTraditional prediction done. Results: {OUTPUT_ROOT}")


def main():
    image_paths = find_input_images()
    export_webp_originals_as_png(image_paths)

    # ==================== Part 1: deep-learning prediction ====================
    if PREDICTION_MODE == "deep_learning":
        run_deep_learning_prediction(image_paths)

    # ====================== Part 2: traditional methods =======================
    elif PREDICTION_MODE == "traditional":
        run_traditional_prediction(image_paths)

    else:
        raise ValueError(
            f"Unsupported PREDICTION_MODE={PREDICTION_MODE!r}; "
            "use 'deep_learning' or 'traditional'"
        )


if __name__ == "__main__":
    main()
