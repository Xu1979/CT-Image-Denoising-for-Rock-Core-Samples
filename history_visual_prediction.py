import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import json
from tqdm import tqdm
import matplotlib
from FCN_complete import get_pretrained_model
from matplotlib import font_manager
try:
    font_manager.findfont('Arial', fallback_to_default=False)
    matplotlib.rcParams['font.family'] = 'Arial'
except:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 11

# ========================================================================
# 1. Patch processing function
# ========================================================================

def split_into_patches(img, patch_size=256, stride=128):
    """Slice image into a list of patches (with padding)"""
    w, h = img.size
    patches, positions = [], []

    y_positions = list(range(0, h - patch_size + 1, stride))
    x_positions = list(range(0, w - patch_size + 1, stride))

    if len(y_positions) == 0 or y_positions[-1] + patch_size < h:
        y_positions.append(max(0, h - patch_size))
    if len(x_positions) == 0 or x_positions[-1] + patch_size < w:
        x_positions.append(max(0, w - patch_size))

    y_positions = sorted(list(set(y_positions)))
    x_positions = sorted(list(set(x_positions)))

    for top in y_positions:
        for left in x_positions:
            # Compute the actual crop region
            right = min(left + patch_size, w)
            bottom = min(top + patch_size, h)

            box = (left, top, right, bottom)
            patch = img.crop(box)

            # Compute real shape
            actual_w = right - left
            actual_h = bottom - top

            # Padding to patch_size
            padded = Image.new("L", (patch_size, patch_size), color=0)
            padded.paste(patch, (0, 0))

            patches.append(padded)
            positions.append((left, top, actual_w, actual_h))

    return patches, positions, (w, h)


def create_blend_weights(h, w, margin_ratio=0.2):
    """
    Create a weight map that gradually decreases from center to edge 
    (for seamless blending)
    """
    margin_h = max(int(h * margin_ratio), 10)
    margin_w = max(int(w * margin_ratio), 10)

    # Prevent margin from exceeding image dimensions 
    margin_h = min(margin_h, h // 2)
    margin_w = min(margin_w, w // 2)

    weight = np.ones((h, w), dtype=np.float32)

    # Top and bottom edge fade
    if margin_h > 0:
        for i in range(margin_h):
            alpha = (i + 1) / (margin_h + 1)
            weight[i, :] *= alpha
            weight[-(i + 1), :] *= alpha

    # Left and right edge fade
    if margin_w > 0:
        for j in range(margin_w):
            alpha = (j + 1) / (margin_w + 1)
            weight[:, j] *= alpha
            weight[:, -(j + 1)] *= alpha

    return weight


def merge_patches(patches, positions, full_size, patch_size=256, stride=128,
                  margin_ratio=0.2):
    """
    Merge patches (using gradient weights)
    """
    full_w, full_h = full_size
    canvas = np.zeros((full_h, full_w), dtype=np.float32)
    weight_sum = np.zeros((full_h, full_w), dtype=np.float32)

    for patch, pos in zip(patches, positions):
        x, y, actual_w, actual_h = pos

        if isinstance(patch, np.ndarray):
            patch_np = patch
        else:
            patch_np = np.array(patch)

        patch_np = patch_np.squeeze()

        # only use real shape region
        patch_valid = patch_np[:actual_h, :actual_w]

        # create gradient weights
        patch_weight = create_blend_weights(
            actual_h, actual_w,
            margin_ratio=margin_ratio
        )

        # Ensure no out-of-bounds access
        y_end = min(y + actual_h, full_h)
        x_end = min(x + actual_w, full_w)

        # Crop to the actual valid region
        valid_h = y_end - y
        valid_w = x_end - x

        patch_valid = patch_valid[:valid_h, :valid_w]
        patch_weight = patch_weight[:valid_h, :valid_w]

        # accumulate to canvas
        canvas[y:y_end, x:x_end] += patch_valid * patch_weight
        weight_sum[y:y_end, x:x_end] += patch_weight

    # normalization
    weight_sum = np.maximum(weight_sum, 1e-8)
    merged = canvas / weight_sum
    merged = np.clip(merged, 0, 1)

    return merged


# ========================================================================
# 2. training history visualization
# ========================================================================

def plot_loss_curve(json_path, start_epoch=0, end_epoch=None, save_fig=True, ylim=None):
    """plot loss curves"""

    with open(json_path, "r") as f:
        history = json.load(f)

    train_losses = history.get("train_losses") or history.get("train_loss")
    val_losses   = history.get("val_losses")   or history.get("val_loss")

    total_epochs = len(train_losses)
    if end_epoch is None or end_epoch > total_epochs:
        end_epoch = total_epochs

    epochs             = list(range(start_epoch, end_epoch))
    train_losses_slice = train_losses[start_epoch:end_epoch]
    val_losses_slice   = val_losses[start_epoch:end_epoch]

    fig, ax = plt.subplots(figsize=(6, 4)) 

    ax.plot(epochs, train_losses_slice, 'b-', label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses_slice,   'r-', label="Val Loss",   linewidth=1.5)
    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("Loss",  fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, linestyle="--", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        min_loss = min(min(train_losses_slice), min(val_losses_slice))
        max_loss = max(max(train_losses_slice), max(val_losses_slice))
        margin   = (max_loss - min_loss) * 0.1
        ax.set_ylim(min_loss - margin, max_loss + margin)

    fig.tight_layout()

    if save_fig:
        loss_fig_path = json_path.replace('.json', '_loss.png')
        fig.savefig(loss_fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Loss curve saved: {loss_fig_path}")

    plt.show()
    best_val_loss = min(val_losses_slice)
    best_epoch    = val_losses_slice.index(best_val_loss) + start_epoch
    return best_epoch, best_val_loss


def plot_psnr_ssim_curves(json_path, start_epoch=0, end_epoch=None, save_fig=True):
    """
    plot validation PSNR and SSIM
    """

    with open(json_path, "r") as f:
        history = json.load(f)

    val_psnrs = history.get("val_psnrs")
    val_ssims = history.get("val_ssims")

    total_epochs = len(val_psnrs)
    if end_epoch is None or end_epoch > total_epochs:
        end_epoch = total_epochs

    epochs          = list(range(start_epoch, end_epoch))
    val_psnrs_slice = val_psnrs[start_epoch:end_epoch]
    val_ssims_slice = val_ssims[start_epoch:end_epoch]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # PSNR
    ax1.plot(epochs, val_psnrs_slice, color='green', linewidth=1.5)
    ax1.set_xlabel("Epoch", fontweight='bold')
    ax1.set_ylabel("PSNR (dB)", fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.3)

    # SSIM
    ax2.plot(epochs, val_ssims_slice, color='mediumpurple', linewidth=1.5)
    ax2.set_xlabel("Epoch", fontweight='bold')
    ax2.set_ylabel("SSIM", fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout(w_pad=3.0)

    if save_fig:
        out_path = json_path.replace('.json', '_psnr_ssim.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✓ PSNR/SSIM curve saved: {out_path}")

    plt.show()

    best_psnr       = max(val_psnrs_slice)
    best_psnr_epoch = val_psnrs_slice.index(best_psnr) + start_epoch
    best_ssim       = max(val_ssims_slice)
    best_ssim_epoch = val_ssims_slice.index(best_ssim) + start_epoch
    return best_psnr_epoch, best_psnr, best_ssim_epoch, best_ssim


# ──Keep the original single-image function ──
def plot_psnr_curve(json_path, start_epoch=0, end_epoch=None, save_fig=True):
    
    with open(json_path, "r") as f:
        history = json.load(f)
    val_psnrs = history.get("val_psnrs")
    total_epochs = len(val_psnrs)
    if end_epoch is None or end_epoch > total_epochs:
        end_epoch = total_epochs
    epochs          = list(range(start_epoch, end_epoch))
    val_psnrs_slice = val_psnrs[start_epoch:end_epoch]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, val_psnrs_slice, color='green', linewidth=1.5)
    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("PSNR (dB)", fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    if save_fig:
        fig.savefig(json_path.replace('.json', '_psnr.png'), dpi=300, bbox_inches='tight')
    plt.show()
    best_psnr       = max(val_psnrs_slice)
    best_psnr_epoch = val_psnrs_slice.index(best_psnr) + start_epoch
    return best_psnr_epoch, best_psnr


def plot_ssim_curve(json_path, start_epoch=0, end_epoch=None, save_fig=True):
    
    with open(json_path, "r") as f:
        history = json.load(f)
    val_ssims = history.get("val_ssims")
    total_epochs = len(val_ssims)
    if end_epoch is None or end_epoch > total_epochs:
        end_epoch = total_epochs
    epochs          = list(range(start_epoch, end_epoch))
    val_ssims_slice = val_ssims[start_epoch:end_epoch]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, val_ssims_slice, color='mediumpurple', linewidth=1.5)
    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("SSIM", fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    if save_fig:
        fig.savefig(json_path.replace('.json', '_ssim.png'), dpi=300, bbox_inches='tight')
    plt.show()
    best_ssim       = max(val_ssims_slice)
    best_ssim_epoch = val_ssims_slice.index(best_ssim) + start_epoch
    return best_ssim_epoch, best_ssim

# ========================================================================
# 3. model performance evaluation
# ========================================================================

def test_model_on_images(model_path, test_noisy_dir, test_clean_dir=None,
                         output_dir='output_test/fcn_patch_wise',
                         patch_size=256, stride=64, margin_ratio=0.3,
                         model_name='fcn_resnet50', normalize=True,
                         show_progress=True):

    print("Start model inference test...")

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # model = get_pretrained_model(model_name).to(device)
    if model_name == 'unet':
        from unet_complete import UNet
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif model_name == 'swinunet':
        from SwinUNet_complete import SwinUNet
        model = SwinUNet(pretrained=False, img_size=224, num_classes=1).to(device)
    else:
        model = get_pretrained_model(model_name).to(device)


    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"load model from Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"  Val PSNR: {checkpoint.get('val_psnr', 'N/A')} dB")
        print(f"  Val SSIM: {checkpoint.get('val_ssim', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
        print("loading model")

    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    test_files = sorted([f for f in os.listdir(test_noisy_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    print(f" find {len(test_files)} test images")
    print(f"Patch size: {patch_size}, Stride: {stride}")
    print(f"Margin ratio: {margin_ratio} (Gradient weight edge ratio)")
    print(f"output directory: {output_dir}\n")

    # Transform
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    psnr_list, ssim_list = [], []
    iterator = tqdm(test_files, desc="Processing") if show_progress else test_files

    for file in iterator:
        # read image
        noisy_img = Image.open(os.path.join(test_noisy_dir, file)).convert('L')

        original_size = noisy_img.size  # (width, height)

        # patching
        patches, positions, full_size = split_into_patches(
            noisy_img, patch_size=patch_size, stride=stride
        )
        print(f"  {file}: {len(patches)} patches from {full_size}")

        # prediction
        denoised_patches = []
        with torch.no_grad():
            for patch in patches:
                assert patch.size == (patch_size, patch_size), \
                    f"Patch size mismatch: {patch.size} vs {(patch_size, patch_size)}"

                tensor = transform(patch).unsqueeze(0).to(device)
                pred = model(tensor)
                if isinstance(pred, dict):
                    pred = pred['out']

                pred_np = pred.squeeze().cpu().numpy()
                if normalize:
                    pred_np = (pred_np + 1) / 2.0

                pred_np = np.clip(pred_np, 0, 1)
                denoised_patches.append(pred_np)

        # combination
        denoised_np = merge_patches(
            denoised_patches, positions, full_size,
            patch_size=patch_size, stride=stride,
            margin_ratio=margin_ratio
        )
        print(f"  Merged shape: {denoised_np.shape}, Expected: {full_size[::-1]}")

        # save
        denoised_img = Image.fromarray((denoised_np * 255).astype(np.uint8))
        denoised_img.save(os.path.join(output_dir, file))

        # metrics calculation
        if test_clean_dir is not None:
            clean_path = os.path.join(test_clean_dir, "clean.png")
            if os.path.exists(clean_path):
                clean_img = Image.open(clean_path).convert('L')
                clean_img = clean_img.resize(full_size, Image.BICUBIC)
                clean_np = np.array(clean_img).astype(np.float32) / 255.0

                if clean_np.shape != denoised_np.shape:
                    print(f"  Shape mismatch: clean {clean_np.shape} vs denoised {denoised_np.shape}")
                    # adjust the shape of denoised image same with that of clean image
                    denoised_img_for_metric = Image.fromarray((denoised_np * 255).astype(np.uint8))
                    denoised_img_for_metric = denoised_img_for_metric.resize(
                        (clean_np.shape[1], clean_np.shape[0]),
                        Image.BICUBIC
                    )
                    denoised_np = np.array(denoised_img_for_metric).astype(np.float32) / 255.0

                psnr_val = peak_signal_noise_ratio(clean_np, denoised_np, data_range=1.0)
                ssim_val = structural_similarity(clean_np, denoised_np, data_range=1.0)

                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                print(f" {file}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")

    # statistic
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
    if psnr_list:
        print(f"Average PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
        print(f"Average SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
        print(f"The best PSNR: {max(psnr_list):.2f} dB")
        print(f"The worst PSNR: {min(psnr_list):.2f} dB")
    print(f"✓ all the denoised images have been saved: {output_dir}")
    print("=" * 70 + "\n")

    return psnr_list, ssim_list


# ========================================================================
# 4. main function
# ========================================================================

if __name__ == '__main__':
    # # ===== Configuration =====
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # # json_path = 'newest_result/unet_training_history.json'
    # # json_path = 'newest_result/fcn_training_history.json'
    # # json_path = 'newest_result/swinunet_training_history.json'

    # # output_dir = 'output_test/FCN_patch_best'
    # # output_dir = 'output_test/UNet_patch_best'
    # # output_dir = 'output_test/SwinUNet_patch_best'

    # # ===== Step 1: plot training history =====
    # # print(" Step 1: plot loss, psnr, ssim curves")
    # # best_loss_epoch, best_loss = plot_loss_curve(json_path, 0, None, True)
    # # best_psnr_epoch, best_psnr = plot_psnr_curve(json_path, 0, None, True)
    # # best_ssim_epoch, best_ssim = plot_ssim_curve(json_path, 0, None, True)

    # # ===== Step 2: Prediction=====

    import pandas as pd
 
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
    # ===== Configuration =====
    MODEL_NAME = 'swinunet'  # 'unet' / 'swinunet' / 'fcn_resnet50'
    # MODEL_PATH = 'newest_result/unet_best_model.pth'
    MODEL_PATH = 'newest_result/swinunet_best_denoising_model.pth'
    # MODEL_PATH = 'newest_result/fcn_best_model.pth'
    # PATCH_SIZE = 256
    # STRIDE = 64
    PATCH_SIZE = 224
    STRIDE = 56
    MARGIN_RATIO = 0.3
 
    # Root directory containing noise subfolders
    SAMPLE_ROOT = r"E:\test_images"
 
    # Clean image directory (contains clean.png)
    CLEAN_DIR = os.path.join(SAMPLE_ROOT, "clean")
 
    # Noise subfolders to process
    NOISE_FOLDERS = [
        "sap_only",
        "ring_only",
        "bh_only",
        "sap_ring",
        "sap_bh",
        "ring_bh",
        "sap_ring_bh",
    ]
 
    # CSV output path
    CSV_OUTPUT = os.path.join(SAMPLE_ROOT, f"{MODEL_NAME}_results.csv")
 
    # ===== Batch processing =====
    all_records = []
 
    for folder in NOISE_FOLDERS:
        noisy_dir = os.path.join(SAMPLE_ROOT, folder)
        output_dir = os.path.join(noisy_dir, f"{MODEL_NAME}_denoised")
 
        if not os.path.isdir(noisy_dir):
            print(f"[SKIP] Folder not found: {noisy_dir}")
            continue
 
        print(f"\n{'=' * 70}")
        print(f"Processing: {folder}")
        print(f"{'=' * 70}")
 
        psnr_list, ssim_list = test_model_on_images(
            model_path=MODEL_PATH,
            test_noisy_dir=noisy_dir,
            test_clean_dir=CLEAN_DIR,
            output_dir=output_dir,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            margin_ratio=MARGIN_RATIO,
            model_name=MODEL_NAME,
            normalize=True,
            show_progress=True
        )
 
        # Record per-image results
        test_files = sorted([f for f in os.listdir(noisy_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        for fname, psnr, ssim in zip(test_files, psnr_list, ssim_list):
            all_records.append({
                "noise_type": folder,
                "image": fname,
                "PSNR": round(psnr, 4),
                "SSIM": round(ssim, 4),
            })
 
    # ===== Generate CSV =====
    df = pd.DataFrame(all_records)
 
    # Add summary rows (mean ± std per noise type)
    summary_records = []
    for folder in NOISE_FOLDERS:
        sub = df[df["noise_type"] == folder]
        if len(sub) > 0:
            summary_records.append({
                "noise_type": folder,
                "image": "MEAN ± STD",
                "PSNR": f"{sub['PSNR'].mean():.2f} ± {sub['PSNR'].std():.2f}",
                "SSIM": f"{sub['SSIM'].mean():.4f} ± {sub['SSIM'].std():.4f}",
            })
 
    df_summary = pd.DataFrame(summary_records)
    df_all = pd.concat([df, df_summary], ignore_index=True)
 
    df_all.to_csv(CSV_OUTPUT, index=False)
    print(f"\n{'=' * 70}")
    print(f"CSV saved: {CSV_OUTPUT}")
    print(f"{'=' * 70}")
 
    # Print summary
    print(f"\n{MODEL_NAME} Summary:")
    print(df_summary.to_string(index=False))
    print("\nDone!")