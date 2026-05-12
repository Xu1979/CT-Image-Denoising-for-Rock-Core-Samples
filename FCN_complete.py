import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as sk_psnr, structural_similarity as sk_ssim
import torch.nn.functional as F
import gc
from torch.amp import autocast, GradScaler

from torchvision.models.segmentation import (
    fcn_resnet50, fcn_resnet101,
    deeplabv3_resnet50, deeplabv3_resnet101
)
from torchvision.models.segmentation import (
    FCN_ResNet50_Weights, FCN_ResNet101_Weights,
    DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
)
from dataset_pre import create_data_loaders


# ========================================================================
# Model Definition
# ========================================================================

def get_pretrained_model(pretrained_model_name: str):
    """
    Load a pretrained segmentation model and adapt it for image denoising.

    Args:
        pretrained_model_name: 'fcn_resnet50', 'fcn_resnet101',
                              'deeplabv3_resnet50', 'deeplabv3_resnet101'

    Returns:
        Modified denoising model
    """

    # Load pretrained model
    if pretrained_model_name == 'fcn_resnet50':
        base_model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    elif pretrained_model_name == 'fcn_resnet101':
        base_model = fcn_resnet101(weights=FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
    elif pretrained_model_name == 'deeplabv3_resnet50':
        base_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    elif pretrained_model_name == 'deeplabv3_resnet101':
        base_model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
    else:
        raise ValueError(f"Unsupported model name: {pretrained_model_name}")

    # Modify input layer: RGB(3) → grayscale(1)
    with torch.no_grad():
        original_conv = base_model.backbone.conv1
        rgb_weights = original_conv.weight.data
        gray_weights = rgb_weights.mean(dim=1, keepdim=True)  # average RGB channel weights

        new_conv = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        new_conv.weight = nn.Parameter(gray_weights.clone())
        base_model.backbone.conv1 = new_conv

    # Modify output layer: segmentation (21 classes) → denoising (1 channel)
    if pretrained_model_name.startswith('fcn'):
        in_ch = base_model.classifier[0].in_channels
    elif pretrained_model_name.startswith('deeplab'):
        in_ch = None
        for m in base_model.classifier.modules():
            if isinstance(m, nn.Conv2d):
                in_ch = m.in_channels
                break
        if in_ch is None:
            raise RuntimeError("Cannot determine classifier input channels")

    base_model.classifier = nn.Sequential(
        nn.Conv2d(in_ch, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.1),
        nn.Conv2d(256, 128, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 1, 1),
        nn.Tanh()  # output range [-1, 1]
    )

    # Remove auxiliary classifier
    if hasattr(base_model, 'aux_classifier'):
        base_model.aux_classifier = None

    return base_model


# ========================================================================
# Metrics
# ========================================================================

def get_denoised_metrics(pred_tensor, gt_tensor, normalize: bool):
    """Compute PSNR and SSIM"""
    if normalize:
        pred_tensor = (pred_tensor * 0.5) + 0.5
        gt_tensor = (gt_tensor * 0.5) + 0.5

    pred_tensor = torch.clamp(pred_tensor, 0, 1)
    gt_tensor = torch.clamp(gt_tensor, 0, 1)

    # Convert to CPU at once to avoid per-sample transfers
    pred_np = pred_tensor.detach().cpu().numpy().squeeze()  # [B,1,H,W] → [B,H,W]
    gt_np = gt_tensor.detach().cpu().numpy().squeeze()

    batch_size = pred_tensor.shape[0]
    total_psnr, total_ssim = 0, 0

    # When batch_size=1, squeeze removes an extra dim — handle separately
    if batch_size == 1:

        mse = np.mean((gt_np - pred_np) ** 2)
        if mse < 1e-10:  # if MSE is near zero
            total_psnr = 100.0  # return a large finite value instead of inf
        else:
            total_psnr = sk_psnr(gt_np, pred_np, data_range=1.0)
        total_ssim = sk_ssim(gt_np, pred_np, data_range=1.0)
    else:
        for i in range(batch_size):
            # add epsilon to prevent division by zero
            mse = np.mean((gt_np[i] - pred_np[i]) ** 2)
            if mse < 1e-10:
                psnr = 100.0
            else:
                psnr = sk_psnr(gt_np[i], pred_np[i], data_range=1.0)
            ssim = sk_ssim(gt_np[i], pred_np[i], data_range=1.0)

            total_psnr += psnr
            total_ssim += ssim

    return total_psnr / batch_size, total_ssim / batch_size


# ========================================================================
# Training Function
# ========================================================================

def train(model, train_loader, val_loader, optimizer, criterion, scheduler,
          num_epochs, normalize, start_epoch, device,
          train_losses=None, val_losses=None, val_psnrs=None, val_ssims=None,
          best_val_loss=None, grad_clip=1.0):
    """Training function"""

    train_losses = train_losses or []
    val_losses = val_losses or []
    val_psnrs = val_psnrs or []
    val_ssims = val_ssims or []
    best_val_loss = best_val_loss if best_val_loss is not None else float('inf')

    scaler = GradScaler('cuda')

    for epoch in range(start_epoch, num_epochs):
        # ========== Training ==========
        model.train()
        train_loss = 0.0

        train_loop = tqdm(train_loader,
                          desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                          ascii=True, leave=True)

        for batch_idx, (noisy_images, clean_images) in enumerate(train_loop):

            noisy_images = noisy_images.to(device, non_blocking=True)
            clean_images = clean_images.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast('cuda'):
                output = model(noisy_images)['out']
                loss = criterion(output, clean_images)

            scaler.scale(loss).backward()

            # gradient clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            train_loss += batch_loss * noisy_images.size(0)
            train_loop.set_postfix({'loss': f'{batch_loss:.4f}'})

            # free GPU memory
            del noisy_images, clean_images, output, loss
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # ========== Validation ==========
        model.eval()
        val_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0

        val_loop = tqdm(val_loader,
                        desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
                        ascii=True, leave=True)

        with torch.no_grad():
            for noisy_images, clean_images in val_loop:
                noisy_images = noisy_images.to(device, non_blocking=True)
                clean_images = clean_images.to(device, non_blocking=True)

                with autocast('cuda'):
                    output = model(noisy_images)['out']
                    loss = criterion(output, clean_images)

                psnr, ssim = get_denoised_metrics(output, clean_images, normalize)

                batch_size = noisy_images.size(0)
                val_loss += loss.item() * batch_size
                total_psnr += psnr * batch_size
                total_ssim += ssim * batch_size

                val_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr:.2f}',
                    'ssim': f'{ssim:.4f}'
                })

                del noisy_images, clean_images, output, loss

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_psnr = total_psnr / len(val_loader.dataset)
        avg_ssim = total_ssim / len(val_loader.dataset)

        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)

        # ==========Log==========
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  Val PSNR:   {avg_psnr:.2f} dB")
        print(f"  Val SSIM:   {avg_ssim:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'=' * 70}\n")

        # ========== Save the best model ==========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_psnr,
                'val_ssim': avg_ssim
            }, 'fcn_best_model.pth')
            print("Best model saved!\n")

        # ========== Save checkpoint (after updating best_val_loss) ==========
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            train_losses, val_losses, val_psnrs, val_ssims, best_val_loss,
            filename="fcn_latest_checkpoint.pth"
        )
        save_training_history_json(
            train_losses, val_losses, val_psnrs, val_ssims,
            best_val_loss, epoch + 1
        )
        # ========== adjust the learning rate ==========
        scheduler.step(avg_val_loss)

        # clean memory
        torch.cuda.empty_cache()
        gc.collect()

    return train_losses, val_losses, val_psnrs, val_ssims, best_val_loss


def save_training_history_json(train_losses, val_losses, val_psnrs, val_ssims,
                               best_val_loss, current_epoch):
    """
    save training history into json file

    """
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'val_ssims': val_ssims,
        'best_val_loss': best_val_loss,
        'num_epochs': len(train_losses),
        'best_epoch': val_losses.index(min(val_losses)) + 1 if val_losses else 0,
        'current_epoch': current_epoch,
        'last_update': datetime.now().isoformat()
    }

    with open('fcn_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)


def save_checkpoint(model, optimizer, scheduler, epoch,
                    train_losses, val_losses, val_psnrs, val_ssims, best_val_loss,
                    filename="checkpoint.pth"):
    """save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'val_ssims': val_ssims,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, scheduler, filename, device):
    """load checkpoint"""
    start_epoch = 0
    train_losses, val_losses, val_psnrs, val_ssims = [], [], [], []
    best_val_loss = float('inf')

    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}...")
        checkpoint = torch.load(filename, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_psnrs = checkpoint.get('val_psnrs', [])
        val_ssims = checkpoint.get('val_ssims', [])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"✓ Resumed from epoch {start_epoch}")
        print(f"  Best Val Loss: {best_val_loss:.6f}\n")
    else:
        print(f"No checkpoint found, starting from scratch\n")

    return start_epoch, train_losses, val_losses, val_psnrs, val_ssims, best_val_loss


# ========================================================================
# Main
# ========================================================================

if __name__ == '__main__':

    # ===== training configuration=====
    num_epochs = 200
    batch_size = 128
    learning_rate = 5e-5
    weight_decay = 1e-5
    normalize = True
    img_size = (256, 256)
    model_name = 'fcn_resnet50'

    # ===== device =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'=' * 70}\n")

    # clean memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Loading {model_name}...")
    model = get_pretrained_model(model_name).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}\n")

    # ===== DataLoader =====
    print("Creating DataLoader...")
    train_loader, val_loader = create_data_loaders(
        "dataset",
        batch_size=batch_size,
        num_workers=6,
        img_size=img_size,
        normalize=normalize,
        on_the_fly=True
    )
    print(f"✓ Train: {len(train_loader.dataset)} samples")
    print(f"✓ Val: {len(val_loader.dataset)} samples\n")

    # ===== loss function =====
    criterion = nn.L1Loss()

    # ===== optimizer =====
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # ===== loading checkpoint =====
    start_epoch, train_losses, val_losses, val_psnrs, val_ssims, best_val_loss = \
        load_checkpoint(model, optimizer, scheduler,
                        filename="fcn_latest_checkpoint.pth", device=device)

    # ===== training =====
    print(f"Starting training from epoch {start_epoch}...\n")

    train_losses, val_losses, val_psnrs, val_ssims, best_val_loss = train(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        num_epochs, normalize, start_epoch, device,
        train_losses, val_losses, val_psnrs, val_ssims, best_val_loss,
        grad_clip=1.0
    )

    # ===== save history =====
    save_training_history_json(
        train_losses, val_losses, val_psnrs, val_ssims,
        best_val_loss, len(train_losses)
    )

    print("\n✓ Training completed!")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Final PSNR: {val_psnrs[-1]:.2f} dB")
    print(f"Final SSIM: {val_ssims[-1]:.4f}\n")