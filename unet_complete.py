import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as sk_psnr, structural_similarity as sk_ssim
import gc
from torch.amp import autocast, GradScaler

from dataset_pre import create_data_loaders


# ========================================================================
# UNet model
# ========================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_DC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_DC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        features = [64, 128, 256, 512, 1024]
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down_DC(features[0], features[1])
        self.down2 = Down_DC(features[1], features[2])
        self.down3 = Down_DC(features[2], features[3])
        self.down4 = Down_DC(features[3], features[4])
        self.up4 = Up_DC(features[4], features[3])
        self.up3 = Up_DC(features[3], features[2])
        self.up2 = Up_DC(features[2], features[1])
        self.up1 = Up_DC(features[1], features[0])
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits


# ========================================================================
# Functions
# ========================================================================

def get_denoised_metrics(pred_tensor, gt_tensor, normalize: bool):
    """Compute PSNR and SSIM"""
    if normalize:
        pred_tensor = (pred_tensor * 0.5) + 0.5
        gt_tensor = (gt_tensor * 0.5) + 0.5

    pred_tensor = torch.clamp(pred_tensor, 0, 1)
    gt_tensor = torch.clamp(gt_tensor, 0, 1)

    pred_np = pred_tensor.detach().cpu().numpy().squeeze()
    gt_np = gt_tensor.detach().cpu().numpy().squeeze()

    batch_size = pred_tensor.shape[0]
    total_psnr, total_ssim = 0, 0

    if batch_size == 1:
        mse = np.mean((gt_np - pred_np) ** 2)
        if mse < 1e-10:
            total_psnr = 100.0
        else:
            total_psnr = sk_psnr(gt_np, pred_np, data_range=1.0)
        total_ssim = sk_ssim(gt_np, pred_np, data_range=1.0)
    else:
        for i in range(batch_size):
            mse = np.mean((gt_np[i] - pred_np[i]) ** 2)
            if mse < 1e-10:
                psnr = 100.0
            else:
                psnr = sk_psnr(gt_np[i], pred_np[i], data_range=1.0)
            ssim = sk_ssim(gt_np[i], pred_np[i], data_range=1.0)

            total_psnr += psnr
            total_ssim += ssim

    return total_psnr / batch_size, total_ssim / batch_size


def train(model, train_loader, val_loader, optimizer, criterion, scheduler,
          num_epochs, normalize, start_epoch, device,
          train_losses=None, val_losses=None, val_psnrs=None, val_ssims=None,
          best_val_loss=None, grad_clip=1.0):

    train_losses = train_losses or []
    val_losses = val_losses or []
    val_psnrs = val_psnrs or []
    val_ssims = val_ssims or []
    best_val_loss = best_val_loss if best_val_loss is not None else float('inf')

    scaler = GradScaler('cuda')

    for epoch in range(start_epoch, num_epochs):
        # ========== Training Phase ==========
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", ascii=True)
        for batch_idx, (noisy_images, clean_images) in enumerate(pbar):
            noisy_images = noisy_images.to(device, non_blocking=True)
            clean_images = clean_images.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(noisy_images)
                loss = criterion(outputs, clean_images)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            train_loss += batch_loss * noisy_images.size(0)

            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

            del noisy_images, clean_images, outputs, loss

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # ========== Validation Phase ==========
        model.eval()
        val_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", ascii=True)
            for noisy_images, clean_images in pbar:
                noisy_images = noisy_images.to(device, non_blocking=True)
                clean_images = clean_images.to(device, non_blocking=True)

                with autocast('cuda'):
                    outputs = model(noisy_images)
                    loss = criterion(outputs, clean_images)

                psnr, ssim = get_denoised_metrics(outputs, clean_images, normalize)

                batch_size = noisy_images.size(0)
                val_loss += loss.item() * batch_size
                total_psnr += psnr * batch_size
                total_ssim += ssim * batch_size

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr:.2f}',
                    'ssim': f'{ssim:.4f}'
                })

                del noisy_images, clean_images, outputs, loss

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_psnr = total_psnr / len(val_loader.dataset)
        avg_ssim = total_ssim / len(val_loader.dataset)

        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)

        # ========== Logging ==========
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  Val PSNR:   {avg_psnr:.2f} dB")
        print(f"  Val SSIM:   {avg_ssim:.4f}")
        print(f"  Best Val Loss: {best_val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'=' * 70}\n")

        # ========== Save Best Model ==========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_psnr,
                'val_ssim': avg_ssim
            }, "unet_best_model.pth")
            print("✓ Saved best model!")

        save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            train_losses, val_losses, val_psnrs, val_ssims, best_val_loss
        )

        save_training_history_json(
            train_losses, val_losses, val_psnrs, val_ssims,
            best_val_loss, epoch + 1
        )

        scheduler.step(avg_val_loss)

        torch.cuda.empty_cache()
        gc.collect()

    return train_losses, val_losses, val_psnrs, val_ssims, best_val_loss

def save_training_history_json(train_losses, val_losses, val_psnrs, val_ssims,
                               best_val_loss, current_epoch):

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

    with open('unet_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

def save_checkpoint(model, optimizer, scheduler, epoch,
                    train_losses, val_losses, val_psnrs, val_ssims, best_val_loss,
                    filename="unet_latest_checkpoint.pth"):

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

    start_epoch = 0
    train_losses, val_losses, val_psnrs, val_ssims = [], [], [], []
    best_val_loss = float("inf")

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
        best_val_loss = checkpoint.get('best_val_loss', float("inf"))

        print(f"✓ Resumed from epoch {start_epoch}")
        print(f"  Best Val Loss: {best_val_loss:.6f}")
    else:
        print(f"No checkpoint found, starting from scratch")

    return start_epoch, train_losses, val_losses, val_psnrs, val_ssims, best_val_loss



if __name__ == "__main__":
    # ===== Configuration =====
    num_epochs = 200
    batch_size = 32
    learning_rate = 1e-5
    weight_decay = 1e-5
    normalize = True
    img_size = (256, 256)

    # ===== Device Check =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'=' * 70}\n")

    # ===== Clear GPU Cache =====
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # ===== Create DataLoader =====
    print("Creating DataLoader...")
    train_loader, val_loader = create_data_loaders(
        "dataset",  # path to the full image directory
        batch_size=batch_size,  # adjust based on GPU memory
        num_workers=6,
        img_size=img_size,
        normalize=True,
        on_the_fly=True  # on-the-fly patch cropping mode
    )
    print(f"✓ Train: {len(train_loader.dataset)} samples")
    print(f"✓ Val: {len(val_loader.dataset)} samples\n")

    # ===== Model Initialization =====
    model = UNet(in_channels=1, out_channels=1).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})\n")

    # ===== Optimizer and Loss Function =====
    criterion = nn.L1Loss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )

    # ===== Load Checkpoint =====
    start_epoch, train_losses, val_losses, val_psnrs, val_ssims, best_val_loss = \
        load_checkpoint(model, optimizer, scheduler, "unet_latest_checkpoint.pth", device)

    # ===== Training =====
    print(f"Starting training from epoch {start_epoch}...\n")

    train_losses, val_losses, val_psnrs, val_ssims, best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=200,
        normalize=True,
        start_epoch=start_epoch,
        device=device,
        train_losses=train_losses,
        val_losses=val_losses,
        val_psnrs=val_psnrs,
        val_ssims=val_ssims,
        best_val_loss=best_val_loss,
        grad_clip=1.0
    )

    print("\n✓ Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final PSNR: {val_psnrs[-1]:.2f} dB")
    print(f"Final SSIM: {val_ssims[-1]:.4f}\n")