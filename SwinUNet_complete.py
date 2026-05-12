import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import torch.nn.functional as F
import timm
from dataset_pre import create_data_loaders
import gc
from torch.amp import autocast, GradScaler


# =====================
# Swin-UNet Model
# =====================
class SwinUNet(nn.Module):
    def __init__(self, pretrained=True, img_size=224, num_classes=1, 
                 pretrained_path=r"E:\pretrained\swin_base_patch4_window7_224.pth"):
        super(SwinUNet, self).__init__()

        self.encoder = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            features_only=True,
            in_chans=3
        )
        if pretrained:
            if pretrained_path is None:
                raise ValueError("pretrained_path must be provided when pretrained=True")

            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

            sd = torch.load(pretrained_path, map_location="cpu")

            target = self.encoder.model if hasattr(self.encoder, "model") else self.encoder
            missing, unexpected = target.load_state_dict(sd, strict=False)

            print(f"[Swin] loaded from: {pretrained_path}")
            print(f"[Swin] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            
        encoder_channels = self.encoder.feature_info.channels()

        self.decoder4 = self._decoder_block(encoder_channels[3] + encoder_channels[2], encoder_channels[2])
        self.decoder3 = self._decoder_block(encoder_channels[2] + encoder_channels[1], encoder_channels[1])
        self.decoder2 = self._decoder_block(encoder_channels[1] + encoder_channels[0], encoder_channels[0])
        self.decoder1 = self._decoder_block(encoder_channels[0], 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.tanh = nn.Tanh()

    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        # ===== convert gray [B,1,H,W] to RGB [B,3,H,W] =====
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feats = self.encoder(x)
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]

        # Decoder
        d4_upsampled = F.interpolate(feats[3], size=feats[2].shape[2:], mode="bilinear", align_corners=False)
        d4 = self.decoder4(torch.cat([d4_upsampled, feats[2]], dim=1))

        d3_upsampled = F.interpolate(d4, size=feats[1].shape[2:], mode="bilinear", align_corners=False)
        d3 = self.decoder3(torch.cat([d3_upsampled, feats[1]], dim=1))

        d2_upsampled = F.interpolate(d3, size=feats[0].shape[2:], mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat([d2_upsampled, feats[0]], dim=1))

        d1 = F.interpolate(d2, size=x.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.decoder1(d1)
        out = self.final_conv(d1)
        out = self.tanh(out)
        return out


# =====================
# Metrics Calculation
# =====================
def get_denoised_metrics(pred_tensor, gt_tensor, normalize: bool):

    if pred_tensor.shape[1] == 3:
        pred_tensor = pred_tensor.mean(dim=1, keepdim=True)

    if gt_tensor.shape[1] == 3:
        gt_tensor = gt_tensor.mean(dim=1, keepdim=True)

    if pred_tensor.shape[-2:] != gt_tensor.shape[-2:]:
        pred_tensor = F.interpolate(pred_tensor, size=gt_tensor.shape[-2:],
                                    mode='bilinear', align_corners=False)

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


# =====================
# Training Function
# =====================
def train(model, train_loader, val_loader, optimizer, criterion, scheduler,
          num_epochs, normalize, start_epoch, device,
          train_losses=None, val_losses=None, val_psnrs=None, val_ssims=None,
          best_val_loss=None, grad_clip=1.0):

    print("Starting training...")

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
                output = model(noisy_images)
                loss = criterion(output, clean_images)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * noisy_images.size(0)
            train_loop.set_postfix({'loss': f'{loss.item():.4f}'})

            del noisy_images, clean_images, output, loss
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # ========== Validation ==========
        model.eval()
        val_loss, total_val_psnr, total_val_ssim = 0.0, 0.0, 0.0

        val_loop = tqdm(val_loader,
                        desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
                        ascii=True, leave=True)

        with torch.no_grad():
            for noisy_images, clean_images in val_loop:
                noisy_images = noisy_images.to(device, non_blocking=True)
                clean_images = clean_images.to(device, non_blocking=True)

                with autocast('cuda'):
                    outputs = model(noisy_images)
                    if outputs.shape != clean_images.shape:
                        outputs = F.interpolate(outputs, size=clean_images.shape[-2:],
                                                mode='bilinear', align_corners=False)
                    loss = criterion(outputs, clean_images)

                current_psnr, current_ssim = get_denoised_metrics(
                    pred_tensor=outputs,
                    gt_tensor=clean_images,
                    normalize=normalize
                )

                batch_size = noisy_images.size(0)
                val_loss += loss.item() * batch_size
                total_val_psnr += current_psnr * batch_size
                total_val_ssim += current_ssim * batch_size

                val_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{current_psnr:.2f}',
                    'ssim': f'{current_ssim:.4f}'
                })

                del noisy_images, clean_images, outputs, loss

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_psnr = total_val_psnr / len(val_loader.dataset)
        avg_val_ssim = total_val_ssim / len(val_loader.dataset)

        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        val_ssims.append(avg_val_ssim)

        # ========== Log ==========
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  Val PSNR:   {avg_val_psnr:.2f} dB")
        print(f"  Val SSIM:   {avg_val_ssim:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'=' * 70}\n")

        # ==========save best model==========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim
            }, 'swinunet_best_denoising_model.pth')
            print("✓ Best model saved!\n")

        # ========== save checkpoint ==========
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            train_losses, val_losses, val_psnrs, val_ssims, best_val_loss,
            filename="swinunet_latest_checkpoint.pth"
        )

        save_training_history_json(
            train_losses, val_losses, val_psnrs, val_ssims,
            best_val_loss, epoch + 1
        )

        # ========== adjust learning rate ==========
        scheduler.step(avg_val_loss)

        # clean memory
        torch.cuda.empty_cache()
        gc.collect()

    print("Training complete!")
    return train_losses, val_losses, val_psnrs, val_ssims, best_val_loss


# =====================
# Helper Functions
# =====================
def save_training_history_json(train_losses, val_losses, val_psnrs, val_ssims,
                               best_val_loss, current_epoch):
    """save training history"""
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
    with open('swinunet_training_history.json', 'w') as f:
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
        print(f"Loading checkpoint '{filename}'...")
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


# =====================
# Main
# =====================
if __name__ == '__main__':
    # ===== Configuration =====
    num_epochs = 200
    batch_size = 128
    learning_rate = 1e-5
    weight_decay = 1e-5
    normalize = True
    img_size = (224, 224)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'=' * 70}\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # ===== create DataLoader =====
    print("Creating DataLoader...")
    train_loader, val_loader = create_data_loaders(
        "dataset",
        batch_size=batch_size,
        num_workers=4,
        img_size=img_size,
        normalize=normalize
    )
    print(f"✓ Train: {len(train_loader.dataset)} samples")
    print(f"✓ Val: {len(val_loader.dataset)} samples\n")

    # ===== initialize model=====
    print("Loading SwinUNet...")
    model = SwinUNet(pretrained=True, img_size=224, num_classes=1).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}\n")

    # ===== loss and optimizer =====
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # ===== load checkpoint =====
    start_epoch, train_losses, val_losses, val_psnrs, val_ssims, best_val_loss = \
        load_checkpoint(model, optimizer, scheduler,
                        filename="swinunet_latest_checkpoint.pth", device=device)

    # ===== train =====
    print(f"Starting training from epoch {start_epoch}...\n")

    try:
        train_losses, val_losses, val_psnrs, val_ssims, best_val_loss = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            num_epochs=num_epochs,
            normalize=normalize,
            start_epoch=start_epoch,
            device=device,
            train_losses=train_losses,
            val_losses=val_losses,
            val_psnrs=val_psnrs,
            val_ssims=val_ssims,
            best_val_loss=best_val_loss,
            grad_clip=1.0
        )
    except KeyboardInterrupt:
        print("\n Training interrupted!")
    except Exception as e:
        print(f"\n Training error: {e}")
    finally:
        if len(train_losses) > 0:
            save_training_history_json(
                train_losses, val_losses, val_psnrs, val_ssims,
                best_val_loss, len(train_losses)
            )

    print("\n✓ Training completed!")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    if val_psnrs:
        print(f"Final PSNR: {val_psnrs[-1]:.2f} dB")
        print(f"Final SSIM: {val_ssims[-1]:.4f}")
    print()