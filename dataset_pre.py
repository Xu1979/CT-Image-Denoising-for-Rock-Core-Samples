import os
import cv2
import numpy as np
import random
import shutil
import xarray as xr
from scipy.special import sph_harm_y
from tqdm import tqdm
from typing import Optional, List
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict
# ========================================================================
# Part 1: Noise Functions
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
    '''
    plot an irregular shape based on a randomly selected point from an image

    image: grayscale image
    lm_list: (l,m), l>=0, -l<m<l
    coeffs: coefficients of each (l,m) for sph_harmonic function
    base_radius:
    scale_factor:
    n_points: number of points in the shape

    return: image with irregular shape, list of points, (center_x, center_y)
    '''

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
    theta = np.pi / 2  # θ=π/2（xy plane）
    radii = np.zeros(n_points)

    for i, phi_value in enumerate(phi):
        total_Y_lm_sum = 0.0
        for (l, m), c in zip(lm_list, coeffs):
            Y_lm = sph_harm_y(l, m, theta, phi_value)
            total_Y_lm_sum += c * Y_lm
        radii[i] = base_radius + scale_factor * np.abs(np.real(total_Y_lm_sum))

    # coordinates of points
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

    # ----------Average Blur-------
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

    # circle mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    result_image = cv2.bitwise_and(image, image, mask=mask)

    return result_image


def apply_random_noise(image, available_noises=None, max_num_noises=3, use_mask: bool = True):
    if available_noises is None:
        available_noises = ['salt_pepper', 'ring', 'beam']

    noisy_image = image.copy()
    possible_noise_counts = list(range(1, min(max_num_noises, len(available_noises)) + 1))
    weights = [0.2, 0.4, 0.4]
    num_noises = random.choices(possible_noise_counts, weights=weights, k=1)[0]
    selected_noises = random.sample(available_noises, num_noises)


    for noise in selected_noises:
        if noise == 'salt_pepper':
            amount = round(random.uniform(0.01, 0.05), 3)
            salt_vs_pepper = round(random.uniform(0.3, 0.7), 2)

            if use_mask:
                salt_pepper_image = add_salt_pepper_noise(noisy_image, amount, salt_vs_pepper)
                salt_pepper_masked_image = draw_limited_by_mask(salt_pepper_image)
                salt_mask = (salt_pepper_masked_image == 255) & (image != 255)
                pepper_mask = (salt_pepper_masked_image == 0) & (image != 0)
                noisy_image[salt_mask] = 255
                noisy_image[pepper_mask] = 0
            else:
                noisy_image = add_salt_pepper_noise(noisy_image, amount, salt_vs_pepper)

        elif noise == 'ring':
            num_rings = random.randint(10, 20)
            intensity = random.randint(30, 60)
            if use_mask:
                ring_artifact_image = add_ring_artifact(noisy_image, num_rings, intensity)
                ring_artifact_masked_image = draw_limited_by_mask(ring_artifact_image)
                mask = ring_artifact_masked_image > 0
                noisy_image[mask] = ring_artifact_masked_image[mask]
            else:
                noisy_image = add_ring_artifact(noisy_image, num_rings, intensity)

        elif noise == 'beam':
            lm_list = [(0, 0), (1, 0), (1, 1), (2, -2), (3, 1), (4, 2), (5, 0), (6, -3), (7, 4)]
            tail_coeffs = np.random.uniform(0, 2, size=7)
            tail_max = np.max(tail_coeffs)
            head_coeffs = np.random.uniform(tail_max + 1, tail_max + 3, size=2)
            coeffs = list(head_coeffs) + list(tail_coeffs)

            img_height, img_width = noisy_image.shape
            base_radius = np.random.randint(img_width / 6, img_width / 3)
            scale_factor = np.random.randint(base_radius / 3, base_radius)
            n_points = 500

            if use_mask:
                beam_image, _, _, _ = sph_harmonic_shape(image, lm_list, coeffs, base_radius, scale_factor, n_points)
                beam_masked_image = draw_limited_by_mask(beam_image)
                mask = beam_masked_image > 0
                noisy_image[mask] = beam_masked_image[mask]
            else:
                noisy_image, _, _, _ = sph_harmonic_shape(image, lm_list, coeffs, base_radius, scale_factor, n_points)

    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


# ========================================================================
# Part 2: Dataset Processing
# ========================================================================

class OnTheFlyPatchDataset(Dataset):
    """
    On-the-fly patch cropping Dataset: dynamically crops patches at training time
    """

    def __init__(self, noisy_dir, clean_dir, patch_size=(256, 256),
                 normalize=True, augment=True, deterministic=False, seed=42,
                 patches_per_image=5):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.patch_size = patch_size
        self.augment = augment
        self.deterministic = deterministic
        self.seed = seed
        self.patches_per_image = patches_per_image

        self.noisy_files = sorted([f for f in os.listdir(noisy_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
        self.clean_files = sorted([f for f in os.listdir(clean_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
        assert len(self.noisy_files) == len(self.clean_files), "Mismatch between the number of Noisy and Clean samples!"

        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        self.transform = transforms.Compose(transform_list)

    def __len__(self):

        return len(self.noisy_files) * self.patches_per_image

    def __getitem__(self, idx):

        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[img_idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[img_idx])

        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)

        h, w = noisy_img.shape

        if self.deterministic:
            
            rng = np.random.RandomState(self.seed + idx)
            if h >= self.patch_size[0] and w >= self.patch_size[1]:
                top = rng.randint(0, h - self.patch_size[0] + 1)
                left = rng.randint(0, w - self.patch_size[1] + 1)
            else:
                top, left = 0, 0
        else:
            if h >= self.patch_size[0] and w >= self.patch_size[1]:
                top = random.randint(0, h - self.patch_size[0])
                left = random.randint(0, w - self.patch_size[1])
            else:
                top, left = 0, 0

        if h >= self.patch_size[0] and w >= self.patch_size[1]:
            noisy_patch = noisy_img[top:top + self.patch_size[0], left:left + self.patch_size[1]]
            clean_patch = clean_img[top:top + self.patch_size[0], left:left + self.patch_size[1]]
        else:
            noisy_patch = np.zeros(self.patch_size, dtype=noisy_img.dtype)
            clean_patch = np.zeros(self.patch_size, dtype=clean_img.dtype)
            noisy_patch[:min(h, self.patch_size[0]), :min(w, self.patch_size[1])] = \
                noisy_img[:min(h, self.patch_size[0]), :min(w, self.patch_size[1])]
            clean_patch[:min(h, self.patch_size[0]), :min(w, self.patch_size[1])] = \
                clean_img[:min(h, self.patch_size[0]), :min(w, self.patch_size[1])]

    
        if self.augment and not self.deterministic:
           # Create a local random number generator
            aug_rng = np.random.RandomState()

            # Random horizontal flip
            if aug_rng.random() > 0.5:
                noisy_patch = np.fliplr(noisy_patch).copy()
                clean_patch = np.fliplr(clean_patch).copy()
            # Random vertical flip
            if aug_rng.random() > 0.5:
                noisy_patch = np.flipud(noisy_patch).copy()
                clean_patch = np.flipud(clean_patch).copy()
            # Random 90-degree rotation
            k = aug_rng.randint(0, 4)
            if k > 0:
                noisy_patch = np.rot90(noisy_patch, k).copy()
                clean_patch = np.rot90(clean_patch, k).copy()

        # Convert to PIL Image before applying transform
        noisy_patch = Image.fromarray(noisy_patch)
        clean_patch = Image.fromarray(clean_patch)

        noisy_patch = self.transform(noisy_patch)
        clean_patch = self.transform(clean_patch)

        return noisy_patch, clean_patch


def create_data_loaders(patch_root, batch_size=16,
                        num_workers=4, img_size=(256, 256), normalize=True,
                        on_the_fly=True):
    """ Create DataLoader """

    print("Using on-the-fly mode: dynamically cropping patches at training time")

    # Training set: randomly crop 5 patches per image
    train_dataset = OnTheFlyPatchDataset(
        os.path.join(patch_root, "train/noisy"),
        os.path.join(patch_root, "train/clean"),
        patch_size=img_size,
        normalize=normalize,
        augment=True,
        deterministic=False,
        patches_per_image=5
    )

    # Validation set: crop 5 fixed patches per image
    val_dataset = OnTheFlyPatchDataset(
        os.path.join(patch_root, "val/noisy"),
        os.path.join(patch_root, "val/clean"),
        patch_size=img_size,
        normalize=normalize,
        augment=False,
        deterministic=True,
        seed=42,
        patches_per_image=5
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ========================================================================
# Part 3: Dataset Generator
# ========================================================================

class AppConfig:
    """A class for storing all paths and configurations"""

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent

        self.nc_input_dir = self.base_dir / 'input' / 'nc'
        self.tiff_with_mask_input_dir = self.base_dir / 'input' / 'tiff' / '14Feb'
        self.tiff_no_mask_input_dir = self.base_dir / 'input' / 'tiff' / 'Carbonate'

        self.output_dir = self.base_dir / 'output'
        self.output_clean_dir = self.output_dir / 'clean'
        self.output_noisy_dir = self.output_dir / 'noisy'

        self.num_noisy_versions = 20
        self.num_digits = 5


class DatasetGenerator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.num_noisy_versions = self.config.num_noisy_versions
        self.num_digits = self.config.num_digits

    def _read_nc_file(self, nc_path: Path, verbose: bool = False) -> Optional[np.ndarray]:
        
        if not nc_path.exists():
            if verbose:
                print(f"Error: File not found at {nc_path}")
            return None

        try:
            with xr.open_dataset(nc_path) as ds:
                suitable_var_name = None

                # --- 1. Automatically select the most appropriate variable ---
                sorted_vars = sorted(ds.data_vars.keys(), key=lambda k: ds[k].ndim, reverse=True)

                for var_name in sorted_vars:
                    var = ds[var_name]
                    if var.dtype.kind not in 'iufc' or var.ndim < 2:
                        if verbose:
                            print(f"Skipping unsuitable variable: {var_name} (dtype: {var.dtype}, ndim: {var.ndim})")
                        continue

                    if min(var.shape[-2:]) > 1:
                        suitable_var_name = var_name
                        if verbose:
                            print(f"Selected variable: '{var_name}' with shape {var.shape}")
                        break

                if suitable_var_name is None:
                    raise ValueError("No suitable 2D or higher dimensional data variable found.")

                # --- 2. Efficient slicing  ---
                var_data = ds[suitable_var_name]

                if var_data.ndim > 2:
                    slicing_dims = var_data.dims[:-2]
                    selection = {}
                    for dim in slicing_dims:
                        middle_index = var_data.sizes[dim] // 2
                        selection[dim] = middle_index

                    if verbose:
                        print(f"Slicing multi-dimensional data at: {selection}")

                    image_slice = var_data.isel(**selection)
                else:
                    image_slice = var_data

                data = image_slice.values
                if verbose:
                    print(f"Data loaded into memory with shape: {data.shape}")

                # --- 3. Data cleaning ---
                if not np.all(np.isfinite(data)):
                    if verbose:
                        nan_count = np.isnan(data).sum()
                        inf_count = np.isinf(data).sum()
                        print(f"Found {nan_count} NaN(s) and {inf_count} Inf(s). Replacing with 0.")
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                # --- 4. Normalize to 0–255 ---
                min_val, max_val = np.min(data), np.max(data)
                if max_val > min_val:
                    data_normalized = ((data - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                    if verbose:
                        print(
                            f"Data normalized to uint8 range [0, 255]. Original range: [{min_val:.2f}, {max_val:.2f}]")
                else:
                    data_normalized = np.zeros_like(data, dtype=np.uint8)
                    if verbose:
                        print(f"Data is constant ({min_val}). Returning a zero image.")

                return data_normalized

        except Exception as e:
            if verbose:
                print(f"An error occurred while reading {os.path.basename(nc_path)}: {e}")
            return None

    def _read_tiff_file(self, tiff_path: Path) -> Optional[np.ndarray]:
       
        try:
            image = cv2.imread(str(tiff_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"OpenCV read failed, trying alternative method: {tiff_path}")
                return None

            return image

        except Exception as e:
            print(f"Error reading TIFF file {tiff_path}: {str(e)}")
            return None

    def _normalize_image_for_save(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure image data is within the correct range for saving
        """
        if image.dtype != np.uint8:
            if image.max() > image.min():
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        return image

    def _process_images_from_folder(self, folder_path: Path, file_format: str, start_counter: int,
                                    source_id_offset: int = 0,
                                    use_mask_override: Optional[bool] = None) -> tuple:
        """
        Process images in the specified folder
        """

        if file_format == 'nc':
            file_pattern = '*.nc'
            read_function = self._read_nc_file
        elif file_format == 'tiff':
            file_pattern = '*.tif*'
            read_function = self._read_tiff_file
        else:
            print(f"Unsupported file format: {file_format}")
            return 0, start_counter, source_id_offset

        file_paths = sorted(list(folder_path.glob(file_pattern)))
        if not file_paths:
            print(f"No {file_format} files found in {folder_path}")
            return 0, start_counter, source_id_offset

        image_counter = start_counter
        source_id = source_id_offset
        successful_processed = 0

        for idx, file_path in enumerate(tqdm(file_paths, desc=f"Processing {file_format.upper()} files")):
            image_data = read_function(file_path)

            if image_data is None:
                continue

            for variant_idx in range(self.num_noisy_versions):
                try:
                    clean_image = self._normalize_image_for_save(image_data)

                    if use_mask_override is not None:
                        should_use_mask = use_mask_override
                    else:
                        should_use_mask = True
                        if file_format == 'nc' and idx < 4:
                            should_use_mask = False

                    noisy_image = apply_random_noise(clean_image, use_mask=should_use_mask)
                    noisy_image = self._normalize_image_for_save(noisy_image)

                    # Filename format: {counter:05d}_s{source_id:04d}_v{variant:02d}.png
                    filename = f"{image_counter:05d}_s{source_id:04d}_v{variant_idx:02d}.png"
                    noisy_path = self.config.output_noisy_dir / filename
                    clean_path = self.config.output_clean_dir / filename

                    cv2.imwrite(str(clean_path), clean_image)
                    cv2.imwrite(str(noisy_path), noisy_image)
                    image_counter += 1

                except Exception as e:
                    print(f"Error processing file {file_path.name} variant {variant_idx}: {e}")
                    continue

            source_id += 1
            successful_processed += 1

        return successful_processed, image_counter, source_id

    def run(self):
        """Generate training dataset"""
        self.config.output_clean_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_noisy_dir.mkdir(parents=True, exist_ok=True)

        # Clear output directory 
        print("...Clear output directory...")
        for f in self.config.output_clean_dir.glob('*.png'):
            f.unlink()
        for f in self.config.output_noisy_dir.glob('*.png'):
            f.unlink()

        image_counter = 0
        source_id = 0

        # Process NC files
        print("\n" + "=" * 70)
        print("Process NC files...")
        print("=" * 70)
        nc_count, image_counter, source_id = self._process_images_from_folder(
            self.config.nc_input_dir,
            'nc',
            start_counter=image_counter,
            source_id_offset=source_id
        )
        print(f"NC files processed successfully")
        print(f" Number of files processed: {nc_count}")
        print(f" Generate image pairs: {image_counter}")
        print(f" Accumulated source images: {source_id}")

        # Process TIFF (14Feb folder)
        print("\n" + "=" * 70)
        print("Start processing TIFF files with MASK (14Feb)")
        print("=" * 70)
        tiff_14feb_count, image_counter, source_id = self._process_images_from_folder(
            folder_path=self.config.tiff_with_mask_input_dir,
            file_format='tiff',
            start_counter=image_counter,
            source_id_offset=source_id,
            use_mask_override=True
        )
        print(f"TIFF files with MASK processing completed")
        print(f" Number of files processed: {tiff_14feb_count}")
        print(f" Generate image pairs: {image_counter - (nc_count * self.num_noisy_versions)}")
        print(f" Accumulated source images: {source_id}")

        # Process TIFF (carbonate)
        print("\n" + "=" * 70)
        print(" Start processing TIFF files without MASK (carbonate)")
        print("=" * 70)
        tiff_carbonate_count, image_counter, source_id = self._process_images_from_folder(
            folder_path=self.config.tiff_no_mask_input_dir,
            file_format='tiff',
            start_counter=image_counter,
            source_id_offset=source_id,
            use_mask_override=False
        )
        print(f"TIFF files without MASK processing completed")
        print(f" Number of files processed: {tiff_carbonate_count}")
        print(f" Generate image pairs: {image_counter - (nc_count + tiff_14feb_count) * self.num_noisy_versions}")
        print(f" Accumulated source images: {source_id}")

        # Final statistics
        print("\n" + "=" * 70)
        print("Data generation complete!")
        print("=" * 70)
        print(f"Total number of source images: {source_id}")
        print(f"Total number of image pairs: {image_counter}")
        print(f"Per source image variants: {self.num_noisy_versions}")
        print(f"Expected number of pairs: {source_id * self.num_noisy_versions}")
        if image_counter == source_id * self.num_noisy_versions:
            print("Count verification passed")
        else:
            print(f"Warning: actual pair count does not match expected (difference: {image_counter - source_id * self.num_noisy_versions})")


# ========================================================================
# Part 4: Utility Functions
# ========================================================================
def split_dataset_by_source_id(
        source_noisy_dir: str,
        source_clean_dir: str,
        dest_base_dir: str = 'dataset',
        split_ratio: float = 0.8,
        seed: int = 42
) -> None:
    """
    Split dataset by source_id in filename, ensuring all variants of the same source image are in the same subset
    """
    print("="*70)
    print("Split dataset by source image ID")
    print("="*70)

    # 1. Split dataset by source image ID
    train_noisy = os.path.join(dest_base_dir, 'train', 'noisy')
    train_clean = os.path.join(dest_base_dir, 'train', 'clean')
    val_noisy = os.path.join(dest_base_dir, 'val', 'noisy')
    val_clean = os.path.join(dest_base_dir, 'val', 'clean')

    for path in [train_noisy, train_clean, val_noisy, val_clean]:
        os.makedirs(path, exist_ok=True)
        print(f" Create directories: {path}")

    # 2. Retrieve all files
    try:
        all_files = sorted(os.listdir(source_noisy_dir))
        if not all_files:
            print(f"Error: the source directory '{source_noisy_dir}'is empty")
            return
    except FileNotFoundError:
        print(f"Error: can't find the source directory '{source_noisy_dir}'")
        return

    print(f"\ntotal files: {len(all_files)}")

    # 3. Group by source_id
    source_groups = defaultdict(list)

    for fname in all_files:
        if not fname.endswith('.png'):
            continue

        # analyze the image name ：00123_s0056_v03.png
        parts = fname.split('_')
        if len(parts) >= 3 and parts[1].startswith('s') and parts[2].startswith('v'):
            try:
                source_id = int(parts[1][1:])  # Remove the 's' prefix
                source_groups[source_id].append(fname)
            except ValueError:
                print(f"can't analyze file: {fname}")
        else:
            print(f"Error in file format: {fname}")

    num_sources = len(source_groups)
    total_files = sum(len(files) for files in source_groups.values())

    print(f"Total number of source images: {num_sources}")
    print(f"Number of valid files: {total_files}")

    # Verify the number of variants per source image
    variants_counts = [len(files) for files in source_groups.values()]
    expected_variants = variants_counts[0] if variants_counts else 0

    if all(c == expected_variants for c in variants_counts):
        print(f"All the source images have {expected_variants} variants")
    else:
        print(f"Warning: Inconsistent number of variants")
        print(f"   Minimum: {min(variants_counts)}, Maximum: {max(variants_counts)}")

    # 4. Randomly split source images
    source_ids = list(source_groups.keys())
    random.seed(seed)
    random.shuffle(source_ids)

    split_point = int(num_sources * split_ratio)
    train_source_ids = source_ids[:split_point]
    val_source_ids = source_ids[split_point:]

    print(f"\nTraining Dataset:")
    print(f"  Number of source images: {len(train_source_ids)}")
    print(f"  Number of files: {len(train_source_ids) * expected_variants}")
    print(f"Validation Dataset:")
    print(f"  Number of source images: {len(val_source_ids)}")
    print(f"  Number of files: {len(val_source_ids) * expected_variants}")

    # 5. Copy files
    def copy_files_by_source(source_ids_list, dest_noisy, dest_clean):
        count = 0
        for sid in tqdm(source_ids_list, desc="Copying files"):
            for fname in source_groups[sid]:
                shutil.copy(
                    os.path.join(source_noisy_dir, fname),
                    os.path.join(dest_noisy, fname)
                )
                shutil.copy(
                    os.path.join(source_clean_dir, fname),
                    os.path.join(dest_clean, fname)
                )
                count += 1
        return count

    print("\nCopying files...")
    train_count = copy_files_by_source(train_source_ids, train_noisy, train_clean)
    val_count = copy_files_by_source(val_source_ids, val_noisy, val_clean)

    print("\n" + "="*70)
    print("Splitting completed")
    print(f"Training dataset: {train_count} pairs ({len(train_source_ids)} source images)")
    print(f"Validation dataset: {val_count} pairs ({len(val_source_ids)} source images)")
    print("="*70)

    # 6. No data leakage in validation set
    print("\nVerify data leakage...")
    verify_no_leakage_by_filename(train_noisy, val_noisy)


def verify_no_leakage_by_filename(train_dir: str, val_dir: str) -> bool:
    """Verify no leakage via filename"""
    train_files = os.listdir(train_dir)
    val_files = os.listdir(val_dir)

    # find source_id
    def extract_source_ids(files):
        sources = set()
        for fname in files:
            if fname.endswith('.png'):
                parts = fname.split('_')
                if len(parts) >= 2 and parts[1].startswith('s'):
                    try:
                        source_id = int(parts[1][1:])
                        sources.add(source_id)
                    except ValueError:
                        pass
        return sources

    train_sources = extract_source_ids(train_files)
    val_sources = extract_source_ids(val_files)

    overlap = train_sources & val_sources

    if overlap:
        print(f"Leakage found: {len(overlap)} source images overlap between train and val")
        print(f" Affected source image IDs (first 10): {sorted(list(overlap))[:10]}")
        return False
    else:
        print(f"No data leakage!")
        print(f"   source images in training set: {len(train_sources)}")
        print(f"   source images in validation set: {len(val_sources)}")
        print(f"   Number of overlaps: 0")
        return True


# ========================================================================
# Main Execution
# ========================================================================

if __name__ == '__main__':
    # Step 1: Generate noisy/clean image pairs from all input data (NC + TIFF)
    print("Starting dataset generation...")
    app_config = AppConfig()
    dataset_generator = DatasetGenerator(config=app_config)
    dataset_generator.run()

    # Step 2: Split generated pairs into train/val by source image ID
    print("\n\nSplitting dataset into train/val...")
    split_dataset_by_source_id(
        source_noisy_dir=str(app_config.output_noisy_dir),
        source_clean_dir=str(app_config.output_clean_dir),
        dest_base_dir='dataset',
        split_ratio=0.8,
        seed=42
    )

    print("\n\nAll operations completed. Dataset is ready for training.")