from pathlib import Path
from contextlib import contextmanager
import random

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ['background','building','road','water','barren','forest','agriculture']
IGNORE_INDEX = 255


@contextmanager
def _temporary_python_random(seed):
    state = random.getstate()
    random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(state)


def _sample_epoch_seed(base_seed, epoch, idx):
    # Stable arithmetic mixing; independent of Python hash randomization.
    return (int(base_seed) * 1000003 + int(epoch) * 9176 + int(idx) * 611953) & 0xFFFFFFFF


def _find_image_dirs(split_dir: Path):
    dirs = sorted([p for p in split_dir.rglob('images_png') if p.is_dir()])
    if not dirs and (split_dir / 'images_png').is_dir():
        dirs = [split_dir / 'images_png']
    return dirs


def build_loveda_samples(data_root, split='Train', require_mask=True):
    data_root = Path(data_root)
    split_dir = data_root / split
    if not split_dir.exists():
        candidates = [p for p in data_root.iterdir() if p.is_dir() and (p / split).exists()] if data_root.exists() else []
        if len(candidates) == 1:
            data_root = candidates[0]
            split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f'Cannot find split directory: {split_dir}')

    image_dirs = _find_image_dirs(split_dir)
    if not image_dirs:
        raise FileNotFoundError(
            f'No images_png folder found under {split_dir}. Expected e.g. Train/Urban/images_png and Train/Rural/images_png.'
        )

    samples = []
    for image_dir in image_dirs:
        domain = image_dir.parent.name
        mask_dir = image_dir.parent / 'masks_png'
        images = sorted([p for p in image_dir.glob('*.png') if p.is_file()])
        if not images:
            images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'}])
        for image_path in images:
            mask_path = mask_dir / (image_path.stem + '.png')
            if require_mask and not mask_path.exists():
                raise FileNotFoundError(f'Missing mask for {image_path}: expected {mask_path}')
            samples.append({
                'image': image_path,
                'mask': mask_path if mask_path.exists() else None,
                'name': image_path.stem + '.png',
                'domain': domain,
            })

    if not samples:
        raise RuntimeError(f'No image files found under {split_dir}')
    return samples


def remap_mask(raw_mask: np.ndarray):
    # Official LoveDA: 0=no-data(ignore), 1..7=seven classes.
    raw_mask = raw_mask.astype(np.int64)
    unique = np.unique(raw_mask)
    if np.any((unique < 0) | (unique > 7)):
        raise ValueError(f'Unexpected LoveDA mask values: {unique.tolist()} (expected 0..7)')
    out = np.full(raw_mask.shape, IGNORE_INDEX, dtype=np.uint8)
    for v in range(1, 8):
        out[raw_mask == v] = v - 1
    return out


def train_to_raw(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    if np.any(mask > 6):
        raise ValueError('Prediction must contain only class ids 0..6 before conversion.')
    return mask + 1


def _pad_to_min_size(image, mask, size):
    w, h = image.size
    pad_w = max(size - w, 0)
    pad_h = max(size - h, 0)
    if pad_w == 0 and pad_h == 0:
        return image, mask
    image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
    if mask is not None:
        mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=IGNORE_INDEX)
    return image, mask


def _crop_at(image, mask, size, left, top):
    box = (left, top, left + size, top + size)
    return image.crop(box), mask.crop(box) if mask is not None else None


def _valid_ratio(mask):
    if mask is None:
        return 1.0
    arr = np.asarray(mask)
    return float(np.mean(arr != IGNORE_INDEX))


def _random_crop(image, mask, size, min_valid_ratio=0.0, max_tries=10):
    """Random crop with optional rejection of near-empty/no-data crops.

    This prevents a crop containing only IGNORE_INDEX=255 from reaching
    CrossEntropyLoss(mean), which can otherwise produce NaN.
    """
    image, mask = _pad_to_min_size(image, mask, size)
    w, h = image.size
    tries = max(int(max_tries), 1)
    best = None
    best_ratio = -1.0
    for _ in range(tries):
        left = random.randint(0, w - size)
        top = random.randint(0, h - size)
        img_c, mask_c = _crop_at(image, mask, size, left, top)
        ratio = _valid_ratio(mask_c)
        if ratio > best_ratio:
            best = (img_c, mask_c)
            best_ratio = ratio
        if ratio >= min_valid_ratio:
            return img_c, mask_c
    # Fall back to the best attempted crop rather than failing a worker.
    return best



def _rare_class_crop(
    image,
    mask,
    size,
    rare_classes=(4, 5),
    rare_min_ratio=0.02,
    min_valid_ratio=0.05,
    max_tries=10,
):
    """Crop around a rare-class pixel, preferring crops where that class occupies enough area.

    Class ids are training ids after LoveDA remapping: barren=4, forest=5.
    Falls back to ordinary valid-aware random cropping when no requested rare class is present.
    """
    if mask is None:
        return _random_crop(image, mask, size, min_valid_ratio=min_valid_ratio, max_tries=max_tries)

    image, mask = _pad_to_min_size(image, mask, size)
    w, h = image.size
    arr = np.asarray(mask)
    present = [int(c) for c in rare_classes if np.any(arr == int(c))]
    if not present:
        return _random_crop(image, mask, size, min_valid_ratio=min_valid_ratio, max_tries=max_tries)

    best = None
    best_score = -1.0
    tries = max(int(max_tries), 1)
    for _ in range(tries):
        target_class = random.choice(present)
        ys, xs = np.where(arr == target_class)
        pick = random.randrange(len(xs))
        x, y = int(xs[pick]), int(ys[pick])

        # Place the chosen target pixel at a random position inside the crop.
        left_min = max(0, x - size + 1)
        left_max = min(x, w - size)
        top_min = max(0, y - size + 1)
        top_max = min(y, h - size)
        left = random.randint(left_min, left_max) if left_max >= left_min else max(min(x - size // 2, w - size), 0)
        top = random.randint(top_min, top_max) if top_max >= top_min else max(min(y - size // 2, h - size), 0)

        img_c, mask_c = _crop_at(image, mask, size, left, top)
        crop_arr = np.asarray(mask_c)
        valid_ratio = float(np.mean(crop_arr != IGNORE_INDEX))
        rare_ratio = float(np.mean(crop_arr == target_class))
        score = rare_ratio if valid_ratio >= min_valid_ratio else -1.0
        if score > best_score:
            best = (img_c, mask_c)
            best_score = score
        if valid_ratio >= min_valid_ratio and rare_ratio >= rare_min_ratio:
            return img_c, mask_c

    if best is not None and best_score >= 0.0:
        return best
    return _random_crop(image, mask, size, min_valid_ratio=min_valid_ratio, max_tries=max_tries)



def _mixed_forest_crop(
    image,
    mask,
    size,
    forest_class=5,
    min_ratio=0.005,
    max_ratio=0.25,
    target_ratio=0.05,
    min_valid_ratio=0.05,
    max_tries=30,
):
    """Prefer crops containing mixed/sparse forest rather than pure forest.

    Forest ratio is calculated over valid non-ignore pixels.
    Candidate crops are anchored near forest boundaries.
    """

    if mask is None:
        return _random_crop(
            image,
            mask,
            size,
            min_valid_ratio=min_valid_ratio,
            max_tries=max_tries,
        )

    image, mask = _pad_to_min_size(image, mask, size)

    w, h = image.size
    arr = np.asarray(mask)

    forest = arr == int(forest_class)

    if not np.any(forest):
        return _random_crop(
            image,
            mask,
            size,
            min_valid_ratio=min_valid_ratio,
            max_tries=max_tries,
        )

    # Find forest boundary pixels using four-neighbour differences.
    # Boundary pixels are more useful than forest interiors for
    # generating mixed forest/background crops.
    boundary = np.zeros_like(forest, dtype=bool)

    boundary[1:, :] |= (
        forest[1:, :]
        & ~forest[:-1, :]
    )
    boundary[:-1, :] |= (
        forest[:-1, :]
        & ~forest[1:, :]
    )
    boundary[:, 1:] |= (
        forest[:, 1:]
        & ~forest[:, :-1]
    )
    boundary[:, :-1] |= (
        forest[:, :-1]
        & ~forest[:, 1:]
    )

    ys, xs = np.where(boundary)

    if len(xs) == 0:
        ys, xs = np.where(forest)

    tries = max(int(max_tries), 1)

    best = None
    best_distance = float("inf")

    for _ in range(tries):

        pick = random.randrange(len(xs))

        x = int(xs[pick])
        y = int(ys[pick])

        left_min = max(0, x - size + 1)
        left_max = min(x, w - size)

        top_min = max(0, y - size + 1)
        top_max = min(y, h - size)

        if left_max >= left_min:
            left = random.randint(
                left_min,
                left_max,
            )
        else:
            left = max(
                min(x - size // 2, w - size),
                0,
            )

        if top_max >= top_min:
            top = random.randint(
                top_min,
                top_max,
            )
        else:
            top = max(
                min(y - size // 2, h - size),
                0,
            )

        img_c, mask_c = _crop_at(
            image,
            mask,
            size,
            left,
            top,
        )

        crop_arr = np.asarray(mask_c)

        valid = crop_arr != IGNORE_INDEX
        n_valid = int(valid.sum())

        if n_valid == 0:
            continue

        valid_ratio = (
            n_valid / float(crop_arr.size)
        )

        if valid_ratio < min_valid_ratio:
            continue

        forest_ratio = float(
            np.mean(
                crop_arr[valid]
                == int(forest_class)
            )
        )

        if (
            min_ratio
            <= forest_ratio
            <= max_ratio
        ):
            distance = abs(
                forest_ratio
                - target_ratio
            )

            if distance < best_distance:
                best = (
                    img_c,
                    mask_c,
                )
                best_distance = distance

    # Important:
    # do NOT fall back to the "best high-forest crop".
    # Otherwise pure/easy forest patches would dominate again.
    if best is not None:
        return best

    return _random_crop(
        image,
        mask,
        size,
        min_valid_ratio=min_valid_ratio,
        max_tries=max_tries,
    )

def _random_scale_pair(image, mask, scales=(0.75, 1.0, 1.25, 1.5)):
    """Resize image/mask by one randomly selected scale before cropping.

    Image uses bilinear interpolation; semantic masks use nearest-neighbor interpolation.
    """
    if not scales:
        return image, mask
    scale = float(random.choice(tuple(scales)))
    if scale <= 0:
        raise ValueError("all multi-scale factors must be > 0")
    w, h = image.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image, mask
    image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    if mask is not None:
        mask = mask.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
    return image, mask


def _apply_train_augmentation(image, mask, mode="basic"):
    mode = str(mode).lower()
    if mode not in {"basic", "strong", "multiscale"}:
        raise ValueError(f"Unknown augmentation mode: {mode}")

    if random.random() < 0.5:
        image = ImageOps.mirror(image)
        if mask is not None:
            mask = ImageOps.mirror(mask)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
        if mask is not None:
            mask = ImageOps.flip(mask)

    if mode in {"strong", "multiscale"}:
        # Remote-sensing overhead imagery has no canonical up direction.
        k = random.randint(0, 3)
        if k:
            transpose = {
                1: Image.Transpose.ROTATE_90,
                2: Image.Transpose.ROTATE_180,
                3: Image.Transpose.ROTATE_270,
            }[k]
            image = image.transpose(transpose)
            if mask is not None:
                mask = mask.transpose(transpose)

        # Mild photometric jitter is kept only for the generic strong mode.
        # The literature-driven multi-scale setting changes geometry only, which
        # keeps the ablation focused on scale robustness.
        if mode == "strong" and random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.90, 1.10))

    return image, mask


def _to_tensor_and_normalize(image):
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()


def _resize_context_to_tensor(image, size=256):
    size = int(size)
    if size <= 0:
        raise ValueError("context size must be > 0")
    image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return _to_tensor_and_normalize(image)


class LoveDADataset(Dataset):
    def __init__(
        self,
        data_root,
        split='Train',
        crop_size=512,
        training=False,
        require_mask=True,
        min_valid_ratio=0.05,
        crop_retries=10,
        rare_crop=False,
        rare_crop_prob=0.4,
        rare_classes=(4, 5),
        rare_min_ratio=0.02,
        mixed_forest_crop=False,
        mixed_forest_prob=0.2,
        mixed_forest_min_ratio=0.005,
        mixed_forest_max_ratio=0.25,
        mixed_forest_target_ratio=0.05,
        mixed_forest_retries=30,
        augmentation="basic",
        multi_scale_values=(0.75, 1.0, 1.25, 1.5),
        return_context=False,
        context_size=256,
        deterministic_augmentation=False,
        data_seed=42,
    ):
        self.samples = build_loveda_samples(data_root, split, require_mask=require_mask)
        self.crop_size = crop_size
        self.training = training
        self.require_mask = require_mask
        self.min_valid_ratio = float(min_valid_ratio)
        self.crop_retries = int(crop_retries)
        self.rare_crop = bool(rare_crop)
        self.rare_crop_prob = float(rare_crop_prob)
        self.rare_classes = tuple(int(c) for c in rare_classes)
        self.rare_min_ratio = float(rare_min_ratio)

        self.mixed_forest_crop = bool(
            mixed_forest_crop
        )
        self.mixed_forest_prob = float(
            mixed_forest_prob
        )
        self.mixed_forest_min_ratio = float(
            mixed_forest_min_ratio
        )
        self.mixed_forest_max_ratio = float(
            mixed_forest_max_ratio
        )
        self.mixed_forest_target_ratio = float(
            mixed_forest_target_ratio
        )
        self.mixed_forest_retries = int(
            mixed_forest_retries
        )

        self.augmentation = str(augmentation).lower()
        self.multi_scale_values = tuple(float(x) for x in multi_scale_values)
        if not self.multi_scale_values or any(x <= 0 for x in self.multi_scale_values):
            raise ValueError("multi_scale_values must contain positive scale factors")
        self.return_context = bool(return_context)
        self.context_size = int(context_size)
        self.deterministic_augmentation = bool(deterministic_augmentation)
        self.data_seed = int(data_seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.training and self.deterministic_augmentation:
            seed = _sample_epoch_seed(self.data_seed, self.epoch, idx)
            with _temporary_python_random(seed):
                return self._getitem_impl(idx)
        return self._getitem_impl(idx)

    def _getitem_impl(self, idx):
        s = self.samples[idx]
        image = Image.open(s['image']).convert('RGB')
        context_source = image.copy() if self.return_context else None
        mask = None
        if s['mask'] is not None:
            raw = np.array(Image.open(s['mask']))
            if raw.ndim == 3:
                raw = raw[..., 0]
            mask = Image.fromarray(remap_mask(raw))

        if self.training:
            # Multi-scale training: rescale the full tile first, then sample the
            # fixed-size crop. This changes the apparent object scale seen by
            # the network while preserving a fixed tensor size for batching.
            if self.augmentation == "multiscale":
                image, mask = _random_scale_pair(image, mask, self.multi_scale_values)
            if self.crop_size and self.crop_size > 0:

                # Use ONE probability draw.
                #
                # E037:
                #   draw < 0.4 -> rare crop
                #
                # E038 Rural:
                #   draw < 0.2 -> mixed forest crop
                #   0.2-0.4    -> original rare crop
                #   >=0.4      -> random crop
                #
                # Therefore total targeted-crop probability
                # remains exactly 0.4.
                #
                # Urban remains equivalent to E037.

                targeted = (
                    self.rare_crop
                    or self.mixed_forest_crop
                )

                crop_draw = (
                    random.random()
                    if targeted
                    else 1.0
                )

                is_rural = (
                    str(s['domain']).lower()
                    == 'rural'
                )

                use_mixed_forest = (
                    self.mixed_forest_crop
                    and is_rural
                    and crop_draw
                    < self.mixed_forest_prob
                )

                use_rare = (
                    self.rare_crop
                    and crop_draw
                    < self.rare_crop_prob
                )

                if use_mixed_forest:

                    image, mask = _mixed_forest_crop(
                        image,
                        mask,
                        self.crop_size,
                        forest_class=5,
                        min_ratio=self.mixed_forest_min_ratio,
                        max_ratio=self.mixed_forest_max_ratio,
                        target_ratio=self.mixed_forest_target_ratio,
                        min_valid_ratio=self.min_valid_ratio,
                        max_tries=self.mixed_forest_retries,
                    )

                elif use_rare:

                    image, mask = _rare_class_crop(
                        image,
                        mask,
                        self.crop_size,
                        rare_classes=self.rare_classes,
                        rare_min_ratio=self.rare_min_ratio,
                        min_valid_ratio=self.min_valid_ratio,
                        max_tries=self.crop_retries,
                    )

                else:

                    image, mask = _random_crop(
                        image,
                        mask,
                        self.crop_size,
                        min_valid_ratio=self.min_valid_ratio,
                        max_tries=self.crop_retries,
                    )
            image, mask = _apply_train_augmentation(image, mask, mode=self.augmentation)

        image_t = _to_tensor_and_normalize(image)
        result = {
            'image': image_t,
            'name': s['name'],
            'domain': s['domain'],
            'orig_size': (image.height, image.width),
        }
        if context_source is not None:
            result['context_image'] = _resize_context_to_tensor(context_source, self.context_size)
        if mask is not None:
            result['mask'] = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()).long()
        return result
