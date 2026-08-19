import random
import numpy as np
import torch

from PIL import Image

from .loveda import (
    LoveDADataset,
    IGNORE_INDEX,
    remap_mask,
    _pad_to_min_size,
    _crop_at,
    _valid_ratio,
    _random_scale_pair,
    _apply_train_augmentation,
    _to_tensor_and_normalize,
)


def _random_crop_with_box(
    image,
    mask,
    size,
    min_valid_ratio=0.0,
    max_tries=10,
):
    """
    Same selection rule as LoveDA _random_crop,
    but also returns crop coordinates and padded source.
    """
    image, mask = _pad_to_min_size(image, mask, size)
    w, h = image.size

    tries = max(int(max_tries), 1)
    best = None
    best_ratio = -1.0

    for _ in range(tries):
        left = random.randint(0, w - size)
        top = random.randint(0, h - size)

        img_c, mask_c = _crop_at(
            image, mask, size, left, top
        )

        ratio = _valid_ratio(mask_c)

        item = (
            img_c,
            mask_c,
            left,
            top,
            image,
            mask,
        )

        if ratio > best_ratio:
            best = item
            best_ratio = ratio

        if ratio >= min_valid_ratio:
            return item

    return best


def _rare_crop_with_box(
    image,
    mask,
    size,
    rare_classes=(4, 5),
    rare_min_ratio=0.02,
    min_valid_ratio=0.05,
    max_tries=10,
):
    """
    Same rare-class selection rule as E037,
    while preserving the selected crop coordinates.
    """
    if mask is None:
        return _random_crop_with_box(
            image,
            mask,
            size,
            min_valid_ratio,
            max_tries,
        )

    image, mask = _pad_to_min_size(
        image,
        mask,
        size,
    )

    w, h = image.size
    arr = np.asarray(mask)

    present = [
        int(c)
        for c in rare_classes
        if np.any(arr == int(c))
    ]

    if not present:
        return _random_crop_with_box(
            image,
            mask,
            size,
            min_valid_ratio,
            max_tries,
        )

    best = None
    best_score = -1.0
    tries = max(int(max_tries), 1)

    for _ in range(tries):
        target_class = random.choice(present)

        ys, xs = np.where(arr == target_class)
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

        valid_ratio = float(
            np.mean(crop_arr != IGNORE_INDEX)
        )

        rare_ratio = float(
            np.mean(crop_arr == target_class)
        )

        score = (
            rare_ratio
            if valid_ratio >= min_valid_ratio
            else -1.0
        )

        item = (
            img_c,
            mask_c,
            left,
            top,
            image,
            mask,
        )

        if score > best_score:
            best = item
            best_score = score

        if (
            valid_ratio >= min_valid_ratio
            and rare_ratio >= rare_min_ratio
        ):
            return item

    if best is not None and best_score >= 0:
        return best

    return _random_crop_with_box(
        image,
        mask,
        size,
        min_valid_ratio,
        max_tries,
    )


def _context_around_detail(
    image,
    mask,
    detail_left,
    detail_top,
    detail_size,
    output_size,
):
    """
    Largest valid square context up to 2x the detail FOV.

    For a normal 1024 tile:
        context HR area = 1024 x 1024
        context network input = 512 x 512

    For a tile reduced by E037's 0.75 scale:
        source becomes ~768 x 768
        context uses the valid ~768 area rather than
        introducing large artificial padding.

    Returns:
        context PIL image
        context box [x1,y1,x2,y2] locating the detail
        inside the resized context input.
        context_span
    """
    w, h = image.size

    max_span = int(detail_size * 2)

    span = min(
        max_span,
        w,
        h,
    )

    span = max(
        int(detail_size),
        int(span),
    )

    cx = detail_left + detail_size / 2.0
    cy = detail_top + detail_size / 2.0

    context_left = int(
        round(cx - span / 2.0)
    )

    context_top = int(
        round(cy - span / 2.0)
    )

    context_left = max(
        0,
        min(context_left, w - span),
    )

    context_top = max(
        0,
        min(context_top, h - span),
    )

    context_img, _ = _crop_at(
        image,
        mask,
        span,
        context_left,
        context_top,
    )

    # Detail coordinates relative to context crop.
    rx1 = detail_left - context_left
    ry1 = detail_top - context_top
    rx2 = rx1 + detail_size
    ry2 = ry1 + detail_size

    scale = output_size / float(span)

    x1 = int(round(rx1 * scale))
    y1 = int(round(ry1 * scale))
    x2 = int(round(rx2 * scale))
    y2 = int(round(ry2 * scale))

    x1 = max(0, min(output_size - 1, x1))
    y1 = max(0, min(output_size - 1, y1))
    x2 = max(x1 + 1, min(output_size, x2))
    y2 = max(y1 + 1, min(output_size, y2))

    context_img = context_img.resize(
        (output_size, output_size),
        resample=Image.Resampling.BILINEAR,
    )

    box = torch.tensor(
        [x1, y1, x2, y2],
        dtype=torch.long,
    )

    return context_img, box, span


def _sync_augment_detail_context(
    detail_img,
    detail_mask,
    context_img,
    context_box,
    mode,
):
    """
    Apply exactly the same stochastic augmentation decisions to
    HR detail and LR context.

    Important:
      - detail augmentation is the original E037 augmentation call;
      - context replays the same Python random state;
      - final RNG state is restored to the state after ONE call,
        preserving E037's random stream;
      - a marker mask tracks the transformed detail ROI inside
        the context image.
    """

    if isinstance(context_box, torch.Tensor):
        x1, y1, x2, y2 = [
            int(v) for v in context_box.tolist()
        ]
    else:
        x1, y1, x2, y2 = [
            int(v) for v in context_box
        ]

    cw, ch = context_img.size

    marker = np.zeros(
        (ch, cw),
        dtype=np.uint8,
    )

    marker[
        y1:y2,
        x1:x2,
    ] = 255

    marker = Image.fromarray(marker)

    # --------------------------------------------------------
    # First call = exactly the E037 detail augmentation.
    # --------------------------------------------------------
    rng_before = random.getstate()

    detail_img, detail_mask = _apply_train_augmentation(
        detail_img,
        detail_mask,
        mode=mode,
    )

    rng_after = random.getstate()

    # --------------------------------------------------------
    # Replay exactly the same random decisions on context.
    # --------------------------------------------------------
    random.setstate(rng_before)

    context_img, marker = _apply_train_augmentation(
        context_img,
        marker,
        mode=mode,
    )

    # Net random-number consumption must equal ONE E037 call.
    random.setstate(rng_after)

    marker_arr = np.asarray(marker)

    ys, xs = np.where(
        marker_arr > 0
    )

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(
            "Context/detail alignment marker vanished "
            "during augmentation."
        )

    new_box = torch.tensor(
        [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        dtype=torch.long,
    )

    return (
        detail_img,
        detail_mask,
        context_img,
        new_box,
    )


class HRDALoveDADataset(LoveDADataset):
    """
    Supervised HRDA-lite dataset.

    Training:
        HR detail = 512x512
        LR context = up to 1024x1024 FOV -> resized to 512x512

    Validation:
        detail = original full tile
        context = same full tile at 0.5 resolution

    Detail and context always refer to the same scene geometry.
    """

    def _getitem_impl(self, idx):
        s = self.samples[idx]

        image = Image.open(
            s["image"]
        ).convert("RGB")

        mask = None

        if s["mask"] is not None:
            raw = np.array(
                Image.open(s["mask"])
            )

            if raw.ndim == 3:
                raw = raw[..., 0]

            mask = Image.fromarray(
                remap_mask(raw)
            )

        # --------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------
        if self.training:

            if self.mixed_forest_crop:
                raise ValueError(
                    "HRDA-lite must not be combined with "
                    "E038 mixed-forest crop."
                )

            # Preserve E037's full-tile scale distribution.
            if self.augmentation == "multiscale":
                image, mask = _random_scale_pair(
                    image,
                    mask,
                    self.multi_scale_values,
                )

            if not self.crop_size or self.crop_size <= 0:
                raise ValueError(
                    "HRDA-lite training requires crop_size > 0."
                )

            # Exactly one E037 targeted-crop probability draw.
            crop_draw = (
                random.random()
                if self.rare_crop
                else 1.0
            )

            use_rare = (
                self.rare_crop
                and crop_draw < self.rare_crop_prob
            )

            if use_rare:
                (
                    detail_img,
                    detail_mask,
                    left,
                    top,
                    source_img,
                    source_mask,
                ) = _rare_crop_with_box(
                    image,
                    mask,
                    self.crop_size,
                    rare_classes=self.rare_classes,
                    rare_min_ratio=self.rare_min_ratio,
                    min_valid_ratio=self.min_valid_ratio,
                    max_tries=self.crop_retries,
                )
            else:
                (
                    detail_img,
                    detail_mask,
                    left,
                    top,
                    source_img,
                    source_mask,
                ) = _random_crop_with_box(
                    image,
                    mask,
                    self.crop_size,
                    min_valid_ratio=self.min_valid_ratio,
                    max_tries=self.crop_retries,
                )

            (
                context_img,
                context_box,
                context_span,
            ) = _context_around_detail(
                source_img,
                source_mask,
                left,
                top,
                self.crop_size,
                self.crop_size,
            )

            # Preserve E037 order:
            #
            #   multi-scale
            #       -> rare/random crop
            #       -> augmentation
            #
            # Context replays exactly the same augmentation
            # without consuming an additional RNG sequence.
            (
                detail_img,
                detail_mask,
                context_img,
                context_box,
            ) = _sync_augment_detail_context(
                detail_img,
                detail_mask,
                context_img,
                context_box,
                mode=self.augmentation,
            )

            image_t = _to_tensor_and_normalize(
                detail_img
            )

            context_t = _to_tensor_and_normalize(
                context_img
            )

            result = {
                "image": image_t,
                "context_image": context_t,
                "context_box": context_box,
                "context_span": int(context_span),
                "name": s["name"],
                "domain": s["domain"],
                "orig_size": (
                    detail_img.height,
                    detail_img.width,
                ),
            }

            if detail_mask is not None:
                result["mask"] = torch.from_numpy(
                    np.asarray(
                        detail_mask,
                        dtype=np.int64,
                    ).copy()
                ).long()

            return result

        # --------------------------------------------------------
        # VAL / TEST
        # --------------------------------------------------------

        image_t = _to_tensor_and_normalize(
            image
        )

        w, h = image.size

        cw = max(1, int(round(w * 0.5)))
        ch = max(1, int(round(h * 0.5)))

        context_img = image.resize(
            (cw, ch),
            resample=Image.Resampling.BILINEAR,
        )

        context_t = _to_tensor_and_normalize(
            context_img
        )

        result = {
            "image": image_t,
            "context_image": context_t,
            "context_box": torch.tensor(
                [0, 0, cw, ch],
                dtype=torch.long,
            ),
            "context_span": int(min(w, h)),
            "name": s["name"],
            "domain": s["domain"],
            "orig_size": (
                image.height,
                image.width,
            ),
        }

        if mask is not None:
            result["mask"] = torch.from_numpy(
                np.asarray(
                    mask,
                    dtype=np.int64,
                ).copy()
            ).long()

        return result
