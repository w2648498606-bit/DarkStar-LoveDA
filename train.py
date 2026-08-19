import argparse
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.loveda import LoveDADataset, CLASS_NAMES, IGNORE_INDEX
from models import MODEL_NAMES, MODEL_LABELS, build_model, count_parameters
from utils.metrics import SegmentationMetrics
from losses import build_loss
from utils.common import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    load_model_weights,
    append_csv,
    ensure_dir,
    save_json,
    load_ablation_base_weights,
)
from utils.repro import EpochShuffleSampler, seed_worker, configure_strict_determinism


def make_autocast(device, enabled):
    if device.type != "cuda" or not enabled:
        return lambda: torch.autocast(device_type=device.type, enabled=False)
    return lambda: torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def binary_fg_aux_loss(fg_logits, masks, ignore_index=255, eps=1e-6):
    """Foreground/background BCE + soft Dice, ignoring LoveDA no-data pixels."""
    valid = masks != ignore_index
    target = ((masks != 0) & valid).float()
    logit = fg_logits[:, 0]

    if not torch.any(valid):
        zero = logit.sum() * 0.0
        return zero, zero

    bce = F.binary_cross_entropy_with_logits(logit[valid], target[valid])
    prob = torch.sigmoid(logit)
    valid_f = valid.float()
    target_v = target * valid_f
    prob_v = prob * valid_f
    inter = (prob_v * target_v).sum(dim=(1, 2))
    denom = prob_v.sum(dim=(1, 2)) + target_v.sum(dim=(1, 2))
    dice = 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()
    return bce, dice



def semantic_background_bce_loss(
    logits,
    masks,
    ignore_index=255,
):
    """
    Auxiliary background-vs-non-background BCE computed directly
    from the existing multi-class semantic logits.

    LoveDA train ids:
        0 = background
        1..6 = foreground semantic classes
        255 = ignore

    Binary background logit:
        z_bg - logsumexp(z_non_bg)

    Target:
        1 = background
        0 = non-background
    """
    if logits.ndim != 4 or logits.size(1) < 2:
        raise ValueError(
            f"Expected logits [B,C,H,W] with C>=2, got {tuple(logits.shape)}"
        )

    valid = masks != int(ignore_index)

    if not torch.any(valid):
        return logits.sum() * 0.0

    # Calculate in FP32 for numerical stability under AMP.
    x = logits.float()

    bg_logit = (
        x[:, 0]
        - torch.logsumexp(
            x[:, 1:],
            dim=1,
        )
    )

    bg_target = (
        (masks == 0)
        & valid
    ).float()

    return F.binary_cross_entropy_with_logits(
        bg_logit[valid],
        bg_target[valid],
    )

def build_scheduler(name, optimizer, epochs, power=0.9):
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=0.0,
        )
    if name == "poly":
        total = max(int(epochs), 1)

        def lr_lambda(epoch_idx):
            progress = min(max(epoch_idx, 0), total) / total
            return max(1.0 - progress, 0.0) ** power

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unknown scheduler: {name}")


from datasets.loveda_hrda import HRDALoveDADataset


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    amp,
    train=True,
    max_batches=0,
    grad_clip=0.0,
    fg_aux_bce_weight=0.7,
    fg_aux_dice_weight=0.5,
    bg_bce_weight=0.0,
    aux_loss_weight=0.4,
):
    model.train(train)
    metric = SegmentationMetrics(7, IGNORE_INDEX)
    total_loss = 0.0
    total_batches = 0
    skipped_all_ignore = 0
    skipped_nonfinite = 0
    total_main_loss = 0.0
    total_fg_bce = 0.0
    total_fg_dice = 0.0
    total_bg_bce = 0.0
    total_aux_ce = 0.0
    autocast_ctx = make_autocast(device, amp)
    pbar = tqdm(loader, desc="train" if train else "val", leave=False)

    for step, batch in enumerate(pbar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        context_images = batch.get("context_image")
        if context_images is not None:
            context_images = context_images.to(device, non_blocking=True)

        context_boxes = batch.get("context_box")
        if context_boxes is not None:
            context_boxes = context_boxes.to(device, non_blocking=True)

        # A mean-reduced CrossEntropy over zero valid pixels is undefined.
        if not torch.any(masks != IGNORE_INDEX):
            skipped_all_ignore += 1
            pbar.set_postfix(skip="all-ignore")
            if max_batches and step >= max_batches:
                break
            continue

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast_ctx():
                if context_images is not None and getattr(model, "uses_context_view", False):
                    if getattr(model, "uses_context_box", False):
                        model_out = model(
                            images,
                            context_images,
                            context_boxes,
                        )
                    else:
                        model_out = model(images, context_images)
                else:
                    model_out = model(images)
                if isinstance(model_out, dict):
                    logits = model_out["logits"]
                    main_loss = criterion(logits, masks)
                    zero = main_loss.detach() * 0.0
                    fg_bce, fg_dice, aux_ce = zero, zero, zero
                    loss = main_loss

                    if "detail_logits" in model_out:
                        aux_ce = F.cross_entropy(
                            model_out["detail_logits"],
                            masks,
                            ignore_index=IGNORE_INDEX,
                        )

                        detail_weight = float(
                            getattr(
                                model,
                                "hrda_detail_loss_weight",
                                0.1,
                            )
                        )

                        loss = (
                            (1.0 - detail_weight) * main_loss
                            + detail_weight * aux_ce
                        )

                    if "fg_logits" in model_out:
                        fg_bce, fg_dice = binary_fg_aux_loss(model_out["fg_logits"], masks, IGNORE_INDEX)
                        loss = loss + fg_aux_bce_weight * fg_bce + fg_aux_dice_weight * fg_dice
                    if "aux_logits" in model_out:
                        aux_ce = criterion(model_out["aux_logits"], masks)
                        loss = loss + aux_loss_weight * aux_ce
                else:
                    logits = model_out
                    main_loss = criterion(logits, masks)
                    zero = main_loss.detach() * 0.0
                    fg_bce, fg_dice, aux_ce = zero, zero, zero
                    loss = main_loss

            # E039: auxiliary background-vs-non-background supervision
            # directly from the existing semantic logits.
            bg_bce = zero
            if bg_bce_weight > 0:
                bg_bce = semantic_background_bce_loss(
                    logits,
                    masks,
                    IGNORE_INDEX,
                )
                loss = (
                    loss
                    + bg_bce_weight * bg_bce
                )

            # Never backpropagate NaN/Inf. This is a safety net in addition to
            # valid-aware cropping in the dataset.
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                if train:
                    optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(skip="non-finite")
                if max_batches and step >= max_batches:
                    break
                continue

            if train:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        loss_value = float(loss.detach())
        total_loss += loss_value
        total_main_loss += float(main_loss.detach())
        total_fg_bce += float(fg_bce.detach())
        total_fg_dice += float(fg_dice.detach())
        total_bg_bce += float(bg_bce.detach())
        total_aux_ce += float(aux_ce.detach())
        total_batches += 1
        pred = logits.argmax(dim=1)
        metric.update(pred, masks)
        pbar.set_postfix(loss=f"{loss_value:.4f}")

        if max_batches and step >= max_batches:
            break

    out = metric.compute()
    out["loss"] = total_loss / max(total_batches, 1)
    out["main_loss"] = total_main_loss / max(total_batches, 1)
    out["fg_bce"] = total_fg_bce / max(total_batches, 1)
    out["fg_dice"] = total_fg_dice / max(total_batches, 1)
    out["bg_bce"] = total_bg_bce / max(total_batches, 1)
    out["aux_ce"] = total_aux_ce / max(total_batches, 1)
    out["used_batches"] = total_batches
    out["skipped_all_ignore"] = skipped_all_ignore
    out["skipped_nonfinite"] = skipped_nonfinite
    return out


def main():
    p = argparse.ArgumentParser(description="LoveDA stable multi-model semantic-segmentation training")
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", choices=MODEL_NAMES, default="segformer_b0")
    p.add_argument("--run-name", default="", help="Experiment folder name. Default: model name.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--context-size", type=int, default=256, help="Full-tile context resize for wide-context models.")
    p.add_argument("--workers", type=int, default=0, help="Windows: start with 0; later try 2/4.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=1.0, help="LR multiplier for model.backbone.")
    p.add_argument("--new-lr-mult", type=float, default=1.0, help="LR multiplier for model.ablation_new_modules() in v7 strict ablations.")
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--scheduler", choices=["none", "cosine", "poly"], default="cosine")
    p.add_argument("--poly-power", type=float, default=0.9)
    p.add_argument("--grad-clip", type=float, default=0.0, help="0 disables gradient clipping.")
    p.add_argument("--min-valid-ratio", type=float, default=0.05,
                   help="Reject random crops with too little non-ignore area. 0 disables rejection.")
    p.add_argument("--crop-retries", type=int, default=10)
    p.add_argument("--rare-crop", action="store_true", help="Enable rare-class-aware crop sampling.")
    p.add_argument("--rare-crop-prob", type=float, default=0.4, help="Probability of rare-aware crop when --rare-crop is enabled.")
    p.add_argument("--rare-classes", default="barren,forest", help="Comma-separated class names or ids for rare-aware cropping.")
    p.add_argument("--rare-min-ratio", type=float, default=0.02, help="Preferred minimum fraction of selected rare class inside a rare crop.")

    p.add_argument(
        "--mixed-forest-crop",
        action="store_true",
        help="Use Rural mixed-forest crops inside the existing targeted-crop probability budget.",
    )
    p.add_argument(
        "--mixed-forest-prob",
        type=float,
        default=0.2,
    )
    p.add_argument(
        "--mixed-forest-min-ratio",
        type=float,
        default=0.005,
    )
    p.add_argument(
        "--mixed-forest-max-ratio",
        type=float,
        default=0.25,
    )
    p.add_argument(
        "--mixed-forest-target-ratio",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--mixed-forest-retries",
        type=int,
        default=30,
    )

    p.add_argument("--loss", choices=["ce", "ce_dice", "ce_lovasz"], default="ce")
    p.add_argument("--dice-weight", type=float, default=0.3, help="For ce_dice: loss=(1-w)*CE+w*Dice.")
    p.add_argument("--lovasz-weight", type=float, default=0.5, help="For ce_lovasz: loss=CE+w*Lovasz-Softmax.")
    p.add_argument(
        "--bg-bce-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary background-vs-non-background BCE weight "
            "computed directly from semantic logits. "
            "0 disables it."
        ),
    )
    p.add_argument("--fg-aux-bce-weight", type=float, default=0.7, help="BG/FG auxiliary BCE weight for bgaux models.")
    p.add_argument("--fg-aux-dice-weight", type=float, default=0.5, help="BG/FG auxiliary Dice weight for bgaux models.")
    p.add_argument("--aux-loss-weight", type=float, default=0.4, help="Auxiliary semantic CE weight for OCR-style structure models.")
    p.add_argument("--augmentation", choices=["basic", "strong", "multiscale"], default="basic")
    p.add_argument("--multi-scale-values", default="0.75,1.0,1.25,1.5", help="Comma-separated scale factors used by augmentation=multiscale.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-seed", type=int, default=-1, help="Separate model RNG seed; -1 uses --seed.")
    p.add_argument("--data-seed", type=int, default=-1, help="Separate sampler/augmentation seed; -1 uses --seed.")
    p.add_argument("--train-seed", type=int, default=-1, help="Training-time torch/Python RNG (e.g. dropout); -1 uses --seed.")
    p.add_argument("--strict-ablation", action="store_true", help="Enable deterministic v7 data order/augmentation and strict initialization checks.")
    p.add_argument("--freeze-base", action="store_true", help="v7.1 warm-up: freeze E009 backbone/decoder/head and train only declared new ablation modules.")
    p.add_argument("--validate-init-from", action="store_true", help="After --init-from, run epoch0 validation and keep that initialized model as the best checkpoint before fine-tuning.")
    p.add_argument("--init-miou-tol", type=float, default=5e-5, help="Allowed |epoch0 mIoU - source best mIoU| for strict ablation initialization.")
    p.add_argument("--output-dir", default="outputs/experiments")
    p.add_argument("--resume", default="", help="Exact resume: model + optimizer + scheduler + scaler.")
    p.add_argument("--init-from", default="", help="Load model weights only and start a new experiment.")
    p.add_argument("--init-encoder-from", default="", help="For legacy MiT-B2 structure models: load only the encoder from E007.")
    p.add_argument("--init-ablation-from", default="", help="v7 strict: load the COMPLETE E009 UPerNet checkpoint; only declared new-module tensors may be missing.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="0=full epoch. Nonzero is only for quick engineering checks.")
    p.add_argument("--max-val-batches", type=int, default=0,
                   help="0=full validation. Nonzero makes mIoU non-comparable across runs.")
    a = p.parse_args()

    if sum(bool(x) for x in (a.resume, a.init_from, a.init_encoder_from, a.init_ablation_from)) > 1:
        raise ValueError("Use only one of --resume, --init-from, --init-encoder-from, or --init-ablation-from.")
    if not (0.0 <= a.min_valid_ratio <= 1.0):
        raise ValueError("--min-valid-ratio must be in [0, 1].")

    if not (0.0 <= a.rare_crop_prob <= 1.0):
        raise ValueError("--rare-crop-prob must be in [0, 1].")
    if not (0.0 <= a.rare_min_ratio <= 1.0):
        raise ValueError("--rare-min-ratio must be in [0, 1].")

    if not (
        0.0
        <= a.mixed_forest_prob
        <= 1.0
    ):
        raise ValueError(
            "--mixed-forest-prob must be in [0, 1]."
        )

    if not (
        0.0
        <= a.mixed_forest_min_ratio
        <= a.mixed_forest_target_ratio
        <= a.mixed_forest_max_ratio
        <= 1.0
    ):
        raise ValueError(
            "Require 0 <= mixed forest min <= target <= max <= 1."
        )

    if a.mixed_forest_retries < 1:
        raise ValueError(
            "--mixed-forest-retries must be >= 1."
        )

    if a.mixed_forest_crop:

        if not a.rare_crop:
            raise ValueError(
                "--mixed-forest-crop currently reserves part "
                "of the existing --rare-crop budget; "
                "also pass --rare-crop."
            )

        if (
            a.mixed_forest_prob
            > a.rare_crop_prob
        ):
            raise ValueError(
                "--mixed-forest-prob must be <= "
                "--rare-crop-prob."
            )

    if not (0.0 <= a.dice_weight <= 1.0):
        raise ValueError("--dice-weight must be in [0, 1].")
    if a.lovasz_weight < 0:
        raise ValueError("--lovasz-weight must be >= 0.")
    if a.bg_bce_weight < 0:
        raise ValueError("--bg-bce-weight must be >= 0.")
    if a.aux_loss_weight < 0:
        raise ValueError("--aux-loss-weight must be >= 0.")
    if a.backbone_lr_mult <= 0:
        raise ValueError("--backbone-lr-mult must be > 0.")
    if a.new_lr_mult <= 0:
        raise ValueError("--new-lr-mult must be > 0.")
    if a.init_miou_tol < 0:
        raise ValueError("--init-miou-tol must be >= 0.")

    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    rare_classes = []
    for token in [x.strip() for x in a.rare_classes.split(",") if x.strip()]:
        if token in class_to_id:
            rare_classes.append(class_to_id[token])
        else:
            try:
                cid = int(token)
            except ValueError as exc:
                raise ValueError(f"Unknown rare class {token!r}; use names {CLASS_NAMES} or ids 0..6") from exc
            if cid < 0 or cid >= len(CLASS_NAMES):
                raise ValueError(f"Rare class id must be 0..6, got {cid}")
            rare_classes.append(cid)
    if a.rare_crop and not rare_classes:
        raise ValueError("--rare-crop requires at least one --rare-classes entry")

    try:
        multi_scale_values = tuple(float(x.strip()) for x in a.multi_scale_values.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError("--multi-scale-values must be comma-separated numbers") from exc
    if not multi_scale_values or any(x <= 0 for x in multi_scale_values):
        raise ValueError("--multi-scale-values must contain positive values")

    model_seed = a.seed if a.model_seed < 0 else a.model_seed
    data_seed = a.seed if a.data_seed < 0 else a.data_seed
    train_seed = a.seed if a.train_seed < 0 else a.train_seed
    set_seed(model_seed)
    configure_strict_determinism(a.strict_ablation)
    if a.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    if device.type == "cuda":
        if not a.strict_ablation:
            torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()

    run_name = a.run_name.strip() or a.model
    run_dir = ensure_dir(Path(a.output_dir) / run_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    log_path = run_dir / "history.csv"
    if log_path.exists() and not a.resume:
        raise FileExistsError(
            f"{log_path} already exists. Use a new --run-name, or use --resume for exact continuation."
        )
    save_json(run_dir / "config.json", vars(a))

    print("=" * 76)
    print(f"run       : {run_name}")
    print(f"model     : {MODEL_LABELS[a.model]} ({a.model})")
    print(f"device    : {device}")
    print(f"crop      : {a.crop_size}, min_valid_ratio={a.min_valid_ratio:g}, retries={a.crop_retries}")
    print(f"batch     : train={a.batch_size}, val={a.val_batch_size}")
    print(f"optimizer : {a.optimizer}, base_lr={a.lr:g}, wd={a.weight_decay:g}, backbone_mult={a.backbone_lr_mult:g}, new_mult={a.new_lr_mult:g}")
    print(f"scheduler : {a.scheduler}")
    loss_extra = f", dice_weight={a.dice_weight:g}" if a.loss == "ce_dice" else (f", lovasz_weight={a.lovasz_weight:g}" if a.loss == "ce_lovasz" else "")
    print(f"loss      : {a.loss}{loss_extra}")
    if a.bg_bce_weight > 0:
        print(
            f"BG BCE    : semantic-logit auxiliary "
            f"weight={a.bg_bce_weight:g}"
        )
    if a.model == "segformer_b2_bgaux":
        print(f"BG/FG aux : BCE*{a.fg_aux_bce_weight:g} + Dice*{a.fg_aux_dice_weight:g}")
    if a.model in {"mit_b2_ocr_fpn", "mit_b2_uper_ocr"}:
        print(f"OCR aux CE: weight={a.aux_loss_weight:g}")
    print(f"augment   : {a.augmentation}" + (f", scales={multi_scale_values}" if a.augmentation == "multiscale" else ""))
    print(f"rare crop : {a.rare_crop}, prob={a.rare_crop_prob:g}, classes={[CLASS_NAMES[i] for i in rare_classes]}, min_ratio={a.rare_min_ratio:g}")
    print(
        f"mixedforest: {a.mixed_forest_crop}, "
        f"rural_prob={a.mixed_forest_prob:g}, "
        f"range=[{a.mixed_forest_min_ratio:g},"
        f"{a.mixed_forest_max_ratio:g}], "
        f"target={a.mixed_forest_target_ratio:g}, "
        f"retries={a.mixed_forest_retries}"
    )
    print(f"AMP       : {device.type == 'cuda' and not a.no_amp}")
    print(f"pretrained: {not a.no_pretrained}")
    print(f"strict v7 : {a.strict_ablation}, model_seed={model_seed}, data_seed={data_seed}, train_seed={train_seed}")
    if a.model == "deeplabv3_resnet50" and a.batch_size == 1:
        print("note      : DeepLabV3 BatchNorm stats are frozen for small-batch training.")
    print("=" * 76)

    needs_context = a.model in {"mit_b2_widecontext_upernet", "mit_b2_upernet_v71_widecontext"}
    needs_hrda = a.model in {
        "mit_b2_hrda_lite_upernet",
        "mit_b2_hrda_daformer",
    }
    if needs_hrda:
        print(
            "HRDA-lite : shared HR detail + LR context + "
            "class-wise scale attention"
        )
        print(
            "HRDA loss : 0.9 * fused(CE+Lovasz) + "
            "0.1 * detail CE"
        )
    if needs_context:
        print(f"context   : full LoveDA tile -> {a.context_size}x{a.context_size} scene view")

    dataset_cls = (
        HRDALoveDADataset
        if needs_hrda
        else LoveDADataset
    )

    train_ds = dataset_cls(
        a.data_root,
        "Train",
        crop_size=a.crop_size,
        training=True,
        require_mask=True,
        min_valid_ratio=a.min_valid_ratio,
        crop_retries=a.crop_retries,
        rare_crop=a.rare_crop,
        rare_crop_prob=a.rare_crop_prob,
        rare_classes=rare_classes,
        rare_min_ratio=a.rare_min_ratio,

        mixed_forest_crop=a.mixed_forest_crop,
        mixed_forest_prob=a.mixed_forest_prob,
        mixed_forest_min_ratio=a.mixed_forest_min_ratio,
        mixed_forest_max_ratio=a.mixed_forest_max_ratio,
        mixed_forest_target_ratio=a.mixed_forest_target_ratio,
        mixed_forest_retries=a.mixed_forest_retries,

        augmentation=a.augmentation,
        multi_scale_values=multi_scale_values,
        return_context=needs_context,
        context_size=a.context_size,
        deterministic_augmentation=a.strict_ablation,
        data_seed=data_seed,
    )
    val_ds = dataset_cls(
        a.data_root,
        "Val",
        crop_size=0,
        training=False,
        require_mask=True,
        return_context=needs_context,
        context_size=a.context_size,
    )
    print(f"train samples={len(train_ds)}, val samples={len(val_ds)}")

    pin = device.type == "cuda"
    train_sampler = EpochShuffleSampler(train_ds, seed=data_seed) if a.strict_ablation else None
    loader_gen = torch.Generator()
    loader_gen.manual_seed(data_seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=a.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=a.workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=(a.workers > 0 and not a.strict_ablation),
        worker_init_fn=seed_worker if a.strict_ablation else None,
        generator=loader_gen if a.strict_ablation else None,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=a.val_batch_size,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=pin,
        persistent_workers=(a.workers > 0),
        worker_init_fn=seed_worker if a.strict_ablation else None,
    )

    # Model RNG is explicitly reset after constructing the data pipeline, so model
    # initialization never changes the sample order/augmentation sequence.
    set_seed(model_seed)
    model = build_model(a.model, num_classes=7, pretrained=not a.no_pretrained).to(device)
    if a.init_encoder_from:
        if not hasattr(model, "load_loveda_segformer_encoder"):
            raise ValueError(f"--init-encoder-from is not supported by model={a.model}")
        raw = torch.load(a.init_encoder_from, map_location="cpu")
        state = raw.get("model", raw) if isinstance(raw, dict) else raw
        source_model = (raw.get("args") or {}).get("model") if isinstance(raw, dict) else None
        if source_model and source_model != "segformer_b2":
            raise ValueError(f"--init-encoder-from expects an E007 segformer_b2 checkpoint, got model={source_model}")
        n_loaded = model.load_loveda_segformer_encoder(state)
        del raw, state
        print(f"encoder init: loaded {n_loaded} tensors from {a.init_encoder_from}")

    ablation_init_report = None
    if a.init_ablation_from:
        expected_source = getattr(model, "ablation_base_model", "mit_b2_upernet")
        _, ablation_init_report = load_ablation_base_weights(
            a.init_ablation_from, model, expected_source_model=expected_source, map_location="cpu"
        )
        save_json(run_dir / "ablation_init_report.json", ablation_init_report)
        print(
            f"strict base init: loaded {ablation_init_report['loaded_base_tensors']} E009 tensors; "
            f"new tensors={ablation_init_report['new_variant_tensors']}"
        )

    if a.freeze_base:
        if not hasattr(model, "ablation_new_modules"):
            raise ValueError("--freeze-base requires a strict ablation model with ablation_new_modules().")
        new_modules = tuple(model.ablation_new_modules())
        if not new_modules:
            raise ValueError("--freeze-base found no new modules to train.")
        for p_ in model.parameters():
            p_.requires_grad = False
        for module in new_modules:
            for p_ in module.parameters():
                p_.requires_grad = True
        print("v7.1 warm-up: base frozen; only new adapter parameters are trainable")

    total_p, trainable_p = count_parameters(model)
    print(f"parameters: total={total_p/1e6:.2f}M, trainable={trainable_p/1e6:.2f}M")

    criterion = build_loss(a.loss, num_classes=7, ignore_index=IGNORE_INDEX, dice_weight=a.dice_weight, lovasz_weight=a.lovasz_weight)

    param_groups = None
    if a.backbone_lr_mult != 1.0 or a.new_lr_mult != 1.0 or a.strict_ablation:
        if not hasattr(model, "backbone"):
            raise ValueError(f"Differential LR requires model.backbone; unsupported by model={a.model}")
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone_params}

        new_modules = tuple(model.ablation_new_modules()) if hasattr(model, "ablation_new_modules") else ()
        new_params = [p for m in new_modules for p in m.parameters() if p.requires_grad]
        new_ids = {id(p) for p in new_params}
        overlap = backbone_ids & new_ids
        if overlap:
            raise RuntimeError("Ablation new-module parameters overlap backbone parameters.")

        base_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in backbone_ids and id(p) not in new_ids
        ]
        param_groups = []
        if base_params:
            param_groups.append({"params": base_params, "lr": a.lr, "name": "base"})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": a.lr * a.backbone_lr_mult, "name": "backbone"})
        if new_params:
            param_groups.append({"params": new_params, "lr": a.lr * a.new_lr_mult, "name": "new"})

        seen = sum(len(g["params"]) for g in param_groups)
        expected = sum(1 for p in model.parameters() if p.requires_grad)
        if seen != expected:
            raise RuntimeError(f"Parameter-group coverage mismatch: grouped={seen}, expected={expected}")
        print("LR groups : " + ", ".join(f"{g['name']}={g['lr']:g}" for g in param_groups))

    opt_params = param_groups if param_groups is not None else model.parameters()
    if a.optimizer == "adamw":
        optimizer = torch.optim.AdamW(opt_params, lr=a.lr, weight_decay=a.weight_decay)
    else:
        optimizer = torch.optim.SGD(opt_params, lr=a.lr, momentum=0.9, weight_decay=a.weight_decay)

    scheduler = build_scheduler(a.scheduler, optimizer, a.epochs, power=a.poly_power)
    amp = (device.type == "cuda") and (not a.no_amp)
    scaler = make_scaler(amp)
    start_epoch, best_miou = 1, -1.0

    init_from_source_best = None
    if a.init_from:
        transfer_b2_to_bgaux = False
        probe = torch.load(a.init_from, map_location="cpu")
        if isinstance(probe, dict):
            probe_model = (probe.get("args") or {}).get("model")
            transfer_b2_to_bgaux = (a.model == "segformer_b2_bgaux" and probe_model == "segformer_b2")
        del probe
        ckpt = load_model_weights(a.init_from, model, map_location=device, strict=not transfer_b2_to_bgaux)
        if isinstance(ckpt, dict):
            ckpt_model = (ckpt.get("args") or {}).get("model")
            if ckpt_model and ckpt_model != a.model and not transfer_b2_to_bgaux:
                raise ValueError(f"Checkpoint model={ckpt_model}, but --model={a.model}")
        if isinstance(ckpt, dict) and ckpt.get("best_miou") is not None:
            init_from_source_best = float(ckpt.get("best_miou"))
        print(f"initialized model weights from {a.init_from}; optimizer/scheduler start fresh")

    if a.resume:
        ckpt = load_checkpoint(
            a.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        if isinstance(ckpt, dict):
            ckpt_model = (ckpt.get("args") or {}).get("model")
            if ckpt_model and ckpt_model != a.model:
                raise ValueError(f"Checkpoint model={ckpt_model}, but --model={a.model}")
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_miou = float(ckpt.get("best_miou", -1.0))
            saved_epochs = (ckpt.get("args") or {}).get("epochs")
            if scheduler is not None and saved_epochs is not None and int(saved_epochs) != a.epochs:
                raise ValueError(
                    f"Checkpoint was created with --epochs={saved_epochs}, but current --epochs={a.epochs}. "
                    "For exact scheduler resume keep the same total epochs. To start a new schedule from these "
                    "weights, use --init-from instead of --resume."
                )
            if scheduler is not None and not ckpt.get("scheduler"):
                print("warning   : checkpoint has no scheduler state; exact LR schedule cannot be restored.")
        print(f"resumed from {a.resume}, next epoch={start_epoch}, best_mIoU={best_miou:.4f}")

    validate_initialized_model = bool(a.init_ablation_from or (a.init_from and a.validate_init_from))
    if validate_initialized_model:
        label = "E009-initialized" if a.init_ablation_from else "init-from checkpoint"
        print(f"v7/v7.1 epoch0: validating {label} model before any optimizer step...")
        va0 = run_epoch(
            model, val_dl, criterion, optimizer, device, scaler, amp=False, train=False,
            max_batches=a.max_val_batches, grad_clip=0.0,
            fg_aux_bce_weight=a.fg_aux_bce_weight, fg_aux_dice_weight=a.fg_aux_dice_weight,
            bg_bce_weight=a.bg_bce_weight,
            aux_loss_weight=a.aux_loss_weight,
        )
        if a.init_ablation_from:
            source_best = ablation_init_report.get("source_best_miou", float("nan")) if ablation_init_report else float("nan")
        else:
            source_best = float(init_from_source_best) if init_from_source_best is not None else float("nan")
        init_payload = {
            "model": a.model,
            "miou": va0["miou"],
            "pixel_acc": va0["pixel_acc"],
            "loss": va0["loss"],
            "iou": {name: float(iou) for name, iou in zip(CLASS_NAMES, va0["iou"])},
            "source_best_miou": source_best,
        }
        save_json(run_dir / "init_metrics.json", init_payload)
        diff = abs(float(va0["miou"]) - float(source_best)) if torch.isfinite(torch.tensor(source_best)) else float("nan")
        print(f"v7/v7.1 epoch0 : mIoU={va0['miou']:.6f}, source_best={source_best:.6f}, |diff|={diff:.2e}")
        if a.strict_ablation and torch.isfinite(torch.tensor(source_best)) and diff > a.init_miou_tol:
            raise RuntimeError(
                f"Strict initialization parity failed: epoch0 mIoU differs from source by {diff:.6g} "
                f"> tolerance {a.init_miou_tol:g}. Do not train this run."
            )
        best_miou = float(va0["miou"])
        args_for_ckpt = vars(a).copy()
        args_for_ckpt["run_name"] = run_name
        save_checkpoint(
            ckpt_dir / "best_miou.pth", model, optimizer, 0, best_miou, args_for_ckpt,
            scheduler=scheduler, scaler=scaler,
        )
        save_checkpoint(
            ckpt_dir / "init.pth", model, optimizer, 0, best_miou, args_for_ckpt,
            scheduler=scheduler, scaler=scaler,
        )

    # Reset training RNG after all model construction/checkpoint loading/epoch0 validation.
    # This prevents different adapter initialization sizes from shifting Dropout masks.
    if a.strict_ablation and not a.resume:
        set_seed(train_seed)

    if start_epoch > a.epochs:
        print(f"Nothing to do: checkpoint is already at epoch {start_epoch-1}, target --epochs={a.epochs}.")
        return

    for epoch in range(start_epoch, a.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        tr = run_epoch(
            model, train_dl, criterion, optimizer, device, scaler, amp,
            train=True, max_batches=a.max_train_batches, grad_clip=a.grad_clip,
            fg_aux_bce_weight=a.fg_aux_bce_weight, fg_aux_dice_weight=a.fg_aux_dice_weight,
            bg_bce_weight=a.bg_bce_weight,
            aux_loss_weight=a.aux_loss_weight,
        )
        va = run_epoch(
            model, val_dl, criterion, optimizer, device, scaler, amp=False,
            train=False, max_batches=a.max_val_batches, grad_clip=0.0,
            fg_aux_bce_weight=a.fg_aux_bce_weight, fg_aux_dice_weight=a.fg_aux_dice_weight,
            bg_bce_weight=a.bg_bce_weight,
            aux_loss_weight=a.aux_loss_weight,
        )
        sec = time.time() - t0
        peak_vram_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if device.type == "cuda" else 0.0
        )

        lr_text = ", ".join(
            f"{g.get('name', i)}={g['lr']:.3e}" for i, g in enumerate(optimizer.param_groups)
        )
        print(f"\nEpoch {epoch}/{a.epochs} | {sec:.1f}s | lr[{lr_text}] | peak_vram={peak_vram_gb:.2f}GB")
        print(
            f"train: loss={tr['loss']:.4f}, mIoU={tr['miou']:.4f}, pixel_acc={tr['pixel_acc']:.4f} "
            f"| skipped(ignore={tr['skipped_all_ignore']}, nonfinite={tr['skipped_nonfinite']})"
        )
        if a.model == "segformer_b2_bgaux":
            print(f"       main={tr['main_loss']:.4f}, fg_bce={tr['fg_bce']:.4f}, fg_dice={tr['fg_dice']:.4f}")
        if a.bg_bce_weight > 0:
            print(
                f"       main={tr['main_loss']:.4f}, "
                f"bg_bce={tr['bg_bce']:.4f}, "
                f"weighted={a.bg_bce_weight * tr['bg_bce']:.4f}"
            )
        if a.model in {"mit_b2_ocr_fpn", "mit_b2_uper_ocr"}:
            print(f"       main={tr['main_loss']:.4f}, aux_ce={tr['aux_ce']:.4f}")
        print(
            f"val  : loss={va['loss']:.4f}, mIoU={va['miou']:.4f}, pixel_acc={va['pixel_acc']:.4f} "
            f"| skipped(ignore={va['skipped_all_ignore']}, nonfinite={va['skipped_nonfinite']})"
        )
        for name, iou in zip(CLASS_NAMES, va["iou"]):
            print(f"  {name:12s} IoU={iou:.4f}")

        row = {
            "run": run_name,
            "model": a.model,
            "epoch": epoch,
            "lr": epoch_lr,
            "scheduler": a.scheduler,
            "backbone_lr_mult": a.backbone_lr_mult,
            "crop_size": a.crop_size,
            "context_size": a.context_size,
            "batch_size": a.batch_size,
            "min_valid_ratio": a.min_valid_ratio,
            "rare_crop": a.rare_crop,
            "rare_crop_prob": a.rare_crop_prob,
            "rare_classes": a.rare_classes,
            "rare_min_ratio": a.rare_min_ratio,

            "mixed_forest_crop": a.mixed_forest_crop,
            "mixed_forest_prob": a.mixed_forest_prob,
            "mixed_forest_min_ratio": a.mixed_forest_min_ratio,
            "mixed_forest_max_ratio": a.mixed_forest_max_ratio,
            "mixed_forest_target_ratio": a.mixed_forest_target_ratio,
            "mixed_forest_retries": a.mixed_forest_retries,

            "loss_name": a.loss,
            "lovasz_weight": a.lovasz_weight,
            "bg_bce_weight": a.bg_bce_weight,
            "multi_scale_values": a.multi_scale_values,
            "dice_weight": a.dice_weight,
            "fg_aux_bce_weight": a.fg_aux_bce_weight,
            "fg_aux_dice_weight": a.fg_aux_dice_weight,
            "aux_loss_weight": a.aux_loss_weight,
            "init_encoder_from": a.init_encoder_from,
            "init_ablation_from": a.init_ablation_from,
            "strict_ablation": a.strict_ablation,
            "model_seed": model_seed,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "new_lr_mult": a.new_lr_mult,
            "lr_base": next((g["lr"] for g in optimizer.param_groups if g.get("name") == "base"), epoch_lr),
            "lr_backbone": next((g["lr"] for g in optimizer.param_groups if g.get("name") == "backbone"), epoch_lr),
            "lr_new": next((g["lr"] for g in optimizer.param_groups if g.get("name") == "new"), 0.0),
            "train_main_loss": tr["main_loss"],
            "train_fg_bce": tr["fg_bce"],
            "train_fg_dice": tr["fg_dice"],
            "train_bg_bce": tr["bg_bce"],
            "train_aux_ce": tr["aux_ce"],
            "augmentation": a.augmentation,
            "train_loss": tr["loss"],
            "train_miou": tr["miou"],
            "train_pixel_acc": tr["pixel_acc"],
            "train_skipped_all_ignore": tr["skipped_all_ignore"],
            "train_skipped_nonfinite": tr["skipped_nonfinite"],
            "val_loss": va["loss"],
            "val_miou": va["miou"],
            "val_pixel_acc": va["pixel_acc"],
            "val_skipped_all_ignore": va["skipped_all_ignore"],
            "val_skipped_nonfinite": va["skipped_nonfinite"],
            "seconds": sec,
            "peak_vram_gb": peak_vram_gb,
        }
        for name, iou in zip(CLASS_NAMES, va["iou"]):
            row[f"val_iou_{name}"] = iou
        append_csv(log_path, row)

        is_best = va["miou"] > best_miou
        if is_best:
            best_miou = va["miou"]

        # Step after this epoch, then checkpoint the state needed for the next epoch.
        if scheduler is not None:
            scheduler.step()

        args_for_ckpt = vars(a).copy()
        args_for_ckpt["run_name"] = run_name
        save_checkpoint(
            ckpt_dir / "last.pth",
            model,
            optimizer,
            epoch,
            best_miou,
            args_for_ckpt,
            scheduler=scheduler,
            scaler=scaler,
        )
        if is_best:
            save_checkpoint(
                ckpt_dir / "best_miou.pth",
                model,
                optimizer,
                epoch,
                best_miou,
                args_for_ckpt,
                scheduler=scheduler,
                scaler=scaler,
            )
            print(f"New best mIoU={best_miou:.4f}; saved {ckpt_dir / 'best_miou.pth'}")

    print(f"Training finished. Best val mIoU: {best_miou:.6f}")
    print("history:", log_path)


if __name__ == "__main__":
    main()
