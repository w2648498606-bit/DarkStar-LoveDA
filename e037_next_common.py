from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import train as base
import dg_sweep_common as dg

NUM_CLASSES = 7
IGNORE_INDEX = 255
CLASS_NAMES = [
    "background", "building", "road", "water",
    "barren", "forest", "agriculture",
]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def replace_argv_value(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    if flag not in out:
        out.extend([flag, str(value)])
        return out
    i = out.index(flag)
    if i + 1 >= len(out):
        raise ValueError(f"Flag {flag} has no value in argv")
    out[i + 1] = str(value)
    return out


def e037_argv(run_name: str, epochs: int = 12) -> list[str]:
    """Exact E037 recipe as frozen by the prior DG suite."""
    return dg.common_e037_argv(run_name, epochs=epochs)


def _backward_step(loss, model, optimizer, scaler, grad_clip: float):
    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


def _plain_train_loop(
    model, loader, criterion, optimizer, device, scaler, amp,
    *, extra_loss_fn=None, extra_name="extra", extra_weight=1.0,
    max_batches=0, grad_clip=0.0,
):
    model.train(True)
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    total = main_total = extra_total = 0.0
    n = skip_i = skip_nf = 0
    autocast_ctx = base.make_autocast(device, amp)
    pbar = tqdm(loader, desc="train", leave=False)

    for step, batch in enumerate(pbar, 1):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        if not torch.any(y != IGNORE_INDEX):
            skip_i += 1
            if max_batches and step >= max_batches:
                break
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            logits = dg.extract_logits(model(x))
            main_loss = criterion(logits, y)
            if extra_loss_fn is None:
                extra_loss = logits.sum() * 0.0
            else:
                extra_loss = extra_loss_fn(logits, y)
            loss = main_loss + float(extra_weight) * extra_loss

        if not torch.isfinite(loss):
            skip_nf += 1
            optimizer.zero_grad(set_to_none=True)
            if max_batches and step >= max_batches:
                break
            continue

        _backward_step(loss, model, optimizer, scaler, grad_clip)
        dg.update_confusion(conf, logits, y)

        total += float(loss.detach())
        main_total += float(main_loss.detach())
        extra_total += float(extra_loss.detach())
        n += 1
        pbar.set_postfix(
            loss=f"{total/max(n,1):.4f}",
            **({extra_name: f"{extra_total/max(n,1):.4f}"} if extra_loss_fn else {}),
        )

        if max_batches and step >= max_batches:
            break

    den = max(n, 1)
    if extra_loss_fn is not None:
        print(
            f"{extra_name} | raw={extra_total/den:.5f} "
            f"weighted={float(extra_weight)*extra_total/den:.5f}"
        )
    return dg.ret_dict(total, main_total, conf, n, skip_i, skip_nf)


# -----------------------------------------------------------------------------
# E086: conservative hard-pixel CE + Lovasz
# -----------------------------------------------------------------------------

class HardPixelCELovaszLoss(nn.Module):
    """Blend ordinary CE with top-k hard-pixel CE, then add Lovasz.

    This deliberately preserves most of E037's CE objective instead of replacing
    it with pure OHEM. Defaults: 0.7*CE_all + 0.3*CE_top30 + 0.5*Lovasz.
    """

    def __init__(
        self,
        num_classes=7,
        ignore_index=255,
        hard_fraction=0.30,
        hard_weight=0.30,
        lovasz_weight=0.50,
    ):
        super().__init__()
        if not (0 < hard_fraction <= 1):
            raise ValueError("hard_fraction must be in (0,1]")
        if not (0 <= hard_weight <= 1):
            raise ValueError("hard_weight must be in [0,1]")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.hard_fraction = float(hard_fraction)
        self.hard_weight = float(hard_weight)
        self.lovasz_weight = float(lovasz_weight)

    def forward(self, logits, target):
        per = F.cross_entropy(
            logits, target, ignore_index=self.ignore_index, reduction="none"
        )
        valid = target != self.ignore_index
        vals = per[valid]
        if vals.numel() == 0:
            return logits.sum() * 0.0

        ce_all = vals.mean()
        k = max(1, int(math.ceil(vals.numel() * self.hard_fraction)))
        ce_hard = torch.topk(vals, k=k, largest=True, sorted=False).values.mean()

        # Reuse the project's tested Lovasz implementation.
        import losses as project_losses
        probs = torch.softmax(logits.float(), dim=1)
        c = probs.shape[1]
        probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, c)
        labels_flat = target.contiguous().view(-1)
        keep = labels_flat != self.ignore_index
        probs_flat = probs_flat[keep]
        labels_flat = labels_flat[keep]
        lovasz = project_losses.lovasz_softmax_flat(
            probs_flat, labels_flat, classes="present"
        )
        ce_mix = (1.0 - self.hard_weight) * ce_all + self.hard_weight * ce_hard
        return ce_mix + self.lovasz_weight * lovasz


def install_hard_pixel_loss(
    *, hard_fraction=0.30, hard_weight=0.30, lovasz_weight=0.50
):
    original = base.build_loss

    def build_loss(name="ce", num_classes=7, ignore_index=255, dice_weight=0.3, lovasz_weight=0.5):
        if str(name).lower() == "ce_lovasz":
            return HardPixelCELovaszLoss(
                num_classes=num_classes,
                ignore_index=ignore_index,
                hard_fraction=hard_fraction,
                hard_weight=hard_weight,
                lovasz_weight=float(lovasz_weight),
            )
        return original(
            name,
            num_classes=num_classes,
            ignore_index=ignore_index,
            dice_weight=dice_weight,
            lovasz_weight=lovasz_weight,
        )

    base.build_loss = build_loss
    return original


# -----------------------------------------------------------------------------
# E087: boundary-aware semantic edge auxiliary loss (training only)
# -----------------------------------------------------------------------------

def _balanced_binary_prob_loss(prob, target, valid, eps=1e-6):
    if not torch.any(valid):
        return prob.sum() * 0.0
    p = prob[valid].float().clamp(eps, 1.0 - eps)
    t = target[valid].float()
    pos = t > 0.5
    neg = ~pos
    pieces = []
    if torch.any(pos):
        pieces.append((-torch.log(p[pos])).mean())
    if torch.any(neg):
        pieces.append((-torch.log1p(-p[neg])).mean())
    if not pieces:
        return prob.sum() * 0.0
    return torch.stack(pieces).mean().to(prob.dtype)


def semantic_boundary_loss(logits, target):
    """Class-transition boundary loss without an inference-time head.

    Edge probability between neighboring pixels is 1 - dot(p_i, p_j). Ground
    truth edge is whether adjacent valid labels differ. Positive and negative
    edge pairs are averaged separately to prevent non-edge domination.
    """
    probs = torch.softmax(logits.float(), dim=1)
    valid = target != IGNORE_INDEX

    # horizontal pairs
    ph = 1.0 - (probs[:, :, :, 1:] * probs[:, :, :, :-1]).sum(dim=1)
    vh = valid[:, :, 1:] & valid[:, :, :-1]
    gh = (target[:, :, 1:] != target[:, :, :-1]) & vh
    lh = _balanced_binary_prob_loss(ph, gh, vh)

    # vertical pairs
    pv = 1.0 - (probs[:, :, 1:, :] * probs[:, :, :-1, :]).sum(dim=1)
    vv = valid[:, 1:, :] & valid[:, :-1, :]
    gv = (target[:, 1:, :] != target[:, :-1, :]) & vv
    lv = _balanced_binary_prob_loss(pv, gv, vv)

    return 0.5 * (lh + lv)


def make_boundary_run_epoch(boundary_weight=0.10):
    boundary_weight = float(boundary_weight)

    def run_epoch(
        model, loader, criterion, optimizer, device, scaler, amp,
        train=True, max_batches=0, grad_clip=0.0,
        fg_aux_bce_weight=0.7, fg_aux_dice_weight=0.5,
        bg_bce_weight=0.0, aux_loss_weight=0.4,
    ):
        if not train:
            return dg.sweep_domain_validation_epoch(
                model, loader, criterion, optimizer, device, scaler, amp,
                max_batches=max_batches, bg_bce_weight=bg_bce_weight,
            )
        return _plain_train_loop(
            model, loader, criterion, optimizer, device, scaler, amp,
            extra_loss_fn=semantic_boundary_loss,
            extra_name="Boundary",
            extra_weight=boundary_weight,
            max_batches=max_batches,
            grad_clip=grad_clip,
        )

    return run_epoch


# -----------------------------------------------------------------------------
# E088: OCR-lite residual object-context refiner
# -----------------------------------------------------------------------------

class OCRLite(nn.Module):
    def __init__(self, channels=256, context_dim=128, num_classes=7):
        super().__init__()
        self.q = nn.Conv2d(channels, context_dim, 1, bias=False)
        self.k = nn.Linear(channels, context_dim, bias=False)
        self.v = nn.Linear(channels, context_dim, bias=False)
        self.out = nn.Conv2d(context_dim, channels, 1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.num_classes = int(num_classes)
        self.scale = context_dim ** -0.5

    def forward(self, feat, coarse_logits):
        b, c, h, w = feat.shape
        if coarse_logits.shape[-2:] != (h, w):
            coarse_logits = F.interpolate(
                coarse_logits, size=(h, w), mode="bilinear", align_corners=False
            )
        probs = torch.softmax(coarse_logits.float(), dim=1)  # B,K,H,W
        feat_tokens = feat.float().flatten(2).transpose(1, 2)  # B,N,C
        weights = probs.flatten(2)  # B,K,N
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        proto = torch.bmm(weights, feat_tokens) / denom  # B,K,C

        q = self.q(feat.float()).flatten(2).transpose(1, 2)  # B,N,D
        k = self.k(proto)  # B,K,D
        v = self.v(proto)  # B,K,D
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        ctx = torch.bmm(attn, v).transpose(1, 2).reshape(b, -1, h, w)
        delta = self.out(ctx).to(feat.dtype)
        return feat + delta


class OCRLiteWrapper(nn.Module):
    """Wraps the project's exact MiT-B2+UPerNet and adds zero-init OCR-lite."""

    def __init__(self, base_model, channels=256, context_dim=128, num_classes=7):
        super().__init__()
        self.base = base_model
        self.ocr_lite = OCRLite(
            channels=channels, context_dim=context_dim, num_classes=num_classes
        )

    @property
    def backbone(self):
        return self.base.backbone

    @property
    def decoder(self):
        return self.base.decoder

    @property
    def head(self):
        return self.base.head

    def ablation_new_modules(self):
        return (self.ocr_lite,)

    def load_loveda_segformer_encoder(self, state):
        return self.base.load_loveda_segformer_encoder(state)

    def _upsample(self, small, x):
        if hasattr(self.base, "_upsample_logits"):
            return self.base._upsample_logits(small, x)
        return F.interpolate(
            small, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

    def _coarse_logits_without_dropout(self, feat):
        # E037 head is Dropout2d(0.10) -> Conv2d.  Calling the full head twice
        # would consume an extra dropout RNG draw and break strict trajectory
        # comparability even though OCR residual starts at zero.  For the coarse
        # object regions, reuse only the classifier conv; final logits still call
        # the complete E037 head exactly once.
        if isinstance(self.base.head, nn.Sequential) and len(self.base.head) >= 1:
            classifier = self.base.head[-1]
            if isinstance(classifier, nn.Conv2d):
                return classifier(feat)
        raise RuntimeError(
            "OCR-lite expects E037 head=Sequential(..., Conv2d); verifier should catch this."
        )

    def forward(self, x):
        feats = self.base.backbone(x)
        feat = self.base.decoder(feats)
        coarse = self._coarse_logits_without_dropout(feat)
        refined = self.ocr_lite(feat, coarse)
        logits = self.base.head(refined)
        return self._upsample(logits, x)


def install_ocr_lite_model():
    original = base.build_model

    def build_model(name, num_classes=7, pretrained=True):
        m = original(name, num_classes=num_classes, pretrained=pretrained)
        if str(name) != "mit_b2_upernet":
            return m
        return OCRLiteWrapper(m, channels=256, context_dim=128, num_classes=num_classes)

    base.build_model = build_model
    return original


# -----------------------------------------------------------------------------
# E089: 768 crop, batch 4, gradient accumulation 2 -> effective batch 8
# -----------------------------------------------------------------------------

def make_grad_accum_run_epoch(accum_steps=2):
    accum_steps = int(accum_steps)
    if accum_steps < 1:
        raise ValueError("accum_steps must be >=1")

    def run_epoch(
        model, loader, criterion, optimizer, device, scaler, amp,
        train=True, max_batches=0, grad_clip=0.0,
        fg_aux_bce_weight=0.7, fg_aux_dice_weight=0.5,
        bg_bce_weight=0.0, aux_loss_weight=0.4,
    ):
        if not train:
            return dg.sweep_domain_validation_epoch(
                model, loader, criterion, optimizer, device, scaler, amp,
                max_batches=max_batches, bg_bce_weight=bg_bce_weight,
            )

        model.train(True)
        conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        total = 0.0
        n = skip_i = skip_nf = 0
        valid_micro = 0
        optimizer_steps = 0
        autocast_ctx = base.make_autocast(device, amp)
        pbar = tqdm(loader, desc="train", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, 1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            if not torch.any(y != IGNORE_INDEX):
                skip_i += 1
                if max_batches and step >= max_batches:
                    break
                continue

            with autocast_ctx():
                logits = dg.extract_logits(model(x))
                raw_loss = criterion(logits, y)
                loss = raw_loss / accum_steps

            if not torch.isfinite(raw_loss):
                skip_nf += 1
                optimizer.zero_grad(set_to_none=True)
                valid_micro = 0
                if max_batches and step >= max_batches:
                    break
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            valid_micro += 1
            if valid_micro >= accum_steps:
                if scaler is not None:
                    if grad_clip and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                valid_micro = 0
                optimizer_steps += 1

            dg.update_confusion(conf, logits, y)
            total += float(raw_loss.detach())
            n += 1
            pbar.set_postfix(loss=f"{total/max(n,1):.4f}", step=optimizer_steps)
            if max_batches and step >= max_batches:
                break

        # Defensive leftover step; normal E089 has an even number of microbatches.
        if valid_micro > 0:
            scale_fix = accum_steps / valid_micro
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale_fix)
            if scaler is not None:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        print(
            f"GradAccum | micro_batch=4 accum={accum_steps} "
            f"effective_batch={4*accum_steps} optimizer_steps={optimizer_steps}"
        )
        return dg.ret_dict(total, total, conf, n, skip_i, skip_nf)

    return run_epoch


# -----------------------------------------------------------------------------
# E090: BLV-lite, class-frequency-dependent training-time logit variation
# -----------------------------------------------------------------------------

class BLVState:
    scales = None
    printed = False


def estimate_class_scales_from_masks(dataset, sample_masks=128):
    """Deterministic approximate LoveDA class frequencies from raw mask PNGs."""
    samples = getattr(dataset, "samples", None)
    if not samples:
        raise RuntimeError("BLV-lite needs dataset.samples with mask paths")
    n = len(samples)
    take = min(int(sample_masks), n)
    indices = np.linspace(0, n - 1, take, dtype=int)
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    from PIL import Image

    for idx in indices:
        path = samples[int(idx)].get("mask")
        if path is None:
            continue
        raw = np.asarray(Image.open(path), dtype=np.int64)
        # Official LoveDA: 0 ignore, 1..7 classes.
        bc = np.bincount(raw.reshape(-1), minlength=8)
        counts += bc[1:8]

    if np.any(counts <= 0):
        raise RuntimeError(f"BLV-lite frequency estimate has zero class count: {counts.tolist()}")
    freq = counts / counts.sum()
    rarity = np.log(freq.max() / freq)
    if rarity.max() > 0:
        rarity = rarity / rarity.max()
    return torch.tensor(rarity, dtype=torch.float32), freq


def make_blv_run_epoch(tau=0.50, sample_masks=128):
    tau = float(tau)
    sample_masks = int(sample_masks)

    def run_epoch(
        model, loader, criterion, optimizer, device, scaler, amp,
        train=True, max_batches=0, grad_clip=0.0,
        fg_aux_bce_weight=0.7, fg_aux_dice_weight=0.5,
        bg_bce_weight=0.0, aux_loss_weight=0.4,
    ):
        if not train:
            return dg.sweep_domain_validation_epoch(
                model, loader, criterion, optimizer, device, scaler, amp,
                max_batches=max_batches, bg_bce_weight=bg_bce_weight,
            )

        if BLVState.scales is None:
            scales, freq = estimate_class_scales_from_masks(
                loader.dataset, sample_masks=sample_masks
            )
            BLVState.scales = scales
            print("BLV-lite | estimated class frequencies:")
            for name, f, s in zip(CLASS_NAMES, freq.tolist(), scales.tolist()):
                print(f"  {name:12s} freq={f:.6f} noise_scale={s:.4f}")

        scales = BLVState.scales.to(device=device).view(1, NUM_CLASSES, 1, 1)
        model.train(True)
        conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        total = 0.0
        n = skip_i = skip_nf = 0
        autocast_ctx = base.make_autocast(device, amp)
        pbar = tqdm(loader, desc="train", leave=False)

        for step, batch in enumerate(pbar, 1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            if not torch.any(y != IGNORE_INDEX):
                skip_i += 1
                if max_batches and step >= max_batches:
                    break
                continue

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                clean_logits = dg.extract_logits(model(x))
                noise = torch.randn_like(clean_logits) * scales.to(clean_logits.dtype) * tau
                perturbed = clean_logits + noise
                loss = criterion(perturbed, y)

            if not torch.isfinite(loss):
                skip_nf += 1
                optimizer.zero_grad(set_to_none=True)
                if max_batches and step >= max_batches:
                    break
                continue

            _backward_step(loss, model, optimizer, scaler, grad_clip)
            # Metrics use the unperturbed model output; BLV disappears at inference.
            dg.update_confusion(conf, clean_logits, y)
            total += float(loss.detach())
            n += 1
            pbar.set_postfix(loss=f"{total/max(n,1):.4f}")
            if max_batches and step >= max_batches:
                break

        print(f"BLV-lite | tau={tau:.2f} sample_masks={sample_masks} | inference=clean")
        return dg.ret_dict(total, total, conf, n, skip_i, skip_nf)

    return run_epoch


# -----------------------------------------------------------------------------
# TTA helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def predict_probs_scaled(model, x, scale: float, flip: bool = False):
    if scale <= 0:
        raise ValueError("scale must be >0")
    h, w = x.shape[-2:]
    if abs(float(scale) - 1.0) < 1e-9:
        xs = x
    else:
        xs = F.interpolate(
            x,
            scale_factor=float(scale),
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
    if flip:
        xs = torch.flip(xs, dims=[3])
    logits = dg.extract_logits(model(xs))
    if flip:
        logits = torch.flip(logits, dims=[3])
    if logits.shape[-2:] != (h, w):
        logits = F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
    return torch.softmax(logits.float(), dim=1)
