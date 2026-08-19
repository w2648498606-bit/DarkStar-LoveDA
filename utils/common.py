import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(path, model, optimizer, epoch, best_miou, args_dict, scheduler=None, scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'best_miou': best_miou,
        'args': args_dict,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    if optimizer is not None and isinstance(ckpt, dict) and ckpt.get('optimizer'):
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and isinstance(ckpt, dict) and ckpt.get('scheduler'):
        scheduler.load_state_dict(ckpt['scheduler'])
    if scaler is not None and isinstance(ckpt, dict) and ckpt.get('scaler'):
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt


def load_model_weights(path, model, map_location='cpu', strict=True):
    """Load model weights only; optimizer/scheduler state is intentionally ignored."""
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict:
        print(f"non-strict init: missing={list(incompatible.missing_keys)}, unexpected={list(incompatible.unexpected_keys)}")
    return ckpt


def append_csv(path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_ablation_base_weights(path, model, expected_source_model="mit_b2_upernet", map_location="cpu"):
    """Load *all* base-model tensors into a strict-ablation variant.

    Unlike a generic non-strict load, this function requires every source tensor to
    exist in the target with exactly the same shape. The only allowed missing target
    tensors are the newly-added ablation modules declared by
    ``model.ablation_new_prefixes``.
    """
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    source_model = (ckpt.get('args') or {}).get('model') if isinstance(ckpt, dict) else None
    if source_model and expected_source_model and source_model != expected_source_model:
        raise ValueError(
            f"Strict ablation expects checkpoint model={expected_source_model}, got {source_model}."
        )

    target = model.state_dict()
    source_keys = set(state.keys())
    target_keys = set(target.keys())

    missing_in_target = sorted(source_keys - target_keys)
    if missing_in_target:
        raise RuntimeError(
            "Target variant does not preserve the full base model. Source-only keys: "
            + str(missing_in_target[:20])
        )

    shape_mismatch = []
    for k in sorted(source_keys):
        if tuple(state[k].shape) != tuple(target[k].shape):
            shape_mismatch.append((k, tuple(state[k].shape), tuple(target[k].shape)))
    if shape_mismatch:
        raise RuntimeError(f"Strict ablation shape mismatch: {shape_mismatch[:10]}")

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected source keys after strict load: {incompatible.unexpected_keys}")

    new_keys = sorted(target_keys - source_keys)
    declared_prefixes = tuple(getattr(model, 'ablation_new_prefixes', ()))
    undeclared = [k for k in new_keys if not any(k.startswith(p) for p in declared_prefixes)]
    if undeclared:
        raise RuntimeError(
            "Target contains non-base tensors outside declared ablation modules: "
            + str(undeclared[:20])
        )
    if sorted(incompatible.missing_keys) != new_keys:
        raise RuntimeError(
            f"Missing-key mismatch after strict load. expected={new_keys[:20]}, "
            f"actual={sorted(incompatible.missing_keys)[:20]}"
        )

    report = {
        'checkpoint': str(path),
        'source_model': source_model,
        'loaded_base_tensors': len(source_keys),
        'new_variant_tensors': len(new_keys),
        'new_keys': new_keys,
        'source_epoch': int(ckpt.get('epoch', -1)) if isinstance(ckpt, dict) else -1,
        'source_best_miou': float(ckpt.get('best_miou', float('nan'))) if isinstance(ckpt, dict) else float('nan'),
    }
    return ckpt, report
