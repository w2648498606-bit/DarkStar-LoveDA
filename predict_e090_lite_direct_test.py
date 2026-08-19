from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.loveda import LoveDADataset
from models import build_model

DATA_ROOT = "/root/autodl-tmp/LoveDA"
CKPT = Path(
    "outputs/experiments/E090_e037_blv_lite/checkpoints/best_miou.pth"
)
TAG = "E090_lite_best_miou_direct"
OUT_DIR = Path("outputs/predictions") / f"{TAG}_test"
ZIP_PATH = Path(f"{TAG}_test.zip")

EXPECTED_TEST = 1796
NUM_CLASSES = 7


def extract_logits(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for k in ("logits", "out", "main"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError(f"Unsupported model output type: {type(out)}")


def load_model(device):
    if not CKPT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CKPT}")

    raw = torch.load(CKPT, map_location="cpu")
    state = raw.get("model", raw.get("state_dict", raw))

    model = build_model(
        "mit_b2_upernet",
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        state2 = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        model.load_state_dict(state2, strict=True)

    model = model.to(device).eval()

    print("checkpoint :", CKPT)
    if isinstance(raw, dict):
        print("epoch      :", raw.get("epoch"))
        print("best_miou  :", raw.get("best_miou"))
        print("model arg  :", (raw.get("args") or {}).get("model")
              if isinstance(raw.get("args"), dict) else None)

    return model


def resolve_names(batch, bs, offset):
    for key in ("name", "filename", "file_name", "image_name"):
        if key in batch:
            vals = batch[key]
            if isinstance(vals, (list, tuple)):
                return [Path(str(v)).with_suffix(".png").name for v in vals]
            if isinstance(vals, str) and bs == 1:
                return [Path(vals).with_suffix(".png").name]

    # Fallback should normally never be used with the current LoveDA dataset.
    return [f"{offset+i:06d}.png" for i in range(bs)]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device     :", device)

    model = load_model(device)

    ds = LoveDADataset(
        DATA_ROOT,
        "Test",
        crop_size=0,
        training=False,
        require_mask=False,
    )
    if len(ds) != EXPECTED_TEST:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST} Test images, got {len(ds)}"
        )

    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_ids = set()
    count = 0
    offset = 0

    with torch.inference_mode():
        for batch in tqdm(
            dl,
            desc="E090-lite direct Test",
            dynamic_ncols=True,
        ):
            x = batch["image"].to(device, non_blocking=True)

            # IMPORTANT:
            # E090-lite perturbation is TRAIN-TIME ONLY.
            # Hidden Test uses clean logits exactly.
            logits = extract_logits(model(x))

            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
            names = resolve_names(batch, len(pred), offset)

            if len(names) != len(pred):
                raise RuntimeError(
                    f"filename count {len(names)} != predictions {len(pred)}"
                )

            for arr, name in zip(pred, names):
                u = np.unique(arr)
                if arr.min() < 0 or arr.max() > 6:
                    raise RuntimeError(
                        f"Invalid class IDs in {name}: {u.tolist()}"
                    )
                all_ids.update(int(v) for v in u.tolist())
                Image.fromarray(arr, mode="L").save(OUT_DIR / name)
                count += 1

            offset += len(pred)

    pngs = sorted(OUT_DIR.glob("*.png"))
    if count != EXPECTED_TEST or len(pngs) != EXPECTED_TEST:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST} PNGs, "
            f"count={count}, files={len(pngs)}"
        )

    if not all_ids.issubset(set(range(7))):
        raise RuntimeError(
            f"Unexpected prediction IDs: {sorted(all_ids)}"
        )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        for p in pngs:
            # Flat ZIP: PNGs directly at root.
            z.write(p, arcname=p.name)

    with zipfile.ZipFile(ZIP_PATH) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC check failed")
        members = z.namelist()
        if len(members) != EXPECTED_TEST:
            raise RuntimeError(
                f"ZIP entries {len(members)} != {EXPECTED_TEST}"
            )
        if any("/" in name for name in members):
            raise RuntimeError("ZIP contains nested directories")

    print("")
    print("=" * 72)
    print("E090-LITE DIRECT HIDDEN TEST READY")
    print("=" * 72)
    print("checkpoint :", CKPT)
    print("images     :", len(pngs))
    print("class ids  :", sorted(all_ids))
    print("label rule : 0..6 directly, NO +1")
    print("submission :", ZIP_PATH.resolve())
    print("=" * 72)


if __name__ == "__main__":
    main()
