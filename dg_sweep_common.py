
import copy, csv, random, shutil, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, DataLoader
from tqdm import tqdm
import train as base

NUM_CLASSES = 7
IGNORE_INDEX = 255
DRO_ETA = 0.001
DRO_STATE = {"q": torch.tensor([0.5, 0.5], dtype=torch.float64), "epoch": 0}

DG_RUN = {"run_name": None}


def prepare_run_dir(run_name):
    """Archive an existing run instead of mixing old/new history or checkpoints."""
    run_dir = Path("outputs/experiments") / str(run_name)
    if run_dir.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archive = run_dir.with_name(run_dir.name + f"__previous_{stamp}")
        suffix = 1
        while archive.exists():
            archive = run_dir.with_name(run_dir.name + f"__previous_{stamp}_{suffix}")
            suffix += 1
        run_dir.rename(archive)
        print(f"DG safety | archived existing run: {run_dir} -> {archive}")
    return run_dir


def set_run_name(run_name):
    DG_RUN["run_name"] = str(run_name)


def _domain_history_path():
    run_name = DG_RUN.get("run_name")
    if not run_name:
        raise RuntimeError("DG run name was not set before validation.")
    return Path("outputs/experiments") / run_name / "domain_history.csv"

def canonical_domain(x):
    s = str(x).lower()
    if "urban" in s: return "urban"
    if "rural" in s: return "rural"
    raise ValueError(f"Unknown domain: {x}")


@contextmanager
def _temporary_python_random(seed):
    state = random.getstate()
    random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(state)


def _duplicate_seed(data_seed, epoch, idx, occurrence):
    # Stable integer mixing; independent of Python hash randomization.
    x = int(data_seed) & ((1 << 63) - 1)
    for v in (epoch, idx, occurrence):
        x ^= (int(v) + 0x9E3779B97F4A7C15 + ((x << 6) & ((1 << 63) - 1)) + (x >> 2))
        x &= ((1 << 63) - 1)
    return x


def make_occurrence_seed_dataset(original_cls):
    """
    Preserve the original strict-ablation augmentation for a sample's first
    occurrence in an epoch, but give deterministic *new* augmentation to any
    oversampled duplicate occurrence.

    Why: a 1:1 domain sampler can repeat the smaller domain. The base LoveDA
    deterministic seed depends only on (epoch, idx), so a repeated idx would
    otherwise receive an identical crop/augmentation in the same epoch.
    """
    class OccurrenceSeedDataset(original_cls):
        def __getitem__(self, key):
            if isinstance(key, tuple):
                idx, occurrence = int(key[0]), int(key[1])
            else:
                idx, occurrence = int(key), 0

            if occurrence == 0:
                # Exact original behavior for first use of this sample.
                return super().__getitem__(idx)

            if self.training and getattr(self, 'deterministic_augmentation', False):
                seed = _duplicate_seed(
                    getattr(self, 'data_seed', 42),
                    getattr(self, 'epoch', 0),
                    idx,
                    occurrence,
                )
                with _temporary_python_random(seed):
                    return self._getitem_impl(idx)

            return self._getitem_impl(idx)

    OccurrenceSeedDataset.__name__ = 'OccurrenceSeedDataset'
    return OccurrenceSeedDataset


class BalancedDomainSampler(Sampler):
    batch_size = 8
    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.batch_size = int(type(self).batch_size)
        if self.batch_size % 2:
            raise ValueError("batch size must be even")
        self.half = self.batch_size // 2
        self.num_batches = len(data_source) // self.batch_size
        self.length = self.num_batches * self.batch_size
        self.urban, self.rural = [], []
        for i, s in enumerate(data_source.samples):
            (self.urban if canonical_domain(s["domain"]) == "urban" else self.rural).append(i)
        if not self.urban or not self.rural:
            raise RuntimeError("Need both Urban and Rural samples")
        print(f"DG sampler | Urban={len(self.urban)} Rural={len(self.rural)} | batches={self.num_batches} | each=4+4")

    def __len__(self): return self.length

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        DRO_STATE["epoch"] = int(epoch)

    @staticmethod
    def _take(pool, cursor, k, rng):
        out = []
        while len(out) < k:
            if cursor >= len(pool):
                rng.shuffle(pool)
                cursor = 0
            n = min(k-len(out), len(pool)-cursor)
            out.extend(pool[cursor:cursor+n])
            cursor += n
        return out, cursor

    def __iter__(self):
        rng = random.Random(self.seed + 1000003*self.epoch)
        u, r = list(self.urban), list(self.rural)
        rng.shuffle(u); rng.shuffle(r)
        iu = ir = 0
        order = []
        seen = {}
        for _ in range(self.num_batches):
            bu, iu = self._take(u, iu, self.half, rng)
            br, ir = self._take(r, ir, self.half, rng)
            b = bu + br
            rng.shuffle(b)
            for idx in b:
                occurrence = seen.get(idx, 0)
                order.append((idx, occurrence))
                seen[idx] = occurrence + 1
        return iter(order)

def install_balanced_sampler(batch_size=8):
    BalancedDomainSampler.batch_size = int(batch_size)
    base.EpochShuffleSampler = BalancedDomainSampler

def reset_groupdro():
    DRO_STATE["q"] = torch.tensor([0.5,0.5], dtype=torch.float64)
    DRO_STATE["epoch"] = 0

def extract_logits(out):
    return out["logits"] if isinstance(out, dict) else out

def domain_indices(domains, device):
    u = [i for i,d in enumerate(domains) if canonical_domain(d)=="urban"]
    r = [i for i,d in enumerate(domains) if canonical_domain(d)=="rural"]
    if not u or not r:
        raise RuntimeError(f"Unbalanced batch: Urban={len(u)} Rural={len(r)}")
    return torch.tensor(u,device=device), torch.tensor(r,device=device)

def groupdro_objective(logits, masks, domains, criterion):
    ui, ri = domain_indices(domains, logits.device)
    lu = criterion(logits.index_select(0,ui), masks.index_select(0,ui))
    lr = criterion(logits.index_select(0,ri), masks.index_select(0,ri))
    with torch.no_grad():
        lv = torch.tensor([float(lu.detach()),float(lr.detach())], dtype=torch.float64)
        q = DRO_STATE["q"] * torch.exp(DRO_ETA*lv)
        q = q/q.sum()
        DRO_STATE["q"] = q
    qf = DRO_STATE["q"].to(device=logits.device,dtype=lu.dtype)
    return qf[0]*lu + qf[1]*lr, lu, lr

def confusion_metrics(conf):
    conf = conf.double()
    tp = torch.diag(conf); gt=conf.sum(1); pr=conf.sum(0)
    union = gt+pr-tp
    iou = torch.zeros(NUM_CLASSES,dtype=torch.float64)
    good = union>0
    iou[good] = tp[good]/union[good]
    miou = float(iou[good].mean()) if torch.any(good) else 0.0
    pa = float(tp.sum()/conf.sum()) if conf.sum()>0 else 0.0
    return iou.tolist(),miou,pa

def update_confusion(conf, logits, masks):
    with torch.no_grad():
        pred = logits.argmax(1)
        valid = masks!=IGNORE_INDEX
        if torch.any(valid):
            ids = masks[valid].long()*NUM_CLASSES + pred[valid].long()
            conf += torch.bincount(ids,minlength=49).reshape(7,7).cpu()

def ret_dict(total_loss,total_main,conf,n,skip_i,skip_nf):
    iou,miou,pa = confusion_metrics(conf)
    n=max(n,1)
    return {"loss":total_loss/n,"main_loss":total_main/n,"fg_bce":0.0,"fg_dice":0.0,
            "bg_bce":0.0,"aux_ce":0.0,"miou":miou,"pixel_acc":pa,"iou":iou,
            "skipped_all_ignore":skip_i,"skipped_nonfinite":skip_nf}


def _append_domain_history(epoch, overall, urban, rural):
    class_names = ["background","building","road","water","barren","forest","agriculture"]
    oi, om, opa = confusion_metrics(overall)
    ui, um, upa = confusion_metrics(urban)
    ri, rm, rpa = confusion_metrics(rural)
    worst = min(um, rm)
    score = 0.5 * om + 0.5 * worst

    row = {
        "epoch": int(epoch),
        "overall_miou": om,
        "urban_miou": um,
        "rural_miou": rm,
        "worst_domain_miou": worst,
        "dg_balanced_score": score,
        "overall_pixel_acc": opa,
        "urban_pixel_acc": upa,
        "rural_pixel_acc": rpa,
    }
    for name, v in zip(class_names, oi): row[f"overall_iou_{name}"] = v
    for name, v in zip(class_names, ui): row[f"urban_iou_{name}"] = v
    for name, v in zip(class_names, ri): row[f"rural_iou_{name}"] = v

    path = _domain_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(
        "DG Val | "
        f"epoch={int(epoch):02d} overall={om:.4f} Urban={um:.4f} Rural={rm:.4f} "
        f"worst={worst:.4f} score={score:.4f}"
    )
    return row


@torch.no_grad()
def domain_validation_epoch(
    model, loader, criterion, optimizer, device, scaler, amp,
    max_batches=0, bg_bce_weight=0.0,
):
    # These experiments deliberately use the plain MiT-B2+UPerNet head and bg_bce=0.
    if float(bg_bce_weight) != 0.0:
        raise RuntimeError("DG suite expects --bg-bce-weight 0 for clean E037 comparison.")

    model.eval()
    overall = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    urban = torch.zeros_like(overall)
    rural = torch.zeros_like(overall)
    total_loss = 0.0
    n = 0
    skip_i = 0
    skip_nf = 0
    autocast_ctx = base.make_autocast(device, amp)
    pbar = tqdm(loader, desc="val", leave=False)

    for step, batch in enumerate(pbar, 1):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        domains = batch["domain"]

        if not torch.any(y != IGNORE_INDEX):
            skip_i += 1
            if max_batches and step >= max_batches:
                break
            continue

        with autocast_ctx():
            logits = extract_logits(model(x))
            loss = criterion(logits, y)

        if not torch.isfinite(loss):
            skip_nf += 1
            if max_batches and step >= max_batches:
                break
            continue

        pred = logits.argmax(1)
        for i, d in enumerate(domains):
            valid = y[i] != IGNORE_INDEX
            if not torch.any(valid):
                continue
            ids = y[i][valid].long() * NUM_CLASSES + pred[i][valid].long()
            hist = torch.bincount(ids, minlength=49).reshape(7,7).cpu()
            overall += hist
            if canonical_domain(d) == "urban": urban += hist
            else: rural += hist

        total_loss += float(loss.detach())
        n += 1
        pbar.set_postfix(loss=f"{total_loss/max(n,1):.4f}")
        if max_batches and step >= max_batches:
            break

    epoch = int(DRO_STATE.get("epoch", 0))
    _append_domain_history(epoch, overall, urban, rural)
    return ret_dict(total_loss, total_loss, overall, n, skip_i, skip_nf)


ORIGINAL_RUN_EPOCH = base.run_epoch

def groupdro_run_epoch(model,loader,criterion,optimizer,device,scaler,amp,train=True,max_batches=0,
                       grad_clip=0.0,fg_aux_bce_weight=0.7,fg_aux_dice_weight=0.5,bg_bce_weight=0.0,
                       aux_loss_weight=0.4):
    if not train:
        return domain_validation_epoch(
            model,loader,criterion,optimizer,device,scaler,False,
            max_batches=max_batches,bg_bce_weight=bg_bce_weight,
        )

    model.train(True)
    conf=torch.zeros((7,7),dtype=torch.int64)
    tl=tm=tu=tr=0.0; n=skip_i=skip_nf=0
    autocast_ctx=base.make_autocast(device,amp)
    pbar=tqdm(loader,desc="train",leave=False)
    for step,batch in enumerate(pbar,1):
        x=batch["image"].to(device,non_blocking=True)
        y=batch["mask"].to(device,non_blocking=True)
        domains=batch["domain"]
        if not torch.any(y!=IGNORE_INDEX):
            skip_i+=1
            continue
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits=extract_logits(model(x))
            loss,lu,lr=groupdro_objective(logits,y,domains,criterion)
        if not torch.isfinite(loss):
            skip_nf+=1; continue
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip>0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            optimizer.step()
        update_confusion(conf,logits,y)
        tl+=float(loss.detach()); tm+=float(loss.detach()); tu+=float(lu.detach()); tr+=float(lr.detach()); n+=1
        q=DRO_STATE["q"]; pbar.set_postfix(loss=f"{tl/n:.4f}",qR=f"{float(q[1]):.3f}")
        if max_batches and step>=max_batches: break
    q=DRO_STATE["q"]; den=max(n,1)
    print(f"GroupDRO | UrbanLoss={tu/den:.4f} RuralLoss={tr/den:.4f} | qUrban={float(q[0]):.3f} qRural={float(q[1]):.3f}")
    return ret_dict(tl,tm,conf,n,skip_i,skip_nf)


def selective_whitening_loss(feat, domains, top_fraction=0.10):
    if feat.ndim!=4:
        raise ValueError(f"Expected BCHW, got {tuple(feat.shape)}")
    u=[i for i,d in enumerate(domains) if canonical_domain(d)=="urban"]
    r=[i for i,d in enumerate(domains) if canonical_domain(d)=="rural"]
    if not u or not r:
        return feat.sum()*0.0,0
    x=feat.float().flatten(2)
    x=x-x.mean(2,keepdim=True)
    x=x/(x.std(2,keepdim=True,unbiased=False)+1e-5)
    corr=torch.bmm(x,x.transpose(1,2))/x.shape[-1]
    ui=torch.tensor(u,device=feat.device); ri=torch.tensor(r,device=feat.device)
    cu=corr.index_select(0,ui).mean(0); cr=corr.index_select(0,ri).mean(0)
    c=cu.shape[0]
    off=~torch.eye(c,device=feat.device,dtype=torch.bool)
    score=(cu-cr).abs().detach()[off]
    k=max(1,int(score.numel()*float(top_fraction)))
    th=torch.topk(score,k=k).values[-1]
    sel=off & ((cu-cr).abs().detach()>=th)
    loss=0.5*(cu[sel].pow(2).mean()+cr[sel].pow(2).mean())
    return loss,int(sel.sum().item())

WHITEN_WEIGHT=0.10

def whitening_groupdro_run_epoch(model,loader,criterion,optimizer,device,scaler,amp,train=True,max_batches=0,
                                 grad_clip=0.0,fg_aux_bce_weight=0.7,fg_aux_dice_weight=0.5,bg_bce_weight=0.0,
                                 aux_loss_weight=0.4):
    if not train:
        return domain_validation_epoch(
            model,loader,criterion,optimizer,device,scaler,False,
            max_batches=max_batches,bg_bce_weight=bg_bce_weight,
        )
    for a in ("backbone","decoder","head"):
        if not hasattr(model,a): raise RuntimeError(f"E070 model missing {a}")
    model.train(True)
    conf=torch.zeros((7,7),dtype=torch.int64)
    tl=tm=tu=tr=tw=0.0; ns=0; n=skip_i=skip_nf=0
    autocast_ctx=base.make_autocast(device,amp)
    pbar=tqdm(loader,desc="train",leave=False)
    for step,batch in enumerate(pbar,1):
        x=batch["image"].to(device,non_blocking=True); y=batch["mask"].to(device,non_blocking=True); domains=batch["domain"]
        if not torch.any(y!=IGNORE_INDEX): skip_i+=1; continue
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            feats=model.backbone(x)
            dec=model.decoder(feats)
            small=model.head(dec)
            logits=model._upsample_logits(small,x) if hasattr(model,"_upsample_logits") else F.interpolate(
                small,size=x.shape[-2:],mode="bilinear",align_corners=False)
            main,lu,lr=groupdro_objective(logits,y,domains,criterion)
            white,selected=selective_whitening_loss(feats[1],domains,0.10)
            loss=main+WHITEN_WEIGHT*white
        if not torch.isfinite(loss): skip_nf+=1; continue
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip>0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            optimizer.step()
        update_confusion(conf,logits,y)
        tl+=float(loss.detach()); tm+=float(main.detach()); tu+=float(lu.detach()); tr+=float(lr.detach())
        tw+=float(white.detach()); ns+=selected; n+=1
        pbar.set_postfix(loss=f"{tl/n:.4f}",white=f"{tw/n:.4f}")
        if max_batches and step>=max_batches: break
    den=max(n,1); q=DRO_STATE["q"]
    print(f"SelectiveWhitening | raw={tw/den:.5f} weighted={WHITEN_WEIGHT*tw/den:.5f} | selected_cov={ns//den} | qU={float(q[0]):.3f} qR={float(q[1]):.3f}")
    return ret_dict(tl,tm,conf,n,skip_i,skip_nf)

IMAGENET_MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
IMAGENET_STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)

def unnormalize_image(t):
    return (t.cpu()*IMAGENET_STD+IMAGENET_MEAN).clamp(0,1)

def normalize_image(t):
    return ((t-IMAGENET_MEAN)/IMAGENET_STD).contiguous().float()

def fourier_style_mix(src,donor,beta=0.05,lam=0.5):
    a=src.detach().cpu().numpy().astype(np.float32)
    b=donor.detach().cpu().numpy().astype(np.float32)
    if a.shape!=b.shape: raise ValueError("shape mismatch")
    fa=np.fft.fft2(a,axes=(-2,-1)); fb=np.fft.fft2(b,axes=(-2,-1))
    aa=np.fft.fftshift(np.abs(fa),axes=(-2,-1)); ab=np.fft.fftshift(np.abs(fb),axes=(-2,-1))
    phase=np.angle(fa)
    h,w=a.shape[-2:]; rad=max(1,int(min(h,w)*beta)); cy,cx=h//2,w//2
    ys=slice(max(0,cy-rad),min(h,cy+rad+1)); xs=slice(max(0,cx-rad),min(w,cx+rad+1))
    aa[:,ys,xs]=(1-lam)*aa[:,ys,xs]+lam*ab[:,ys,xs]
    aa=np.fft.ifftshift(aa,axes=(-2,-1))
    out=np.fft.ifft2(aa*np.exp(1j*phase),axes=(-2,-1)).real
    return torch.from_numpy(np.clip(out,0,1).astype(np.float32))

def make_fosmix_dataset(original_cls,probability=0.50,beta=0.05):
    class FOSMixDataset(original_cls):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.fosmix_probability=float(probability); self.fosmix_beta=float(beta)
            self._u=[]; self._r=[]
            for i,s in enumerate(self.samples):
                (self._u if canonical_domain(s["domain"])=="urban" else self._r).append(i)
        def _getitem_impl(self,idx):
            result=original_cls._getitem_impl(self,idx)
            if not self.training or random.random()>=self.fosmix_probability:
                return result
            pool=self._r if canonical_domain(self.samples[idx]["domain"])=="urban" else self._u
            donor_idx=random.choice(pool)
            donor=original_cls._getitem_impl(self,donor_idx)
            src=unnormalize_image(result["image"]); dnr=unnormalize_image(donor["image"])
            lam=random.uniform(0.30,0.70)
            result["image"]=normalize_image(fourier_style_mix(src,dnr,self.fosmix_beta,lam))
            return result
    FOSMixDataset.__name__="FOSMixDataset"
    return FOSMixDataset

def install_epoch_snapshots(run_name):
    original=base.save_checkpoint
    snap=Path("outputs/experiments")/run_name/"checkpoints"/"epochs"
    def wrapped(path,model,optimizer,epoch,best_miou,args,scheduler=None,scaler=None):
        original(path,model,optimizer,epoch,best_miou,args,scheduler=scheduler,scaler=scaler)
        if int(epoch)>0 and Path(path).name=="last.pth":
            snap.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,snap/f"epoch_{int(epoch):02d}.pth")
    base.save_checkpoint=wrapped
    return snap

def average_checkpoints(paths,out_path,meta=None):
    ck=[torch.load(Path(x),map_location="cpu") for x in paths]
    states=[c["model"] for c in ck]
    out_state={}
    for k in states[0]:
        vals=[s[k] for s in states]
        if torch.is_floating_point(vals[0]):
            acc=vals[0].double().clone()
            for v in vals[1:]: acc+=v.double()
            out_state[k]=(acc/len(vals)).to(vals[0].dtype)
        else:
            out_state[k]=vals[len(vals)//2].clone()
    out=copy.deepcopy(ck[len(ck)//2]); out["model"]=out_state
    out["swad_source_epochs"]=[int(c.get("epoch",-1)) for c in ck]
    if meta: out["swad_meta"]=dict(meta)
    out_path=Path(out_path); out_path.parent.mkdir(parents=True,exist_ok=True); torch.save(out,out_path)
    return out_path

def choose_swad_window(history_csv,snapshot_dir):
    rows=[]
    with open(history_csv,"r",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            try: rows.append((int(row["epoch"]),float(row["val_loss"])))
            except: pass
    if not rows: raise RuntimeError("history.csv has no valid rows")
    center=min(rows,key=lambda z:z[1])[0]
    available={e for e,_ in rows}
    chosen=[e for e in (center-1,center,center+1) if e in available and (Path(snapshot_dir)/f"epoch_{e:02d}.pth").exists()]
    if len(chosen)<2:
        cand=sorted(available,key=lambda e:abs(e-center))
        chosen=sorted([e for e in cand if (Path(snapshot_dir)/f"epoch_{e:02d}.pth").exists()][:3])
    if len(chosen)<2: raise RuntimeError("not enough snapshots")
    return center,chosen

# ------------------------------------------------------------------
# Domain-aware post-evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint_domains(
    checkpoint,
    data_root="/root/autodl-tmp/LoveDA",
    batch_size=8,
    workers=8,
    device="cuda",
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = base.build_model(
        "mit_b2_upernet",
        num_classes=NUM_CLASSES,
        pretrained=False,
    ).to(dev)

    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state, strict=True)
    model.eval()

    ds = base.LoveDADataset(
        data_root,
        "Val",
        crop_size=0,
        training=False,
        require_mask=True,
    )

    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(int(workers) > 0),
    )

    overall = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    urban = torch.zeros_like(overall)
    rural = torch.zeros_like(overall)

    pbar = tqdm(dl, desc="domain-val", leave=False)

    for batch in pbar:
        images = batch["image"].to(dev, non_blocking=True)
        masks = batch["mask"].to(dev, non_blocking=True)
        domains = batch["domain"]

        logits = extract_logits(model(images))
        pred = logits.argmax(dim=1)

        for i, d in enumerate(domains):
            valid = masks[i] != IGNORE_INDEX
            if not torch.any(valid):
                continue

            ids = masks[i][valid].long() * NUM_CLASSES + pred[i][valid].long()
            hist = torch.bincount(
                ids,
                minlength=NUM_CLASSES * NUM_CLASSES,
            ).reshape(NUM_CLASSES, NUM_CLASSES).cpu()

            overall += hist
            if canonical_domain(d) == "urban":
                urban += hist
            else:
                rural += hist

    class_names = [
        "background",
        "building",
        "road",
        "water",
        "barren",
        "forest",
        "agriculture",
    ]

    results = {}
    print(f"\nDG POST-EVAL | checkpoint={checkpoint}")

    for name, conf in (("overall", overall), ("Urban", urban), ("Rural", rural)):
        iou, miou, pa = confusion_metrics(conf)
        results[name] = {
            "miou": miou,
            "pixel_acc": pa,
            "iou": iou,
        }

        print(f"[{name}] mIoU={miou:.6f} pixel_acc={pa:.6f}")
        for cls, value in zip(class_names, iou):
            print(f"  {cls:12s} IoU={value:.6f}")

    worst = min(results["Urban"]["miou"], results["Rural"]["miou"])
    balanced_score = 0.5 * results["overall"]["miou"] + 0.5 * worst

    print(f"worst_domain_mIoU={worst:.6f}")
    print(f"DG_balanced_score={balanced_score:.6f}")

    results["worst_domain_miou"] = worst
    results["dg_balanced_score"] = balanced_score
    return results



def _history_rows(history_csv):
    rows = []
    with open(history_csv, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    'epoch': int(row['epoch']),
                    'val_miou': float(row['val_miou']),
                    'val_loss': float(row['val_loss']),
                })
            except Exception:
                continue
    if not rows:
        raise RuntimeError(f'No valid history rows in {history_csv}')
    return rows


def choose_dg_candidates(history_csv, snapshot_dir, top_miou=2):
    rows = _history_rows(history_csv)
    ranked = sorted(rows, key=lambda r: r['val_miou'], reverse=True)
    chosen = [r['epoch'] for r in ranked[:int(top_miou)]]
    chosen.append(min(rows, key=lambda r: r['val_loss'])['epoch'])

    out = []
    for ep in chosen:
        p = Path(snapshot_dir) / f'epoch_{ep:02d}.pth'
        if p.exists() and ep not in out:
            out.append(ep)
    if not out:
        raise RuntimeError('No candidate epoch snapshots found for DG selection.')
    return out


def _domain_history_rows(path):
    rows=[]
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "epoch": int(row["epoch"]),
                    "overall_miou": float(row["overall_miou"]),
                    "urban_miou": float(row["urban_miou"]),
                    "rural_miou": float(row["rural_miou"]),
                    "worst_domain_miou": float(row["worst_domain_miou"]),
                    "dg_balanced_score": float(row["dg_balanced_score"]),
                })
            except Exception:
                continue
    if not rows:
        raise RuntimeError(f"No valid domain-history rows in {path}")
    return rows


def select_best_dg_checkpoint(run_name, data_root='/root/autodl-tmp/LoveDA'):
    run_dir = Path('outputs/experiments') / run_name
    domain_history = run_dir / 'domain_history.csv'
    snapshot_dir = run_dir / 'checkpoints' / 'epochs'
    rows = _domain_history_rows(domain_history)
    best = max(rows, key=lambda r: r['dg_balanced_score'])
    src = snapshot_dir / f"epoch_{best['epoch']:02d}.pth"
    if not src.exists():
        raise FileNotFoundError(f"DG-selected snapshot missing: {src}")
    dst = run_dir / 'checkpoints' / 'best_dg.pth'
    shutil.copy2(src, dst)
    print(
        'DG BEST | '
        f"epoch={best['epoch']} overall={best['overall_miou']:.6f} "
        f"Urban={best['urban_miou']:.6f} Rural={best['rural_miou']:.6f} "
        f"worst={best['worst_domain_miou']:.6f} score={best['dg_balanced_score']:.6f} | "
        f'saved={dst}'
    )
    return dst, best


@torch.no_grad()
def recalibrate_batchnorm(
    checkpoint,
    output_path,
    data_root="/root/autodl-tmp/LoveDA",
    epoch=1,
    batch_size=8,
    workers=8,
    device="cuda",
):
    """Recompute BN running statistics after checkpoint weight averaging."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("model", raw) if isinstance(raw, dict) else raw

    model = base.build_model(
        "mit_b2_upernet", num_classes=NUM_CLASSES, pretrained=False
    ).to(dev)
    model.load_state_dict(state, strict=True)

    bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_modules:
        shutil.copy2(checkpoint, output_path)
        print("SWAD BN recalibration | no BatchNorm modules; checkpoint copied unchanged")
        return Path(output_path)

    original_cls = base.LoveDADataset
    occ_cls = make_occurrence_seed_dataset(original_cls)
    ds = occ_cls(
        data_root, "Train", crop_size=512, training=True, require_mask=True,
        min_valid_ratio=0.05, crop_retries=10,
        rare_crop=True, rare_crop_prob=0.4, rare_classes=(4,5), rare_min_ratio=0.02,
        augmentation="multiscale", multi_scale_values=(0.75,1.0,1.25,1.5),
        deterministic_augmentation=True, data_seed=20260809,
    )
    sampler = BalancedDomainSampler(ds, seed=20260809)
    ds.set_epoch(int(epoch)); sampler.set_epoch(int(epoch))
    dl = DataLoader(
        ds, batch_size=int(batch_size), sampler=sampler, shuffle=False,
        num_workers=int(workers), pin_memory=(dev.type=="cuda"), drop_last=True,
        persistent_workers=False,
    )

    model.eval()
    old_momenta = {}
    for m in bn_modules:
        old_momenta[m] = m.momentum
        m.reset_running_stats()
        m.momentum = None
        m.train()

    pbar = tqdm(dl, desc="SWAD-BN", leave=False)
    for batch in pbar:
        x = batch["image"].to(dev, non_blocking=True)
        model(x)

    for m in bn_modules:
        m.momentum = old_momenta[m]
    model.eval()

    if isinstance(raw, dict):
        raw["model"] = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        raw["bn_recalibrated"] = True
        raw["bn_recalibration_epoch"] = int(epoch)
        out_obj = raw
    else:
        out_obj = {k:v.detach().cpu() for k,v in model.state_dict().items()}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, output_path)
    print(f"SWAD BN recalibration | modules={len(bn_modules)} | saved={output_path}")
    return output_path


# ------------------------------------------------------------------
# Exact E037 baseline arguments reused by E068-E071
# ------------------------------------------------------------------

def common_e037_argv(run_name, epochs=12):
    return [
        "train_dg.py",
        "--data-root", "/root/autodl-tmp/LoveDA",
        "--model", "mit_b2_upernet",
        "--run-name", str(run_name),
        "--epochs", str(int(epochs)),
        "--batch-size", "8",
        "--val-batch-size", "8",
        "--crop-size", "512",
        "--min-valid-ratio", "0.05",
        "--crop-retries", "10",
        "--workers", "8",
        "--optimizer", "adamw",
        "--lr", "1e-4",
        "--backbone-lr-mult", "0.2",
        "--weight-decay", "1e-4",
        "--scheduler", "cosine",
        "--augmentation", "multiscale",
        "--multi-scale-values", "0.75,1.0,1.25,1.5",
        "--rare-crop",
        "--rare-crop-prob", "0.4",
        "--rare-classes", "barren,forest",
        "--rare-min-ratio", "0.02",
        "--loss", "ce_lovasz",
        "--lovasz-weight", "0.5",
        "--bg-bce-weight", "0",
        "--init-encoder-from",
        "outputs/experiments/E007_segformer_b2_20ep/checkpoints/best_miou.pth",
        "--strict-ablation",
        "--model-seed", "3407",
        "--data-seed", "20260809",
        "--train-seed", "271828",
        "--device", "cuda",
    ]



# ======================================================================
# E072-E079 unattended sweep extensions
# ======================================================================

import json as _json

SWEEP_STATE = {
    "epoch": 0,
    "last_domain_row": None,
    "best_dg_score": -1.0,
    "best_dg_epoch": None,
}

ORIGINAL_EPOCH_SAMPLER = base.EpochShuffleSampler
ORIGINAL_DATASET_CLASS = base.LoveDADataset


def reset_sweep_state():
    SWEEP_STATE["epoch"] = 0
    SWEEP_STATE["last_domain_row"] = None
    SWEEP_STATE["best_dg_score"] = -1.0
    SWEEP_STATE["best_dg_epoch"] = None


class TrackingEpochSampler(ORIGINAL_EPOCH_SAMPLER):
    """Exact base sampler plus an epoch tracker for domain-aware validation."""
    def set_epoch(self, epoch):
        SWEEP_STATE["epoch"] = int(epoch)
        return super().set_epoch(epoch)


class SweepBalancedDomainSampler(BalancedDomainSampler):
    """V3 4+4 sampler plus sweep epoch tracking."""
    def set_epoch(self, epoch):
        SWEEP_STATE["epoch"] = int(epoch)
        return super().set_epoch(epoch)


def install_tracking_sampler():
    base.EpochShuffleSampler = TrackingEpochSampler


def install_sweep_balanced_sampler(batch_size=8):
    SweepBalancedDomainSampler.batch_size = int(batch_size)
    base.EpochShuffleSampler = SweepBalancedDomainSampler


def _sweep_domain_history_path():
    run_name = DG_RUN.get("run_name")
    if not run_name:
        raise RuntimeError("Sweep run name was not set.")
    return Path("outputs/experiments") / run_name / "domain_history.csv"


def _sweep_append_domain_history(epoch, overall, urban, rural):
    class_names = [
        "background", "building", "road", "water",
        "barren", "forest", "agriculture",
    ]
    oi, om, opa = confusion_metrics(overall)
    ui, um, upa = confusion_metrics(urban)
    ri, rm, rpa = confusion_metrics(rural)

    worst = min(um, rm)
    score = 0.5 * om + 0.5 * worst

    row = {
        "epoch": int(epoch),
        "overall_miou": float(om),
        "urban_miou": float(um),
        "rural_miou": float(rm),
        "worst_domain_miou": float(worst),
        "dg_balanced_score": float(score),
        "overall_pixel_acc": float(opa),
        "urban_pixel_acc": float(upa),
        "rural_pixel_acc": float(rpa),
    }

    for name, v in zip(class_names, oi):
        row[f"overall_iou_{name}"] = float(v)
    for name, v in zip(class_names, ui):
        row[f"urban_iou_{name}"] = float(v)
    for name, v in zip(class_names, ri):
        row[f"rural_iou_{name}"] = float(v)

    path = _sweep_domain_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    SWEEP_STATE["last_domain_row"] = dict(row)

    print(
        "DG Val | "
        f"epoch={int(epoch):02d} overall={om:.4f} "
        f"Urban={um:.4f} Rural={rm:.4f} "
        f"worst={worst:.4f} score={score:.4f}"
    )
    return row


@torch.no_grad()
def sweep_domain_validation_epoch(
    model, loader, criterion, optimizer, device, scaler, amp,
    max_batches=0, bg_bce_weight=0.0,
):
    if float(bg_bce_weight) != 0.0:
        raise RuntimeError(
            "E072-E079 expect --bg-bce-weight 0 for a clean E037 comparison."
        )

    model.eval()

    overall = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    urban = torch.zeros_like(overall)
    rural = torch.zeros_like(overall)

    total_loss = 0.0
    n = 0
    skip_i = 0
    skip_nf = 0

    autocast_ctx = base.make_autocast(device, amp)
    pbar = tqdm(loader, desc="val", leave=False)

    for step, batch in enumerate(pbar, 1):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        domains = batch["domain"]

        if not torch.any(y != IGNORE_INDEX):
            skip_i += 1
            if max_batches and step >= max_batches:
                break
            continue

        with autocast_ctx():
            logits = extract_logits(model(x))
            loss = criterion(logits, y)

        if not torch.isfinite(loss):
            skip_nf += 1
            if max_batches and step >= max_batches:
                break
            continue

        pred = logits.argmax(1)

        for i, d in enumerate(domains):
            valid = y[i] != IGNORE_INDEX
            if not torch.any(valid):
                continue

            ids = (
                y[i][valid].long() * NUM_CLASSES
                + pred[i][valid].long()
            )

            hist = torch.bincount(
                ids,
                minlength=NUM_CLASSES * NUM_CLASSES,
            ).reshape(NUM_CLASSES, NUM_CLASSES).cpu()

            overall += hist

            if canonical_domain(d) == "urban":
                urban += hist
            else:
                rural += hist

        total_loss += float(loss.detach())
        n += 1

        pbar.set_postfix(
            loss=f"{total_loss / max(n,1):.4f}"
        )

        if max_batches and step >= max_batches:
            break

    epoch = int(SWEEP_STATE.get("epoch", 0))

    # Defensive fallback if an external sampler does not expose set_epoch.
    if epoch <= 0:
        prev = SWEEP_STATE.get("last_domain_row")
        epoch = 1 if prev is None else int(prev["epoch"]) + 1
        SWEEP_STATE["epoch"] = epoch

    _sweep_append_domain_history(
        epoch,
        overall,
        urban,
        rural,
    )

    return ret_dict(
        total_loss,
        total_loss,
        overall,
        n,
        skip_i,
        skip_nf,
    )


def _compact_payload(model, epoch, best_miou, args, extra=None):
    payload = {
        "model": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
        "epoch": int(epoch),
        "best_miou": float(best_miou),
        "compact_checkpoint": True,
    }

    try:
        payload["args"] = dict(vars(args))
    except Exception:
        payload["args"] = {}

    row = SWEEP_STATE.get("last_domain_row")
    if row is not None:
        payload["dg_metrics"] = dict(row)

    if extra:
        payload.update(dict(extra))

    return payload


def install_compact_checkpointing(run_name):
    """
    Avoid the V3 disk-space problem.

    - last.pth is compact (model only, no optimizer state)
    - best_miou.pth is compact
    - best_dg.pth is saved automatically when DG score improves
    - no per-epoch snapshot directory is created
    """
    run_name = str(run_name)

    def wrapped(
        path,
        model,
        optimizer,
        epoch,
        best_miou,
        args,
        scheduler=None,
        scaler=None,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = _compact_payload(
            model,
            epoch,
            best_miou,
            args,
        )

        torch.save(payload, path)

        row = SWEEP_STATE.get("last_domain_row")

        if (
            path.name == "last.pth"
            and row is not None
            and int(row["epoch"]) == int(epoch)
        ):
            score = float(row["dg_balanced_score"])

            if score > float(SWEEP_STATE["best_dg_score"]):
                SWEEP_STATE["best_dg_score"] = score
                SWEEP_STATE["best_dg_epoch"] = int(epoch)

                dg_path = (
                    Path("outputs/experiments")
                    / run_name
                    / "checkpoints"
                    / "best_dg.pth"
                )

                torch.save(
                    _compact_payload(
                        model,
                        epoch,
                        best_miou,
                        args,
                        extra={
                            "selected_by": "DG_balanced_score",
                            "dg_balanced_score": score,
                        },
                    ),
                    dg_path,
                )

                print(
                    "DG BEST UPDATE | "
                    f"epoch={int(epoch)} score={score:.6f} | "
                    f"saved={dg_path}"
                )

    base.save_checkpoint = wrapped


def finalize_compact_run(run_name, method_name):
    run_dir = Path("outputs/experiments") / str(run_name)
    hist = run_dir / "domain_history.csv"

    rows = _domain_history_rows(hist)

    best_dg = max(
        rows,
        key=lambda r: r["dg_balanced_score"],
    )

    best_overall = max(
        rows,
        key=lambda r: r["overall_miou"],
    )

    summary = {
        "run_name": str(run_name),
        "method": str(method_name),
        "best_dg_epoch": int(best_dg["epoch"]),
        "best_dg_score": float(best_dg["dg_balanced_score"]),
        "best_dg_overall_miou": float(best_dg["overall_miou"]),
        "best_dg_urban_miou": float(best_dg["urban_miou"]),
        "best_dg_rural_miou": float(best_dg["rural_miou"]),
        "best_dg_worst_miou": float(best_dg["worst_domain_miou"]),
        "best_overall_epoch": int(best_overall["epoch"]),
        "best_overall_miou": float(best_overall["overall_miou"]),
    }

    summary_path = run_dir / "sweep_summary.json"
    summary_path.write_text(
        _json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # last.pth is only useful for interrupted training. Remove after a
    # successful completed run to reduce disk usage.
    last = run_dir / "checkpoints" / "last.pth"
    if last.exists():
        last.unlink()

    print(
        "SWEEP SUMMARY | "
        f"{run_name} | "
        f"DG epoch={summary['best_dg_epoch']} "
        f"score={summary['best_dg_score']:.6f} "
        f"overall={summary['best_dg_overall_miou']:.6f} "
        f"Urban={summary['best_dg_urban_miou']:.6f} "
        f"Rural={summary['best_dg_rural_miou']:.6f}"
    )

    return summary


def clean_train_run_epoch(
    model, loader, criterion, optimizer, device, scaler, amp,
    train=True, max_batches=0, grad_clip=0.0,
    fg_aux_bce_weight=0.7, fg_aux_dice_weight=0.5,
    bg_bce_weight=0.0, aux_loss_weight=0.4,
):
    if not train:
        return sweep_domain_validation_epoch(
            model, loader, criterion, optimizer, device, scaler, amp,
            max_batches=max_batches,
            bg_bce_weight=bg_bce_weight,
        )

    # The E072-E079 suite intentionally keeps the E037 segmentation objective
    # plain: criterion = CE + 0.5 Lovasz. No teacher, no barren auxiliary.
    model.train(True)

    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    tl = 0.0
    n = 0
    skip_i = 0
    skip_nf = 0

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
            logits = extract_logits(model(x))
            loss = criterion(logits, y)

        if not torch.isfinite(loss):
            skip_nf += 1
            optimizer.zero_grad(set_to_none=True)
            if max_batches and step >= max_batches:
                break
            continue

        if scaler is not None:
            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()

        update_confusion(conf, logits, y)

        tl += float(loss.detach())
        n += 1

        pbar.set_postfix(
            loss=f"{tl/max(n,1):.4f}"
        )

        if max_batches and step >= max_batches:
            break

    return ret_dict(
        tl,
        tl,
        conf,
        n,
        skip_i,
        skip_nf,
    )


# ----------------------------------------------------------------------
# MixStyle: lightweight decoder-feature adaptation.
# ----------------------------------------------------------------------

def _opposite_domain_permutation(domains, device):
    u = [
        i for i, d in enumerate(domains)
        if canonical_domain(d) == "urban"
    ]
    r = [
        i for i, d in enumerate(domains)
        if canonical_domain(d) == "rural"
    ]

    if not u or not r:
        return torch.randperm(
            len(domains),
            device=device,
        )

    # With the balanced sampler this is exactly 4+4.
    random.shuffle(u)
    random.shuffle(r)

    perm = list(range(len(domains)))

    for a, b in zip(u, r):
        perm[a] = b
        perm[b] = a

    return torch.tensor(
        perm,
        device=device,
        dtype=torch.long,
    )


def mixstyle_feature(
    feat,
    domains=None,
    p=0.5,
    alpha=0.1,
    cross_domain=False,
):
    if not feat.requires_grad:
        # This is only a training-time feature transform.
        pass

    if random.random() >= float(p):
        return feat

    if feat.ndim != 4 or feat.shape[0] < 2:
        return feat

    x = feat.float()

    mu = x.mean(
        dim=(2,3),
        keepdim=True,
    )

    var = x.var(
        dim=(2,3),
        keepdim=True,
        unbiased=False,
    )

    sig = torch.sqrt(
        var + 1e-6
    )

    x_norm = (
        x - mu
    ) / sig

    b = x.shape[0]

    if cross_domain:
        if domains is None:
            raise ValueError(
                "cross_domain MixStyle requires domains."
            )

        perm = _opposite_domain_permutation(
            domains,
            x.device,
        )

    else:
        perm = torch.randperm(
            b,
            device=x.device,
        )

    beta = torch.distributions.Beta(
        float(alpha),
        float(alpha),
    )

    lam = beta.sample(
        (b,1,1,1)
    ).to(
        device=x.device,
        dtype=x.dtype,
    )

    mu2 = mu.index_select(
        0,
        perm,
    )

    sig2 = sig.index_select(
        0,
        perm,
    )

    mu_mix = (
        lam * mu
        + (1.0 - lam) * mu2
    )

    sig_mix = (
        lam * sig
        + (1.0 - lam) * sig2
    )

    out = (
        x_norm * sig_mix
        + mu_mix
    )

    return out.to(feat.dtype)


def make_mixstyle_run_epoch(
    p=0.5,
    alpha=0.1,
    cross_domain=False,
):
    p = float(p)
    alpha = float(alpha)
    cross_domain = bool(cross_domain)

    def run_epoch(
        model, loader, criterion, optimizer, device, scaler, amp,
        train=True, max_batches=0, grad_clip=0.0,
        fg_aux_bce_weight=0.7, fg_aux_dice_weight=0.5,
        bg_bce_weight=0.0, aux_loss_weight=0.4,
    ):
        if not train:
            return sweep_domain_validation_epoch(
                model, loader, criterion, optimizer, device, scaler, amp,
                max_batches=max_batches,
                bg_bce_weight=bg_bce_weight,
            )

        for attr in ("backbone", "decoder", "head"):
            if not hasattr(model, attr):
                raise RuntimeError(
                    f"MixStyle experiment requires model.{attr}"
                )

        model.train(True)

        conf = torch.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=torch.int64,
        )

        tl = 0.0
        n = 0
        skip_i = 0
        skip_nf = 0

        autocast_ctx = base.make_autocast(
            device,
            amp,
        )

        pbar = tqdm(
            loader,
            desc="train",
            leave=False,
        )

        for step, batch in enumerate(
            pbar,
            1,
        ):
            x = batch["image"].to(
                device,
                non_blocking=True,
            )

            y = batch["mask"].to(
                device,
                non_blocking=True,
            )

            domains = batch["domain"]

            if not torch.any(
                y != IGNORE_INDEX
            ):
                skip_i += 1
                continue

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast_ctx():
                feats = list(
                    model.backbone(x)
                )

                # Decoder-level multi-scale MixStyle:
                # modify early MiT features before UPerNet fusion.
                feats[0] = mixstyle_feature(
                    feats[0],
                    domains,
                    p=p,
                    alpha=alpha,
                    cross_domain=cross_domain,
                )

                feats[1] = mixstyle_feature(
                    feats[1],
                    domains,
                    p=p,
                    alpha=alpha,
                    cross_domain=cross_domain,
                )

                dec = model.decoder(
                    feats
                )

                small = model.head(
                    dec
                )

                if hasattr(
                    model,
                    "_upsample_logits",
                ):
                    logits = model._upsample_logits(
                        small,
                        x,
                    )
                else:
                    logits = F.interpolate(
                        small,
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                loss = criterion(
                    logits,
                    y,
                )

            if not torch.isfinite(loss):
                skip_nf += 1
                optimizer.zero_grad(
                    set_to_none=True
                )
                continue

            if scaler is not None:
                scaler.scale(
                    loss
                ).backward()

                if grad_clip and grad_clip > 0:
                    scaler.unscale_(
                        optimizer
                    )
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )

                scaler.step(
                    optimizer
                )
                scaler.update()

            else:
                loss.backward()

                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )

                optimizer.step()

            update_confusion(
                conf,
                logits,
                y,
            )

            tl += float(
                loss.detach()
            )

            n += 1

            pbar.set_postfix(
                loss=f"{tl/max(n,1):.4f}"
            )

            if max_batches and step >= max_batches:
                break

        print(
            "MixStyle | "
            f"p={p:.2f} alpha={alpha:.2f} "
            f"cross_domain={int(cross_domain)} "
            f"stages=1+2"
        )

        return ret_dict(
            tl,
            tl,
            conf,
            n,
            skip_i,
            skip_nf,
        )

    return run_epoch


# ----------------------------------------------------------------------
# Cross-domain supervised ClassMix.
# ----------------------------------------------------------------------

def _random_class_mask(mask):
    valid = mask != IGNORE_INDEX
    classes = torch.unique(
        mask[valid]
    )

    if classes.numel() <= 1:
        return torch.zeros_like(
            mask,
            dtype=torch.bool,
        )

    classes = classes[
        torch.randperm(
            classes.numel(),
            device=classes.device,
        )
    ]

    k = max(
        1,
        int(classes.numel() // 2),
    )

    selected = classes[:k]

    out = torch.zeros_like(
        mask,
        dtype=torch.bool,
    )

    for c in selected:
        out |= (
            mask == c
        )

    return out


def cross_domain_classmix(
    images,
    masks,
    domains,
    p=0.5,
):
    if random.random() >= float(p):
        return images, masks

    u = [
        i for i, d in enumerate(domains)
        if canonical_domain(d) == "urban"
    ]

    r = [
        i for i, d in enumerate(domains)
        if canonical_domain(d) == "rural"
    ]

    if not u or not r:
        return images, masks

    random.shuffle(u)
    random.shuffle(r)

    x0 = images.clone()
    y0 = masks.clone()

    x = images.clone()
    y = masks.clone()

    for ui, ri in zip(u, r):
        mu = _random_class_mask(
            y0[ui]
        )

        mr = _random_class_mask(
            y0[ri]
        )

        # Urban-selected semantic regions pasted onto Rural context.
        x[ui] = torch.where(
            mu.unsqueeze(0),
            x0[ui],
            x0[ri],
        )

        y[ui] = torch.where(
            mu,
            y0[ui],
            y0[ri],
        )

        # Rural-selected semantic regions pasted onto Urban context.
        x[ri] = torch.where(
            mr.unsqueeze(0),
            x0[ri],
            x0[ui],
        )

        y[ri] = torch.where(
            mr,
            y0[ri],
            y0[ui],
        )

    return x, y


def make_classmix_run_epoch(p=0.5):
    p = float(p)

    def run_epoch(
        model, loader, criterion, optimizer, device, scaler, amp,
        train=True, max_batches=0, grad_clip=0.0,
        fg_aux_bce_weight=0.7, fg_aux_dice_weight=0.5,
        bg_bce_weight=0.0, aux_loss_weight=0.4,
    ):
        if not train:
            return sweep_domain_validation_epoch(
                model, loader, criterion, optimizer, device, scaler, amp,
                max_batches=max_batches,
                bg_bce_weight=bg_bce_weight,
            )

        model.train(True)

        conf = torch.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=torch.int64,
        )

        tl = 0.0
        n = 0
        skip_i = 0
        skip_nf = 0

        autocast_ctx = base.make_autocast(
            device,
            amp,
        )

        pbar = tqdm(
            loader,
            desc="train",
            leave=False,
        )

        for step, batch in enumerate(
            pbar,
            1,
        ):
            x = batch["image"].to(
                device,
                non_blocking=True,
            )

            y = batch["mask"].to(
                device,
                non_blocking=True,
            )

            domains = batch["domain"]

            x, y = cross_domain_classmix(
                x,
                y,
                domains,
                p=p,
            )

            if not torch.any(
                y != IGNORE_INDEX
            ):
                skip_i += 1
                continue

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast_ctx():
                logits = extract_logits(
                    model(x)
                )

                loss = criterion(
                    logits,
                    y,
                )

            if not torch.isfinite(loss):
                skip_nf += 1
                optimizer.zero_grad(
                    set_to_none=True
                )
                continue

            if scaler is not None:
                scaler.scale(
                    loss
                ).backward()

                if grad_clip and grad_clip > 0:
                    scaler.unscale_(
                        optimizer
                    )
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )

                scaler.step(
                    optimizer
                )
                scaler.update()

            else:
                loss.backward()

                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )

                optimizer.step()

            update_confusion(
                conf,
                logits,
                y,
            )

            tl += float(
                loss.detach()
            )

            n += 1

            pbar.set_postfix(
                loss=f"{tl/max(n,1):.4f}"
            )

            if max_batches and step >= max_batches:
                break

        print(
            f"CrossDomainClassMix | p={p:.2f}"
        )

        return ret_dict(
            tl,
            tl,
            conf,
            n,
            skip_i,
            skip_nf,
        )

    return run_epoch


def setup_run(
    run_name,
    *,
    balanced=False,
    fosmix_probability=None,
    fosmix_beta=None,
    train_run_epoch=None,
):
    """
    Shared setup for E072-E079.

    Returns the untouched original dataset class so the caller can restore it.
    """
    reset_sweep_state()

    prepare_run_dir(
        run_name
    )

    set_run_name(
        run_name
    )

    original_dataset = base.LoveDADataset

    if balanced:
        occurrence_dataset = make_occurrence_seed_dataset(
            original_dataset
        )

        working_dataset = occurrence_dataset

        install_sweep_balanced_sampler(
            batch_size=8
        )

    else:
        working_dataset = original_dataset

        install_tracking_sampler()

    if fosmix_probability is not None:
        if fosmix_beta is None:
            raise ValueError(
                "fosmix_beta must be supplied."
            )

        working_dataset = make_fosmix_dataset(
            working_dataset,
            probability=float(
                fosmix_probability
            ),
            beta=float(
                fosmix_beta
            ),
        )

    base.LoveDADataset = working_dataset

    base.run_epoch = (
        clean_train_run_epoch
        if train_run_epoch is None
        else train_run_epoch
    )

    install_compact_checkpointing(
        run_name
    )

    return original_dataset


def finish_run(
    run_name,
    method_name,
    original_dataset,
):
    base.LoveDADataset = original_dataset
    return finalize_compact_run(
        run_name,
        method_name,
    )
