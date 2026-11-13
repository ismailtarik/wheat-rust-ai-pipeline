#!/usr/bin/env python3
"""Multi-architecture, multi-seed benchmark runner (reviewer revision R5).

Same protocol as the ResNet50 baseline in train_eval.py (224x224 resize-with-padding,
ImageNet normalisation, Adam lr=1e-4, batch 32, ReduceLROnPlateau on val macro-F1,
best checkpoint by val macro-F1), except: up to 50 epochs + early stopping, so
training converges. All architectures share exactly the same pipeline.

Architectures (torchvision, ImageNet-pretrained, 224x224):
    resnet50 | efficientnet_b0 | convnext_tiny | vit_b_16

Examples (from benchmark/):
    # one architecture, 3 seeds
    python train_multi_arch.py --arch efficientnet_b0 --seeds 42 43 44 \
        --manifest ../data/processed/classification_v4_group/classification_manifest_group.csv \
        --out ../outputs/benchmark_v4

    # all four, 3 seeds
    for a in resnet50 efficientnet_b0 convnext_tiny vit_b_16; do
      python train_multi_arch.py --arch $a --seeds 42 43 44 --manifest <M> --out <O>
    done

    # leave-one-source-out: test on every image of source "mendeley", train/val on the rest
    python train_multi_arch.py --arch resnet50 --seeds 42 --holdout-source mendeley \
        --source-col source_dataset --manifest <M> --out <O>

    # then aggregate: mean +- std over seeds
    python train_multi_arch.py --aggregate --out <O>
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

ARCHS = ["resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16"]
IMAGE_SIZE, BATCH_SIZE = 224, 32
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
def build_model(arch, num_classes, pretrained=True):
    import torch.nn as nn
    from torchvision import models as m

    if arch == "resnet50":
        net = m.resnet50(weights=m.ResNet50_Weights.DEFAULT if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        net = m.efficientnet_b0(weights=m.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    elif arch == "convnext_tiny":
        net = m.convnext_tiny(weights=m.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
    elif arch == "vit_b_16":
        net = m.vit_b_16(weights=m.ViT_B_16_Weights.DEFAULT if pretrained else None)
        net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
    else:
        raise ValueError(arch)
    return net


def resize_with_padding(img, size=IMAGE_SIZE):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def make_dataset(df, path_col):
    import torch
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            r = df.iloc[i]
            with Image.open(r[path_col]) as im:
                x = tf(resize_with_padding(im))
            return x, int(r["label_idx"])

    return DS()


def metrics(y, p, n_cls):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_recall_fscore_support)
    labels = list(range(n_cls))
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
        "per_class_f1": [float(v) for v in f1],
        "per_class_support": [int(v) for v in sup],
    }


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    import torch
    train = optimizer is not None
    model.train(train)
    ys, ps, tot, n = [], [], 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
            ys.append(y.cpu().numpy())
            ps.append(out.argmax(1).cpu().numpy())
    return tot / n, np.concatenate(ys), np.concatenate(ps)


def train_one(a, seed, df, classes, run_dir):
    import torch
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, te = (df[df["_split"] == s] for s in ("train", "val", "test"))
    assert len(tr) and len(va) and len(te), "empty partition"
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=BATCH_SIZE, num_workers=a.workers, pin_memory=True)
    ltr = DataLoader(make_dataset(tr, a.path_col), shuffle=True, generator=g, **kw)
    lva = DataLoader(make_dataset(va, a.path_col), shuffle=False, **kw)
    lte = DataLoader(make_dataset(te, a.path_col), shuffle=False, **kw)

    model = build_model(a.arch, len(classes), pretrained=not a.no_pretrained).to(device)
    crit = torch.nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=a.lr)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1, best_ep, bad, hist = -1.0, 0, 0, []
    ckpt = run_dir / "best.pt"
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        tl, ty, tp = run_epoch(model, ltr, device, crit, opt, scaler)
        vl, vy, vp = run_epoch(model, lva, device, crit)
        tm, vm = metrics(ty, tp, len(classes)), metrics(vy, vp, len(classes))
        sched.step(vm["macro_f1"])
        hist.append(dict(epoch=ep, train_loss=tl, train_acc=tm["accuracy"], val_loss=vl,
                         val_acc=vm["accuracy"], val_macro_f1=vm["macro_f1"], lr=opt.param_groups[0]["lr"]))
        print(f"[{a.arch} s{seed}] ep {ep:02d} train_acc {tm['accuracy']:.4f} "
              f"val_acc {vm['accuracy']:.4f} val_F1 {vm['macro_f1']:.4f}", flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, best_ep, bad = vm["macro_f1"], ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, y, p = run_epoch(model, lte, device, crit)
    res = metrics(y, p, len(classes))
    res.update(arch=a.arch, seed=seed, best_epoch=best_ep, epochs_run=len(hist),
               converged_before_max=bool(best_ep < a.epochs), best_val_macro_f1=best_f1,
               train_time_s=round(time.time() - t0, 1), n_train=len(tr), n_val=len(va),
               n_test=len(te), holdout_source=a.holdout_source, classes=classes,
               split_file=str(a.manifest))

    # per-source breakdown on the test set (if a source column exists)
    if a.source_col and a.source_col in df.columns:
        te_src = te[a.source_col].astype(str).values
        res["per_source"] = {
            s: {"n": int((te_src == s).sum()),
                "accuracy": float((y[te_src == s] == p[te_src == s]).mean())}
            for s in sorted(set(te_src))}

    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    out = te[[a.path_col]].copy()
    out["y_true"], out["y_pred"] = y, p
    out.to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


# ----------------------------------------------------------------------------
def aggregate(out):
    rows = []
    for f in Path(out).rglob("metrics.json"):
        r = json.loads(f.read_text())
        rows.append(dict(arch=r["arch"], holdout=r.get("holdout_source") or "-", seed=r["seed"],
                         accuracy=r["accuracy"], balanced_accuracy=r["balanced_accuracy"],
                         macro_f1=r["macro_f1"], best_epoch=r["best_epoch"],
                         converged_before_max=r["converged_before_max"]))
    d = pd.DataFrame(rows)
    assert len(d), f"no metrics.json under {out}"
    g = d.groupby(["arch", "holdout"])
    summ = g[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"]).round(4)
    summ["n_seeds"] = g.size()
    d.to_csv(Path(out) / "all_runs.csv", index=False)
    summ.to_csv(Path(out) / "summary_mean_std.csv")
    print(summ.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-col", default="new_split", help="column with train/val/test")
    ap.add_argument("--path-col", default="v3_destination_path")
    ap.add_argument("--class-col", default="canonical_class")
    ap.add_argument("--source-col", default=None, help="source/provenance column (per-source + LOSO)")
    ap.add_argument("--holdout-source", default=None,
                    help="LOSO: all images of this source form the test set; train/val keep the rest")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()

    if a.aggregate:
        return aggregate(a.out)
    assert a.arch and a.manifest, "--arch and --manifest are required"

    df = pd.read_csv(a.manifest)
    for c in (a.split_col, a.path_col, a.class_col):
        assert c in df.columns, f"missing column {c!r}"
    df["_split"] = df[a.split_col].astype(str).str.lower().replace({"validation": "val"})
    classes = sorted(df[a.class_col].unique())
    df["label_idx"] = df[a.class_col].map({c: i for i, c in enumerate(classes)})

    tag = a.arch
    if a.holdout_source:
        assert a.source_col in df.columns, "--source-col required for --holdout-source"
        is_h = df[a.source_col].astype(str) == a.holdout_source
        assert is_h.any(), f"source {a.holdout_source!r} not found"
        df.loc[is_h, "_split"] = "test"
        df.loc[~is_h & (df["_split"] == "test"), "_split"] = "train"   # keep all non-held-out data
        tag = f"{a.arch}/loso_{a.holdout_source}"
        missing = set(df.loc[is_h, a.class_col]) - set(df.loc[~is_h, a.class_col])
        if missing:
            print("WARNING: classes only present in the held-out source:", sorted(missing))

    for seed in a.seeds:
        run_dir = Path(a.out) / tag / f"seed{seed}"
        if (run_dir / "metrics.json").exists():
            print("skip (done):", run_dir)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        r = train_one(a, seed, df, classes, run_dir)
        print(f"==> {a.arch} seed {seed}: acc {r['accuracy']:.4f} bal {r['balanced_accuracy']:.4f} "
              f"F1 {r['macro_f1']:.4f} (best epoch {r['best_epoch']}/{a.epochs})")


if __name__ == "__main__":
    main()
