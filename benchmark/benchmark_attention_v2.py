#!/usr/bin/env python3
"""Controlled E0-E4 attention benchmark for wheat classification_v3.

All attention modules are inserted after ResNet50.layer4 and before avgpool.
Preprocessing: EXIF transpose, RGB, aspect-preserving thumbnail, centered
128-gray padding to 224x224, ToTensor, ImageNet normalization.

Examples:
  python benchmark_attention.py --mode smoke
  python benchmark_attention.py --mode train --seeds 42
  python benchmark_attention.py --mode train --seeds 42,123,2024
  python benchmark_attention.py --mode evaluate --seeds 42
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tam_v2 import TAMv2, make_optimizer, diagnose_old_tam  # tam_v2.py doit être dans le même dossier

AUGMENT, DIFF_LR = False, False  # fixés par --augment / --diff-lr

IMAGE_SIZE, BATCH_SIZE, EPOCHS = 224, 32, 20
LEARNING_RATE, PATIENCE, NUM_WORKERS = 1e-4, 5, 2
VARIANTS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8")
NAMES = {"E0":"ResNet50", "E1":"ResNet50_SE", "E2":"ResNet50_CBAM",
         "E3":"ResNet50_TripletAttention", "E4":"ResNet50_TAM_Gabor",
         "E5":"ResNet50_TAMv2_after_layer4", "E6":"ResNet50_TAMv2_after_layer3",
         "E7":"ResNet50_TAMv2_learned_after_layer4", "E8":"ResNet50_TAMv2_nofilter_after_layer4"}
EXPECTED_CLASSES = sorted([
    "Aphid", "Black_Point", "Black_Rust", "Brown_Rust", "Common_Root_Rot",
    "Fusarium_Foot_Rot", "Fusarium_Head_Blight", "Healthy", "Leaf_Blight",
    "Mite", "Powdery_Mildew", "Septoria", "Smut", "Stem_Fly", "Tan_Spot",
    "Wheat_Blast", "Yellow_Rust"
])

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def resize_with_padding(image, size=IMAGE_SIZE):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
    return canvas

class WheatDataset(Dataset):
    def __init__(self, df, class_to_idx, transform):
        self.df = df.reset_index(drop=True).copy()
        self.class_to_idx, self.transform = class_to_idx, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(str(row["v3_destination_path"])) as im:
            im = resize_with_padding(im)
        return self.transform(im), self.class_to_idx[str(row["canonical_class"])]

class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        h=max(c//r,1)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Conv2d(c,h,1),nn.ReLU(inplace=True),nn.Conv2d(h,c,1),nn.Sigmoid())
    def forward(self,x): return x*self.fc(self.pool(x))

class ChannelAttention(nn.Module):
    def __init__(self,c,r=16):
        super().__init__(); h=max(c//r,1)
        self.mlp=nn.Sequential(nn.Conv2d(c,h,1,bias=False),nn.ReLU(inplace=True),nn.Conv2d(h,c,1,bias=False))
    def forward(self,x):
        return x*torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x,1))+self.mlp(F.adaptive_max_pool2d(x,1)))

class SpatialAttention(nn.Module):
    def __init__(self,k=7):
        super().__init__(); self.conv=nn.Conv2d(2,1,k,padding=k//2,bias=False)
    def forward(self,x):
        return x*torch.sigmoid(self.conv(torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True).values],1)))

class CBAM(nn.Module):
    def __init__(self,c): super().__init__(); self.ca=ChannelAttention(c); self.sa=SpatialAttention(7)
    def forward(self,x): return self.sa(self.ca(x))

class TripletGate(nn.Module):
    def __init__(self):
        super().__init__(); self.conv=nn.Conv2d(2,1,7,padding=3,bias=False); self.bn=nn.BatchNorm2d(1)
    def forward(self,x):
        z=torch.cat([x.max(1,keepdim=True).values,x.mean(1,keepdim=True)],1)
        return x*torch.sigmoid(self.bn(self.conv(z)))

class TripletAttention(nn.Module):
    def __init__(self): super().__init__(); self.cw=TripletGate(); self.hc=TripletGate(); self.hw=TripletGate()
    def forward(self,x):
        a=self.cw(x.permute(0,2,1,3).contiguous()).permute(0,2,1,3).contiguous()
        b=self.hc(x.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous()
        c=self.hw(x)
        return (a+b+c)/3.0

def make_gabor_bank(k=9,sigma=2.0,wavelength=4.0,gamma=0.5):
    if k%2==0: raise ValueError("Gabor kernel size must be odd")
    coords=torch.arange(k,dtype=torch.float32)-(k-1)/2
    yy,xx=torch.meshgrid(coords,coords,indexing="ij")
    kernels=[]
    for theta in np.linspace(0,np.pi,8,endpoint=False):
        xr=xx*np.cos(theta)+yy*np.sin(theta)
        yr=-xx*np.sin(theta)+yy*np.cos(theta)
        g=torch.exp(-(xr**2+gamma**2*yr**2)/(2*sigma**2))
        kernel=g*torch.cos(2*np.pi*xr/wavelength)
        kernel=kernel-kernel.mean()
        kernels.append(kernel/(kernel.abs().sum()+1e-8))
    return torch.stack(kernels).unsqueeze(1)

class TAM(nn.Module):
    """Benchmark TAM definition: channel-mean map, 8 fixed Gabor orientations,
    mean absolute texture response, learned spatial sigmoid gate, residual scaling."""
    def __init__(self,c):
        super().__init__()
        self.register_buffer("gabor_bank",make_gabor_bank())
        self.proj=nn.Conv2d(1,1,1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
    def forward(self,x):
        texture=x.mean(1,keepdim=True)
        responses=F.conv2d(texture,self.gabor_bank,padding=self.gabor_bank.shape[-1]//2)
        strength=responses.abs().mean(1,keepdim=True)
        gate=torch.sigmoid(self.proj(strength))
        return x*(1.0+gate)

class AttentionResNet50(nn.Module):
    """Same ResNet50 backbone/head; variant module is placed after layer4."""
    def __init__(self,n_classes,variant,pretrained=True):
        super().__init__()
        if variant not in VARIANTS: raise ValueError(f"Unknown variant: {variant}")
        backbone=resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        for name in ("conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4","avgpool"):
            setattr(self,name,getattr(backbone,name))
        self.attention={"E0":nn.Identity(),"E1":SE(2048),"E2":CBAM(2048),
                        "E3":TripletAttention(),"E4":TAM(2048)}.get(variant,nn.Identity())
        self.fc=nn.Linear(backbone.fc.in_features,n_classes)
        # nouveaux modules créés APRES fc pour ne pas changer l'init aléatoire de E0-E4
        self.tam3=nn.Identity()
        if variant=="E5": self.attention=TAMv2(2048,k=3)   # carte 7x7 -> noyau 3x3
        if variant=="E6": self.tam3=TAMv2(1024,k=5)        # carte 14x14 -> noyau 5x5
        if variant=="E7": self.attention=TAMv2(2048,k=3,mode="learned")  # ablation : filtres appris
        if variant=="E8": self.attention=TAMv2(2048,k=3,mode="none")     # ablation : sans filtrage spatial
        self.variant=variant
    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.tam3(x); x=self.layer4(x)
        x=self.attention(x) # layer4 -> attention -> global average pooling
        x=self.avgpool(x).flatten(1)
        return self.fc(x)

def load_data(base_dir):
    path=base_dir/"classification_manifest.csv"
    if not path.is_file(): raise FileNotFoundError(f"Manifest introuvable: {path}")
    df=pd.read_csv(path)
    required={"canonical_class","new_split","v3_destination_path"}
    if not required.issubset(df.columns): raise ValueError(f"Colonnes manquantes: {required-set(df.columns)}")
    classes=sorted(df["canonical_class"].dropna().unique().tolist())
    if classes!=EXPECTED_CLASSES: raise ValueError(f"Mapping/classes inattendus: {classes}")
    class_to_idx={c:i for i,c in enumerate(classes)}
    expected={"train":8761,"val":1859,"test":1864}; parts={}
    for split,n in expected.items():
        part=df[df.new_split.astype(str)==split].copy().reset_index(drop=True)
        if len(part)!=n: raise ValueError(f"{split}: {len(part)} images, {n} attendues")
        bad=[p for p in part.v3_destination_path.astype(str) if not Path(p).is_file()]
        if bad: raise FileNotFoundError(f"{len(bad)} chemins invalides; exemple: {bad[0]}")
        parts[split]=part
    norm=[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
    tfm=transforms.Compose(norm)
    train_tfm=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15,fill=(128,128,128)),transforms.ColorJitter(.1,.1,.1)]+norm) if AUGMENT else tfm
    return {s:WheatDataset(d,class_to_idx,train_tfm if s=="train" else tfm) for s,d in parts.items()},class_to_idx,classes

def make_loaders(ds,seed):
    gen=torch.Generator().manual_seed(seed); pin=torch.cuda.is_available()
    return {
      "train":DataLoader(ds["train"],batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=pin,generator=gen),
      "val":DataLoader(ds["val"],batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=pin),
      "test":DataLoader(ds["test"],batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=pin)
    }

def metrics(y,p):
    pr,re,_,_=precision_recall_fscore_support(y,p,average="macro",zero_division=0)
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_precision":float(pr),"macro_recall":float(re),
            "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
            "weighted_f1":float(f1_score(y,p,average="weighted",zero_division=0))}

def epoch_run(model,loader,criterion,device,optimizer=None):
    train=optimizer is not None; model.train(train); loss_sum=0.; y=[]; p=[]
    for x,t in loader:
        x=x.to(device,non_blocking=True); t=t.to(device,non_blocking=True)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits=model(x); loss=criterion(logits,t)
            if train: loss.backward(); optimizer.step()
        loss_sum+=loss.item()*x.size(0); y.extend(t.detach().cpu().tolist()); p.extend(logits.argmax(1).detach().cpu().tolist())
    out=metrics(y,p); out["loss"]=loss_sum/len(loader.dataset); return out

def evaluate(model,loader,device,classes,outdir):
    model.eval(); y=[]; p=[]; conf=[]
    with torch.inference_mode():
        for x,t in loader:
            prob=torch.softmax(model(x.to(device,non_blocking=True)),1); c,q=prob.max(1)
            y.extend(t.tolist()); p.extend(q.cpu().tolist()); conf.extend(c.cpu().tolist())
    result=metrics(y,p); result.update(test_samples=len(y),errors=int(np.sum(np.array(y)!=np.array(p))),correct=int(np.sum(np.array(y)==np.array(p))))
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"test_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    report=classification_report(y,p,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(outdir/"classification_report.csv",encoding="utf-8-sig")
    pd.DataFrame(confusion_matrix(y,p,labels=list(range(len(classes)))),index=classes,columns=classes).to_csv(outdir/"confusion_matrix.csv")
    pd.DataFrame({"true_label":[classes[i] for i in y],"predicted_label":[classes[i] for i in p],"confidence":conf,"correct":np.array(y)==np.array(p)}).to_csv(outdir/"test_predictions.csv",index=False,encoding="utf-8-sig")
    return result

def smoke_test(ds,classes,device):
    loader=DataLoader(ds["train"],batch_size=2,shuffle=False,num_workers=0)
    x,y=next(iter(loader)); x=x.to(device); y=y.to(device); results=[]
    for variant in VARIANTS:
        seed_all(42); model=AttentionResNet50(len(classes),variant,pretrained=False).to(device)
        model.train(); logits=model(x)
        if tuple(logits.shape)!=(2,len(classes)): raise RuntimeError(f"{variant}: bad output shape {tuple(logits.shape)}")
        loss=nn.CrossEntropyLoss()(logits,y); loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not torch.isfinite(logits).all() or not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"{variant}: non-finite output/loss/gradient")
        row={"variant":variant,"architecture":NAMES[variant],"status":"PASS","output_shape":list(logits.shape),
             "loss":float(loss.detach().cpu()),"parameters":sum(p.numel() for p in model.parameters()),
             "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
             "gradient_tensors":len(grads)}
        results.append(row); print(f"{variant}: PASS | shape={row['output_shape']} | params={row['parameters']:,} | loss={row['loss']:.4f}")
        del model,logits,loss
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results

def train_one(variant,seed,ds,class_to_idx,classes,outroot,device):
    seed_all(seed); loaders=make_loaders(ds,seed)
    run=outroot/NAMES[variant]/f"seed_{seed}"; run.mkdir(parents=True,exist_ok=True)
    model=AttentionResNet50(len(classes),variant,pretrained=True).to(device)
    criterion=nn.CrossEntropyLoss(); optimizer=make_optimizer(model) if DIFF_LR else Adam(model.parameters(),lr=LEARNING_RATE)
    scheduler=ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=2)
    best=-1.; best_epoch=0; waiting=0; history=[]; start=time.time(); ckpt=run/"best_model.pth"
    for e in range(1,EPOCHS+1):
        t=time.time(); tr=epoch_run(model,loaders["train"],criterion,device,optimizer)
        va=epoch_run(model,loaders["val"],criterion,device); scheduler.step(va["macro_f1"])
        history.append({"epoch":e,**{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()},"lr":optimizer.param_groups[0]["lr"]})
        print(f"[{variant} seed={seed}] epoch {e:02d}: train F1={tr['macro_f1']:.4f}, val F1={va['macro_f1']:.4f}, val acc={va['accuracy']:.4f}, {time.time()-t:.1f}s")
        if va["macro_f1"]>best:
            best,best_epoch,waiting=va["macro_f1"],e,0
            torch.save({"variant":variant,"epoch":e,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                        "best_val_macro_f1":best,"class_to_idx":class_to_idx,"image_size":IMAGE_SIZE,"seed":seed,
                        "attention_placement":"after layer4 before avgpool","preprocessing":"aspect-preserving thumbnail + centered gray padding + ImageNet normalization"},ckpt)
        else: waiting+=1
        if waiting>=PATIENCE: print(f"Early stopping at epoch {e}"); break
    pd.DataFrame(history).to_csv(run/"training_history.csv",index=False)
    summary={"variant":variant,"architecture":NAMES[variant],"seed":seed,"best_epoch":best_epoch,"best_validation_macro_f1":best,
             "epochs_completed":len(history),"training_seconds":time.time()-start,"device":str(device),"batch_size":BATCH_SIZE,
             "image_size":IMAGE_SIZE,"learning_rate":LEARNING_RATE,"optimizer":"Adam","loss":"CrossEntropyLoss",
             "augment":AUGMENT,"diff_lr":DIFF_LR,"selection_metric":"validation macro-F1","pretrained_weights":"ImageNet"}
    (run/"training_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    saved=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(saved["model_state_dict"])
    test=evaluate(model,loaders["test"],device,classes,run/"test_evaluation")
    summary.update({f"test_{k}":v for k,v in test.items()})
    (run/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"[{variant} seed={seed}] TEST accuracy={test['accuracy']:.4f}; macro-F1={test['macro_f1']:.4f}")
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",choices=["smoke","train","evaluate","diagnose"],default="smoke")
    ap.add_argument("--base-dir",type=Path,default=Path("/content/wheat-rust-ai-pipeline/data/processed/classification_v3"))
    ap.add_argument("--output-dir",type=Path,default=None)
    ap.add_argument("--seeds",default="42",help="e.g. 42 or 42,123,2024")
    ap.add_argument("--variants",default="E0,E1,E2,E3,E4")
    ap.add_argument("--augment",action="store_true",help="augmentation de données (train uniquement)")
    ap.add_argument("--diff-lr",action="store_true",help="LR x10 pour modules neufs + fc, AdamW")
    args=ap.parse_args(); global AUGMENT,DIFF_LR; AUGMENT,DIFF_LR=args.augment,args.diff_lr; out=args.output_dir or args.base_dir/"experiments"/"attention_benchmark"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}\nDataset={args.base_dir}\nOutput={out}")
    ds,class_to_idx,classes=load_data(args.base_dir)
    print(f"Splits: train={len(ds['train'])}, val={len(ds['val'])}, test={len(ds['test'])}; classes={len(classes)}")
    if args.mode=="smoke":
        result=smoke_test(ds,classes,device); out.mkdir(parents=True,exist_ok=True)
        (out/"smoke_test_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(f"Smoke test saved: {out/'smoke_test_results.json'}"); return
    variants=[v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants or set(variants)-set(VARIANTS): raise ValueError(f"Variants must be in {VARIANTS}")
    seeds=[int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds: raise ValueError("At least one seed is required")
    if args.mode=="diagnose":
        seed=seeds[0]; loaders=make_loaders(ds,seed)
        ckpt=out/NAMES["E4"]/f"seed_{seed}"/"best_model.pth"
        if not ckpt.is_file(): raise FileNotFoundError(ckpt)
        saved=torch.load(ckpt,map_location=device,weights_only=False)
        model=AttentionResNet50(len(classes),"E4",pretrained=False).to(device)
        model.load_state_dict(saved["model_state_dict"],strict=True)
        diagnose_old_tam(model,loaders["val"],device); return
    if args.mode=="train":
        rows=[]
        for seed in seeds:
            for variant in variants: rows.append(train_one(variant,seed,ds,class_to_idx,classes,out,device))
        out.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(rows)
        df.to_csv(out/"benchmark_results_by_run.csv",index=False)
        metric_cols=[f"test_{m}" for m in ["accuracy","balanced_accuracy","macro_precision","macro_recall","macro_f1","weighted_f1"]]
        df.groupby("architecture")[metric_cols].agg(["mean","std"]).to_csv(out/"benchmark_results_mean_std.csv")
        print(f"Benchmark terminé. Résultats dans {out}")
        return
    loaders_cache={}
    rows=[]
    for seed in seeds:
        loaders=make_loaders(ds,seed)
        for variant in variants:
            run=out/NAMES[variant]/f"seed_{seed}"; ckpt=run/"best_model.pth"
            if not ckpt.is_file(): print(f"SKIP checkpoint absent: {ckpt}"); continue
            saved=torch.load(ckpt,map_location=device,weights_only=False)
            if saved["class_to_idx"]!=class_to_idx: raise ValueError(f"Class mapping mismatch: {ckpt}")
            model=AttentionResNet50(len(classes),variant,pretrained=False).to(device)
            model.load_state_dict(saved["model_state_dict"],strict=True)
            m=evaluate(model,loaders["test"],device,classes,run/"test_evaluation")
            rows.append({"variant":variant,"architecture":NAMES[variant],"seed":seed,**m})
    if rows: pd.DataFrame(rows).to_csv(out/"reevaluated_test_results.csv",index=False)
    else: print("Aucun checkpoint disponible.")

if __name__=="__main__": main()