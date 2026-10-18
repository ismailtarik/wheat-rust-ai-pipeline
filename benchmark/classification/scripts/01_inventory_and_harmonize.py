#!/usr/bin/env python3
"""Inventory raw images and apply the notebook's explicit canonical label map.
Does not copy, delete, split, or modify images.
"""
import argparse, hashlib
from pathlib import Path
import pandas as pd

EXTS = {'.jpg','.jpeg','.png','.bmp','.webp'}
LABELS = {
    'Aphid':'Aphid','Black Rust':'Black_Rust','Blast':'Wheat_Blast',
    'Brown Rust':'Brown_Rust','Common Root Rot':'Common_Root_Rot',
    'Fusarium Head Blight':'Fusarium_Head_Blight','Healthy':'Healthy',
    'Leaf Blight':'Leaf_Blight','Mildew':'Powdery_Mildew','Mite':'Mite',
    'Septoria':'Septoria','Smut':'Smut','Stem fly':'Stem_Fly','Tan spot':'Tan_Spot',
    'Yellow Rust':'Yellow_Rust','BlackPoint':'Black_Point',
    'FusariumFootRot':'Fusarium_Foot_Rot','HealthyLeaf':'Healthy',
    'LeafBlight':'Leaf_Blight','WheatBlast':'Wheat_Blast','BrownRust':'Brown_Rust',
    'YellowRust':'Yellow_Rust'
}
def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-root',required=True,help='Racine contenant les datasets bruts')
    ap.add_argument('--output',default='data/reports/classification_v3_build/raw_inventory.csv')
    args=ap.parse_args(); root=Path(args.raw_root).resolve()
    if not root.is_dir(): raise FileNotFoundError(root)
    rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in EXTS: continue
        rel=p.relative_to(root); parts=[x.lower() for x in rel.parts]
        split='unassigned'
        if any(x in ('train','training') for x in parts): split='train'
        elif any(x in ('valid','validation','val') for x in parts): split='val'
        elif any(x in ('test','testing') for x in parts): split='test'
        original=p.parent.name
        rows.append({'dataset':rel.parts[0] if len(rel.parts)>1 else root.name,
                     'original_label':original,'canonical_class':LABELS.get(original,original),
                     'partition_hint':split,'filename':p.name,'path':str(p),
                     'sha256':sha256(p),'bytes':p.stat().st_size})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(out,index=False)
    print(f'Images inventoriées: {len(df)}; labels canoniques observés: {df.canonical_class.nunique() if len(df) else 0}')
    print(f'Labels sans mapping explicite: {sorted(set(df.loc[df.original_label.map(lambda x:x not in LABELS),"original_label"])) if len(df) else []}')
    print(f'CSV: {out.resolve()}')
if __name__=='__main__': main()
