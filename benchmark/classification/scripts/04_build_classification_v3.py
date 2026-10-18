#!/usr/bin/env python3
"""Rebuild classification_v3 from the FINAL APPROVED manifest produced by notebook cells 92-97.
This deliberately does not invent duplicate-review decisions or re-randomize splits.
Dry-run is default. Use --execute only after reviewing the manifest and paths.
"""
import argparse, hashlib, shutil
from pathlib import Path
import pandas as pd
EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def first_existing(row, cols):
    for col in cols:
        if col in row and str(row[col]).strip() and str(row[col]).lower()!='nan':
            p=Path(str(row[col]))
            if p.is_file(): return p
    return None
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--approved-manifest',required=True,help='CSV final avec sha256, canonical_class, new_split et source path')
    ap.add_argument('--output-dir',default='data/processed/classification_v3_rebuilt')
    ap.add_argument('--execute',action='store_true',help='Autorise réellement les copies; sinon dry-run')
    ap.add_argument('--allow-existing',action='store_true',help='Autorise un dossier de sortie déjà existant, sans le supprimer')
    args=ap.parse_args(); mf=Path(args.approved_manifest).resolve(); out=Path(args.output_dir).resolve()
    if not mf.is_file(): raise FileNotFoundError(mf)
    df=pd.read_csv(mf,dtype=str).fillna('')
    required={'sha256','canonical_class','new_split'}
    missing=required-set(df.columns)
    if missing: raise ValueError(f'Colonnes obligatoires absentes: {sorted(missing)}')
    if df.sha256.str.lower().duplicated().any(): raise ValueError('SHA-256 non uniques dans le manifeste approuvé.')
    df['new_split']=df.new_split.str.lower().replace({'validation':'val','valid':'val','training':'train','testing':'test'})
    if not set(df.new_split).issubset({'train','val','test'}): raise ValueError('new_split doit être train/val/test.')
    if df[['canonical_class','sha256']].eq('').any().any(): raise ValueError('Classe ou SHA vide.')
    if out.exists() and any(out.iterdir()) and not args.allow_existing: raise FileExistsError(f'{out} existe et n’est pas vide. Choisir un nouveau dossier ou --allow-existing.')
    candidates=['destination_path_v2','destination_path','image_path','path','source']
    missing_sources=[]; plan=[]
    for _,r in df.iterrows():
        src=first_existing(r,candidates)
        if src is None: missing_sources.append(str(r.get('filename',r['sha256']))) ; continue
        expected=str(r.sha256).lower(); actual=sha256(src)
        if actual!=expected: raise ValueError(f'SHA mismatch source {src}: expected {expected}, got {actual}')
        ext=src.suffix.lower()
        if ext not in EXTS: raise ValueError(f'Extension non prise en charge: {src}')
        dst=out/r.new_split/r.canonical_class/f'{expected}{ext}'
        plan.append((src,dst,expected))
    if missing_sources: raise FileNotFoundError(f'{len(missing_sources)} sources introuvables. Exemples: {missing_sources[:10]}')
    print(f'Manifeste approuvé: {len(df)} lignes, {df.canonical_class.nunique()} classes')
    print('Répartition:',df.new_split.value_counts().to_dict())
    print('Destination:',out)
    if not args.execute:
        print('DRY-RUN: aucune image copiée. Vérifiez les chemins puis relancez avec --execute.')
        return
    out.mkdir(parents=True,exist_ok=True)
    for src,dst,expected in plan:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha256(dst)==expected: continue
            raise FileExistsError(f'Destination déjà existante avec contenu différent: {dst}')
        shutil.copy2(src,dst)
        if sha256(dst)!=expected: raise IOError(f'Vérification SHA après copie échouée: {dst}')
    # Save relative destinations so the output remains portable.
    df['rebuilt_path']=[str(out/split/cls/(sha+Path(first_existing(r,candidates)).suffix.lower())) for _,r in df.iterrows() for split,cls,sha in [(r['new_split'],r['canonical_class'],r['sha256'].lower())]]
    df.to_csv(out/'classification_manifest.csv',index=False)
    print(f'Copie vérifiée: {len(plan)} fichiers. Manifeste: {out/"classification_manifest.csv"}')
if __name__=='__main__': main()
