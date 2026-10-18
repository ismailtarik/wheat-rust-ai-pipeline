#!/usr/bin/env python3
"""Validate physical files and manifest for a classification_v3 folder."""
import argparse, hashlib
from pathlib import Path
import pandas as pd
EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--manifest',default=''); ap.add_argument('--expect-12484',action='store_true'); a=ap.parse_args()
 root=Path(a.root).resolve(); mf=Path(a.manifest).resolve() if a.manifest else root/'classification_manifest.csv'
 if not root.is_dir(): raise FileNotFoundError(root)
 if not mf.is_file(): raise FileNotFoundError(mf)
 df=pd.read_csv(mf,dtype=str).fillna(''); required={'sha256','canonical_class','new_split'}
 if not required.issubset(df.columns): raise ValueError(f'Colonnes manquantes: {required-set(df.columns)}')
 files=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTS]
 missing=[]; wrong=[]
 for _,r in df.iterrows():
  if 'rebuilt_path' in df and str(r.get('rebuilt_path','')).strip(): p=Path(r['rebuilt_path'])
  elif str(r.get('v3_destination_path','')).strip(): p=Path(r['v3_destination_path'])
  else: p=root/str(r['new_split'])/str(r['canonical_class'])/f"{str(r['sha256']).lower()}.jpg"
  if not p.is_absolute(): p=root/p
  if not p.is_file(): missing.append(str(p)); continue
  if sha256(p)!=str(r['sha256']).lower(): wrong.append(str(p))
 print('Manifest rows:',len(df),'| files on disk:',len(files),'| classes:',df.canonical_class.nunique())
 print('Splits:',df.new_split.value_counts().to_dict(),'| SHA unique:',df.sha256.str.lower().nunique())
 print('Missing:',len(missing),'| hash errors:',len(wrong))
 if missing: print('Missing examples:',missing[:5])
 if wrong: print('Hash error examples:',wrong[:5])
 assert not missing and not wrong
 assert df.sha256.str.lower().is_unique
 if a.expect_12484:
  assert len(df)==12484 and len(files)==12484 and df.canonical_class.nunique()==17
  assert df.new_split.value_counts().to_dict()=={'train':8761,'val':1859,'test':1864}
 print('VALIDATION OK')
if __name__=='__main__': main()
