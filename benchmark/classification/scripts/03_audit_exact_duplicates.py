#!/usr/bin/env python3
"""Report exact duplicates and cross-label/split conflicts using SHA-256."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inventory',required=True); ap.add_argument('--output-dir',default='data/reports/classification_v3_build'); args=ap.parse_args()
    df=pd.read_csv(args.inventory).fillna(''); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if 'sha256' not in df: raise ValueError('Colonne sha256 absente')
    dup=df[df.duplicated('sha256',keep=False)].sort_values('sha256')
    dup.to_csv(out/'exact_duplicate_groups.csv',index=False)
    summaries=[]
    for sha,g in dup.groupby('sha256'):
        summaries.append({'sha256':sha,'n_files':len(g),'labels':' | '.join(sorted(set(g.get('canonical_class',pd.Series(dtype=str)).astype(str)))),
                          'splits':' | '.join(sorted(set(g.get('partition_hint',pd.Series(dtype=str)).astype(str)))),
                          'paths':' | '.join(g['path'].astype(str))})
    summary=pd.DataFrame(summaries); summary.to_csv(out/'exact_duplicate_summary.csv',index=False)
    print(f'Images inventoriées: {len(df)}; lignes concernées par doublons exacts: {len(dup)}; groupes: {len(summary)}')
    print('Les conflits de classe/partition doivent être examinés; aucun fichier n’est supprimé.')
if __name__=='__main__': main()
