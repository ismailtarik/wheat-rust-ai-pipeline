#!/usr/bin/env python3
"""Check image readability and export a report; never deletes files."""
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inventory',required=True); ap.add_argument('--output',default='data/reports/classification_v3_build/image_validation.csv'); args=ap.parse_args()
    df=pd.read_csv(args.inventory); rows=[]
    for _,r in df.iterrows():
        p=Path(str(r['path'])); rec={'path':str(p),'readable':False,'width':None,'height':None,'format':None,'error':''}
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                rec.update(readable=True,width=im.width,height=im.height,format=im.format or '')
        except Exception as e: rec['error']=f'{type(e).__name__}: {e}'
        rows.append(rec)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
    print(f'Valid images: {sum(x["readable"] for x in rows)}/{len(rows)}; report: {out.resolve()}')
if __name__=='__main__': main()
