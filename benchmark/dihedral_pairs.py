#!/usr/bin/env python3
"""Find near-duplicates that plain pHash misses: flipped / rotated copies (typical augmentations).

For each image, pHash is computed on its 8 dihedral variants (4 rotations x optional mirror).
Two images are linked if the minimum Hamming distance between image A and ANY variant of B
is <= --max-dist. Output = original pairs file + the new pairs (same columns), ready for
make_group_split.py --pairs.

Usage (from benchmark/):
    python dihedral_pairs.py \
      --manifest ../data/processed/classification_v3/classification_manifest.csv \
      --pairs    ../data/reports/near_duplicate_audit/near_duplicate_pairs.csv \
      --out      ../data/reports/near_duplicate_audit/near_duplicate_pairs_dihedral.csv \
      --path-col v3_destination_path --only-path-contains Mendeley   # or omit to scan all images
Cost: 8 hashes per image + an N x N x 8 distance matrix per class (fast for ~12k images).
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash(img):
    """Same algorithm as imagehash.phash (hash_size=8, highfreq_factor=4) -> uint64."""
    g = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    d = dctn(np.asarray(g, dtype=np.float64), type=2)[:8, :8]
    bits = (d > np.median(d)).flatten()
    return np.packbits(bits).view(">u8")[0]


def dihedral_hashes(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        out = []
        for flip in (False, True):
            base = ImageOps.mirror(im) if flip else im
            for k in range(4):
                out.append(phash(base.rotate(90 * k, expand=True)))
    return np.array(out, dtype=np.uint64)          # shape (8,)


def hamming(a, b):
    x = np.bitwise_xor(a, b).view(np.uint8).reshape(*np.broadcast(a, b).shape, 8)
    return POP[x].sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--path-col", default="v3_destination_path")
    ap.add_argument("--only-path-contains", default=None, help="restrict to paths containing this text")
    ap.add_argument("--max-dist", type=int, default=6)
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    if a.only_path_contains:
        df = df[df[a.path_col].astype(str).str.contains(a.only_path_contains, case=False)]
    df = df.reset_index(drop=True)
    print("images scanned:", len(df))

    H = np.stack([dihedral_hashes(p) for p in df[a.path_col]])        # (N, 8)
    ident = H[:, 0]                                                    # untouched orientation
    new = []
    # same-class comparison is where augmentations live; cross-class kept too (cheap, all-vs-all)
    for i in range(len(df)):
        d = hamming(ident[i + 1:, None], H[i][None, :]).min(1)         # min over 8 variants of i
        for j in np.nonzero(d <= a.max_dist)[0] + i + 1:
            new.append(dict(sha256_1=df.sha256[i], sha256_2=df.sha256[j],
                            class_1=df.canonical_class[i], class_2=df.canonical_class[j],
                            hamming_distance=int(d[j - i - 1]),
                            same_class=bool(df.canonical_class[i] == df.canonical_class[j]),
                            source="dihedral"))
    new = pd.DataFrame(new)
    old = pd.read_csv(a.pairs)
    known = set(zip(old.sha256_1, old.sha256_2)) | set(zip(old.sha256_2, old.sha256_1))
    if len(new):
        new = new[[(s1, s2) not in known for s1, s2 in zip(new.sha256_1, new.sha256_2)]]
    print(f"pairs already in pHash audit: {len(old)} | NEW pairs found only via flips/rotations: {len(new)}")
    pd.concat([old, new], ignore_index=True).to_csv(a.out, index=False)
    print("written:", a.out)


if __name__ == "__main__":
    main()
