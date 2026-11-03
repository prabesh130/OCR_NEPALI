import argparse
import os

import pandas as pd
from PIL import Image

from src.dataset import OCRDataset, pad_resize
from src.utils import load_processor


def main(a):
    print("Loading processor...", flush=True)
    proc = load_processor()
    tok = proc.tokenizer
    size = proc.image_processor.size["height"]
    print("image size:", size, flush=True)

    df = pd.read_csv("data/splits/train.csv", dtype=str, keep_default_na=False)
    print("train rows:", len(df), flush=True)

    # 1. tokenizer coverage and label length
    unk = tok.unk_token_id
    lens, bad = [], []
    for t in df["text"]:
        ids = tok(t).input_ids
        lens.append(len(ids))
        if unk is not None and unk in ids:
            bad.append(t)
    print("max len:", max(lens), "| mean:", round(sum(lens) / len(lens), 1))
    print("labels with unknown tokens:", len(bad), "of", len(df))
    print("examples:", bad[:10])

    mism = [
        t for t in df["text"][:500]
        if tok.decode(tok(t).input_ids, skip_special_tokens=True).strip() != t
    ]
    print("round-trip mismatches (first 500):", len(mism), mism[:5])

    # 2. image sizes
    sizes = [Image.open(f"{a.root}/{p}").size for p in df["image_path"][:300]]
    w = pd.Series([s[0] for s in sizes])
    h = pd.Series([s[1] for s in sizes])
    print("width median/max:", w.median(), w.max(),
          "| height median/max:", h.median(), h.max())

    # 3. one dataset item
    ds = OCRDataset("data/splits/train.csv", a.root, proc, max_len=32)
    item = ds[0]
    print("pixel_values:", tuple(item["pixel_values"].shape),
          "| labels:", tuple(item["labels"].shape))

    # 4. save padded samples to look at
    os.makedirs("outputs/check", exist_ok=True)
    for i in range(10):
        r = df.iloc[i]
        pad_resize(Image.open(f"{a.root}/{r.image_path}"), size).save(f"outputs/check/{i}.png")
    print("saved 10 padded images to outputs/check", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    main(p.parse_args())
