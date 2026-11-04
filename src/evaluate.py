import argparse

import jiwer
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel

from src.dataset import OCRDataset
from src.utils import MODEL, load_processor


@torch.no_grad()
def predict(model, loader, proc, device, num_beams=4, max_new_tokens=32):
    model.eval()
    preds = []
    for batch in loader:
        out = model.generate(
            batch["pixel_values"].to(device),
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        preds += proc.tokenizer.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    return [p.strip() for p in preds]


def score(refs, preds):
    return {
        "CER": jiwer.cer(refs, preds),
        "WER": jiwer.wer(refs, preds),
        "exact_match": sum(r == p for r, p in zip(refs, preds)) / len(refs),
    }


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = load_processor()
    model = VisionEncoderDecoderModel.from_pretrained(a.model).to(device)
    print("decoder_start_token_id:", model.config.decoder_start_token_id,
          "| pad:", model.config.pad_token_id, "| eos:", model.config.eos_token_id)

    ds = OCRDataset(f"{a.splits}/{a.split}.csv", a.root, proc, max_len=32)
    loader = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=2)

    preds = predict(model, loader, proc, device)
    refs = ds.df["text"].tolist()
    res = score(refs, preds)
    print(f"{a.split}: n={len(refs)}", {k: round(v, 4) for k, v in res.items()})

    out = ds.df.copy()
    out["pred"] = predscat > src/evaluate_cheque.py << 'EOF'
import argparse
import re

import jiwer
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel

import src.dataset as dsmod
from src.dataset import OCRDataset
from src.utils import load_processor


def clean(t, field):
    t = " ".join(str(t).split())
    if field == "amount_in_numbers":
        t = re.sub(r"^[/\-=.\s]+|[/\-=.\s]+$", "", t).replace(" ", "")
    return t


@torch.no_grad()
def predict(model, loader, proc, device):
    model.eval()
    out_text = []
    for batch in loader:
        ids = model.generate(batch["pixel_values"].to(device), num_beams=4, max_length=24)
        out_text += proc.tokenizer.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [t.strip() for t in out_text]


def main(a):
    dsmod.RESIZE_MODE = a.resize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = load_processor()
    model = VisionEncoderDecoderModel.from_pretrained(a.model).to(device)

    ds = OCRDataset(a.csv, a.root, proc, max_len=32)
    if a.limit:
        ds.df = ds.df.head(a.limit).reset_index(drop=True)
    df = ds.df.copy()
    loader = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=2)

    raw = predict(model, loader, proc, device)
    df["pred"] = [clean(p, f) for p, f in zip(raw, df["field"])]
    df["text"] = [clean(t, f) for t, f in zip(df["text"], df["field"])]
    df["exact"] = df["text"] == df["pred"]
    df["len_ok"] = df["text"].str.len() == df["pred"].str.len()

    print(f"\nmodel: {a.model} | resize: {a.resize} | n={len(df)}")
    for field, g in df.groupby("field"):
        cer = jiwer.cer(g["text"].tolist(), [p if p else " " for p in g["pred"]])
        print(f"{field:18s} n={len(g):5d}  CER={cer:.4f}  exact={g['exact'].mean():.4f}"
              f"  length_match={g['len_ok'].mean():.4f}")

    df.to_csv(a.out, index=False)
    print("saved", a.out)
    wrong = df[~df["exact"]]
    print(wrong.groupby("field").head(8)[["field", "text", "pred"]].to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--root", required=True, help="folder with the cheque crops")
    p.add_argument("--model", required=True)
    p.add_argument("--resize", default="stretch", choices=["stretch", "pad"])
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="cheque_preds.csv")
    main(p.parse_args())
EOF
grep -n "RESIZE_MODE" src/dataset.py
    out.to_csv(f"outputs/{a.split}_preds_{a.tag}.csv", index=False)
    print("worst examples:")
    print(out[out["text"] != out["pred"]].head(15)[["text", "pred"]].to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--split", default="test")
    p.add_argument("--splits", default="data/splits")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--tag", default="baseline")
    main(p.parse_args())
