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
    out["pred"] = preds 
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
