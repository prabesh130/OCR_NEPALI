import argparse, csv, unicodedata
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

def clean(t):
    return " ".join(unicodedata.normalize("NFC", str(t)).split())

def main(a):
    df = pd.read_csv(a.tsv, sep="\t", header=None, names=["image_path", "text"],
                     quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    df["text"] = df["text"].map(clean)
    df = df[df["text"].str.len() > 0]
    df["source_image"] = df["image_path"].str.split("_").str[0]

    ok = []
    for p in df["image_path"]:
        try:
            Image.open(f"{a.root}/{p}").verify(); ok.append(True)
        except Exception:
            ok.append(False)
    ok = pd.Series(ok, index=df.index)
    print("dropped unreadable:", (~ok).sum())
    df = df[ok]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, rest_idx = next(gss.split(df, groups=df["source_image"]))
    train, rest = df.iloc[tr_idx], df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    v_idx, t_idx = next(gss2.split(rest, groups=rest["source_image"]))
    val, test = rest.iloc[v_idx], rest.iloc[t_idx]

    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_csv(f"{a.out}/{name}.csv", index=False)
        print(name, len(d), "pages:", d["source_image"].nunique())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tsv"); p.add_argument("--root"); p.add_argument("--out", default="data/splits")
    main(p.parse_args())