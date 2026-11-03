import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def pad_resize(img, size=224):
    img = img.convert("RGB")
    scale = min(size / img.width, size / img.height)
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(new_size, Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


class OCRDataset(Dataset):
    def __init__(self, csv_path, root, processor, max_len=32, augment=None):
        self.df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        self.root, self.proc, self.max_len, self.augment = root, processor, max_len, augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(f"{self.root}/{r.image_path}")
        if self.augment:
            img = self.augment(img)
        img = pad_resize(img, self.proc.image_processor.size["height"])
        pixel_values = self.proc(images=img, return_tensors="pt").pixel_values.squeeze(0)
        ids = self.proc.tokenizer(r.text, padding="max_length", max_length=self.max_len,
                                  truncation=True).input_ids
        labels = [t if t != self.proc.tokenizer.pad_token_id else -100 for t in ids]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}
