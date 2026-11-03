import random

from PIL import Image, ImageEnhance, ImageFilter


class TrainAugment:
    """Light augmentation for word crops. Applied to train only."""

    def __call__(self, img):
        img = img.convert("RGB")
        if random.random() < 0.5:
            img = img.rotate(random.uniform(-3, 3), resample=Image.BICUBIC,
                             expand=True, fillcolor=(255, 255, 255))
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
        if random.random() < 0.4:
            w, h = img.size
            s = random.uniform(0.85, 1.0)
            img = img.resize((max(8, int(w * s)), max(8, int(h * random.uniform(0.9, 1.0)))),
                             Image.BICUBIC)
        return img
