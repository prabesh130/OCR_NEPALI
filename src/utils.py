from transformers import AutoTokenizer, TrOCRProcessor, ViTImageProcessor

MODEL = "paudelanil/trocr-devanagari-2"


def load_processor(model_name=MODEL):
    image_processor = ViTImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 224},
        resample=2,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
