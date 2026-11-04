import argparse

import jiwer
import torch
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          VisionEncoderDecoderModel, default_data_collator)

from src.augment import TrainAugment
from src.dataset import OCRDataset
from src.utils import MODEL, load_processor


def main(a):
    proc = load_processor()
    tok = proc.tokenizer

    model = VisionEncoderDecoderModel.from_pretrained(a.model)
    cfg = model.config
    cfg.decoder_start_token_id = tok.cls_token_id
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.sep_token_id
    cfg.vocab_size = cfg.decoder.vocab_size

    train_ds = OCRDataset(f"{a.splits}/train.csv", a.root, proc, a.max_len, augment=TrainAugment())
    val_ds = OCRDataset(f"{a.splits}/val.csv", a.root, proc, a.max_len)
    if a.max_samples:
        train_ds.df = train_ds.df.head(a.max_samples)
        val_ds.df = val_ds.df.head(max(8, a.max_samples // 4))

    def compute_metrics(p):
        pred_ids = p.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = p.label_ids.copy()
        label_ids[label_ids == -100] = tok.pad_token_id
        preds = tok.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        refs = tok.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        preds = [x.strip() for x in preds]
        return {"cer": jiwer.cer(refs, preds), "wer": jiwer.wer(refs, preds)}

    steps_per_epoch = max(1, -(-len(train_ds) // (a.bs * a.accum)))
    warmup_steps = max(1, int(0.1 * steps_per_epoch * a.epochs))
    print('steps/epoch:', steps_per_epoch, '| warmup steps:', warmup_steps)

    args = Seq2SeqTrainingArguments(
        output_dir=a.out,
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=torch.cuda.is_available() and not a.no_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=a.max_len,
        generation_num_beams=4,
        logging_steps=20,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        use_cpu=a.cpu,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(f"{a.out}/best")
    tok.save_pretrained(f"{a.out}/best")
    print("saved best model to", f"{a.out}/best")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--splits", default="data/splits")
    p.add_argument("--out", default="outputs/run1")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())
