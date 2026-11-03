# trocr-nepali

Fine-tune `paudelanil/trocr-devanagari-2` (TrOCR) on Nepali handwritten text. Training is intended for Kaggle with a T4 GPU, with local debugging support.

## Project structure

```text
trocr-nepali/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── debug.yaml
│   └── train.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── augment.py
│   ├── make_splits.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   └── kaggle_run.ipynb
└── outputs/
```

## Workflow

1. Run a local debug pass with `configs/debug.yaml` to validate the pipeline on a tiny subset.
2. Upload or clone the repository in Kaggle, install the requirements, and run the full training configuration with `configs/train.yaml` on a T4 GPU.
