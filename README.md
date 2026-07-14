# Fake News Detection: TF-IDF vs BERT vs DistilBERT

A controlled comparison of five text classification models on the [GonzaloA/fake_news](https://huggingface.co/datasets/GonzaloA/fake_news) dataset, investigating whether the computational cost of large pretrained language models is justified for fake news detection.

**Best model:** DistilBERT (fine-tuned) — **98.63%** accuracy on 8,101 test articles.

---

## Research Questions

1. Does fine-tuning outperform freezing the pretrained encoder for fake news detection?
2. Does the larger BERT model justify its higher computational cost compared to the smaller DistilBERT?

---

## Results

All models were trained under identical conditions (same splits, hyperparameters, preprocessing, seed = 42) on a Google Colab T4 GPU.

| Model | Accuracy | F1 Macro | Train (s) | Inference (s) | GPU (GB) | Params (M) |
|-------|:--------:|:--------:|:---------:|:-------------:|:--------:|:----------:|
| TF-IDF + Logistic Regression | 0.9680 | 0.9679 | **20.9** | **0.008** | 0.00 | 0.0 |
| DistilBERT Frozen | 0.9124 | 0.9117 | 266.6 | 72.0 | 0.84 | 67.0 |
| **DistilBERT Fine-tuned** | **0.9863** | **0.9862** | 711.4 | 69.0 | 2.76 | 67.0 |
| BERT Frozen | 0.6803 | 0.6321 | 495.2 | 131.5 | 2.19 | 109.5 |
| BERT Fine-tuned | 0.9846 | 0.9845 | 1383.8 | 131.3 | 5.75 | 109.5 |

![Full model comparison](figures/full_comparison.png)

---

## Key Findings

**1. Fine-tuning is essential.** Freezing BERT drops accuracy by 30 percentage points and collapses the model to a majority-class predictor. Freezing DistilBERT loses 7 points but keeps useful discrimination.

**2. BERT does not justify its cost.** Fine-tuned DistilBERT beats fine-tuned BERT by 0.17 points while using half the parameters, half the training time, and less than half the GPU memory. On this task, DistilBERT is Pareto-dominant.

**3. TF-IDF is a surprisingly strong baseline.** A simple TF-IDF + Logistic Regression reaches 96.8% accuracy — only 1.8 points below the best Transformer — while training in 21 seconds versus 12 minutes. Fake news in this dataset can be identified largely by vocabulary.

**4. Hard examples are a labelling problem.** The 62 articles misclassified by all three main models are mostly polemical opinion pieces whose stylistic markers resemble fabricated content, reflecting genuine ambiguity in the ground-truth labels.

---

## Project Structure

```
fake-news-detection/
├── README.md                       ← you are here
├── report/
│   └── fake_news_report.pdf        ← full 8-page academic report
├── notebook/
│   └── fake_news_detection.ipynb   ← runnable end-to-end pipeline
├── figures/                        ← all confusion matrices + comparison chart
├── results/
│   └── final_results.csv           ← reproducible final numbers
└── requirements.txt
```

---

## Reproducing the Results

**Environment:** Python 3.10+, CUDA-capable GPU recommended (tested on Google Colab T4).

```bash
git clone <this-repo>
cd fake-news-detection
pip install -r requirements.txt
jupyter notebook notebook/fake_news_detection.ipynb
```

Run all cells top to bottom. The notebook loads the dataset automatically from HuggingFace, applies preprocessing, trains all five models, and saves all figures and CSVs.

**Fixed hyperparameters** (identical across all Transformer runs):

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Max sequence length | 256 tokens |
| Batch size | 16 |
| Learning rate | 2e-5 (AdamW, weight_decay = 0.01) |
| Warmup | 10% linear |
| Epochs | 3 (with early-stopping, patience = 2) |
| Training subset size | 10,000 articles |

---

## Data

- **Source:** [GonzaloA/fake_news](https://huggingface.co/datasets/GonzaloA/fake_news) on HuggingFace Datasets
- **Licence:** CC BY 4.0
- **Task:** binary classification (FAKE = 0, REAL = 1)
- **Test set:** 8,101 articles after cleaning
- **Class balance:** ~55% REAL / 45% FAKE

Preprocessing includes lowercasing, URL/email removal, non-alphabetical character removal, deduplication and empty-text filtering.

---

## Full Report

The complete 8-page academic report — with methodology, background, per-model confusion matrix analysis, error analysis and limitations — is available at [`report/fake_news_report.pdf`](report/fake_news_report.pdf).

---

## Tech Stack

- **PyTorch** — model training
- **HuggingFace Transformers** — BERT, DistilBERT
- **HuggingFace Datasets** — data loading
- **scikit-learn** — TF-IDF baseline and metrics
- **pandas, matplotlib, seaborn** — analysis and visualization

---

## Context

This project was completed as the final submission for the *Python for NLP* course at Heinrich-Heine-Universität Düsseldorf.

---

## License

MIT for the code in this repository. The dataset retains its original CC BY 4.0 licence from the source.
