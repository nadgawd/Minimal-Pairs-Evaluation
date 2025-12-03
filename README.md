# 🎓 Minimal-Pairs Evaluation: DeepSeek • Qwen • Mistral

<div align="center">

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/Transformers-4.x-FFD21E?style=for-the-badge&logo=huggingface" />
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch" />

<br><br>

A notebook-driven evaluation framework for comparing **sentence-level log probabilities** across large language models.

</div>

---

## 📌 Overview

This repository evaluates **minimal sentence pairs** (grammatical/plausible vs. ungrammatical/implausible) using three different causal language models:

- **DeepSeek-R1-Distill-Llama-8B**
- **Mistral-7B-v0.3**
- **Qwen2.5-7B**

Each notebook computes:

- Sentence log-probability  
- Perplexity  
- Confidence score  
- Verdict (does model prefer the good sentence?)  
- CSV outputs written to `/outputs/`

This project is **not notebook-agnostic**, meaning **all evaluation runs directly inside Jupyter notebooks**. No CLI or web interface is included.

---

## 🧠 Models Used

| Notebook | Model | HuggingFace ID |
|----------|--------|----------------|
| `deepseek-r1-distill-llama-8b.ipynb` | DeepSeek R1 Distill | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| `mistral-7b-v0-3.ipynb` | Mistral 7B v0.3 | `mistralai/Mistral-7B-v0.3` |
| `Qwen2_5_7B.ipynb` | Qwen 2.5 7B | `Qwen/Qwen2.5-7B` |

All models are loaded using `AutoModelForCausalLM`.

---

## ✨ Features

- Notebook-based execution  
- Direct minimal-pair evaluation  
- Log-probability comparison  
- Perplexity computation  
- Confidence scoring  
- CSV export  
- Supports any causal LM on HuggingFace  

---

## 📁 Repository Structure

```
deepseek-qwen-mistral-eval/
│
├── notebooks/
│   ├── deepseek-r1-distill-llama-8b.ipynb
│   ├── mistral-7b-v0-3.ipynb
│   └── Qwen2_5_7B.ipynb
│
├── data/              # Input CSV files used by notebooks
├── outputs/           # Generated CSVs from notebooks
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/nadgawd/deepseek-qwen-mistral-eval
cd deepseek-qwen-mistral-eval
```

### 2. Install dependencies

Create a `requirements.txt` (recommended):

```
transformers>=4.35.0
torch
pandas
numpy
tqdm
jupyter
```

Install:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Notebooks

Start Jupyter:

```bash
jupyter lab
```

Then open:

```
notebooks/deepseek-r1-distill-llama-8b.ipynb
notebooks/mistral-7b-v0-3.ipynb
notebooks/Qwen2_5_7B.ipynb
```

Run all cells (top → bottom).

### 📥 Input Files
Place your sentence-pair CSVs in:

```
data/
```

### 📤 Output Files
Notebooks save outputs to:

```
outputs/
```

---

## 📊 Scoring Method

All notebooks use the same scoring core:

```python
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss.item()                 # mean negative log-prob per token
num_tokens = inputs["input_ids"].shape[1]

sentence_log_prob = -loss * (num_tokens - 1)
perplexity = math.exp(loss)
confidence = abs(log_prob_s1 - log_prob_s2)
correct = (log_prob_s1 > log_prob_s2)
```

### Metrics produced
- `logP(sentence)`
- `perplexity`
- `confidence`
- `correct` verdict

---

## 📄 Output Format

Saved CSV structure:

| good_sentence | bad_sentence | logP_good | logP_bad | PPL_good | PPL_bad | confidence | correct |
|---------------|--------------|-----------|----------|----------|---------|------------|---------|

Example:

| I ate the apple | I ate the computer | -22.39 | -48.20 | 6.23 | 13.19 | 25.81 | TRUE |

---

## 🛠️ Troubleshooting

### GPU Out of Memory
Switch to CPU:

```python
device_map="cpu"
```

### Kaggle paths failing
Replace lines like:

```
/kaggle/input/...
```

with:

```
data/your_file.csv
```

### Slow model load
These are 7–8B parameter models, so expect large downloads.

---

## 🤝 Contributing

Suggestions welcome!

Potential improvements:

- Add unified scoring util (`score_utils.py`)
- Add CLI runner (`evaluate.py`)
- Add accuracy plots  
- Add normalization (0–10 plausibility scores)

---

## 📜 License

This project is intended for academic/research purposes.  
Model licenses follow their respective HuggingFace terms.

