# 🌿 India Home Remedies SLM — Project Plan

> Build a Small Language Model (SLM) fine-tuned on Indian home remedies that answers symptom/allergy queries with traditional Ayurvedic and household solutions.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Collection & Preparation](#4-dataset-collection--preparation)
5. [Base Model Selection](#5-base-model-selection)
6. [Fine-Tuning Strategy](#6-fine-tuning-strategy)
7. [Training Pipeline](#7-training-pipeline)
8. [Evaluation & Testing](#8-evaluation--testing)
9. [Deployment](#9-deployment)
10. [Project Folder Structure](#10-project-folder-structure)
11. [Cost Estimate](#11-cost-estimate)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Disclaimer Requirement](#13-disclaimer-requirement)
14. [Master Project Checklist](#14-master-project-checklist)

---

## 1. Project Overview

### Goal
Build a lightweight, domain-specific language model that:
- Accepts **symptoms / allergies / health complaints** as input
- Returns **traditional Indian home remedies** as output
- Runs efficiently on consumer-grade hardware (CPU or single GPU)
- Can be deployed as a local app, chatbot, or API

### Input / Output Examples

| Input (Symptom) | Output (Remedy) |
|---|---|
| "I have a cold and blocked nose" | "Try steam inhalation with eucalyptus oil. Drink warm tulsi-ginger tea with honey." |
| "Feeling bloated after eating" | "Take roasted ajwain with a pinch of black salt and warm water." |
| "Sudden toothache" | "Place a clove or apply clove oil on the affected tooth for temporary relief." |
| "Mild fever" | "Drink tulsi decoction, coconut water, and stay hydrated with herbal teas." |

---

## 2. Tech Stack

| Layer | Tool / Library | Reason |
|---|---|---|
| Language | Python 3.10+ | Industry standard for ML |
| Base Model | `TinyLlama-1.1B` or `Phi-2 (2.7B)` or `Mistral-7B` | Small, fast, fine-tunable |
| Fine-Tuning | `LoRA` via `PEFT` + `Transformers` | Parameter-efficient, low VRAM |
| Training Framework | `Hugging Face Transformers` + `TRL (SFTTrainer)` | Best ecosystem support |
| Dataset Format | JSONL (instruction-response pairs) | Standard fine-tuning format |
| Quantization | `bitsandbytes` (4-bit / 8-bit) | Reduces memory footprint |
| Experiment Tracking | `Weights & Biases (wandb)` | Monitor training metrics |
| Serving / Inference | `llama.cpp` or `FastAPI` + `Uvicorn` | Lightweight serving |
| Frontend (optional) | `Gradio` or `Streamlit` | Quick demo UI |
| Version Control | Git + GitHub | Code management |
| Environment | Conda / venv | Dependency isolation |

---

## 3. Environment Setup

### Step 1 — Prerequisites

```bash
# Required
Python >= 3.10
CUDA >= 11.8 (if using GPU)
Git

# Check GPU availability
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 2 — Create Virtual Environment

```bash
conda create -n remedies-slm python=3.10 -y
conda activate remedies-slm
```

### Step 3 — Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers==4.40.0 \
            datasets \
            peft \
            trl \
            bitsandbytes \
            accelerate \
            scipy \
            sentencepiece \
            wandb \
            gradio \
            fastapi \
            uvicorn \
            python-dotenv \
            evaluate \
            rouge_score \
            nltk
```

### Step 4 — Login to Hugging Face & W&B

```bash
huggingface-cli login        # Paste your HF token
wandb login                  # Paste your W&B API key
```

---

## 4. Dataset Collection & Preparation

This is the **most critical step**. The quality of the dataset directly determines model quality.

### Step 4.1 — Data Sources

Collect Q&A pairs from the following sources:

| Source | Type | Notes |
|---|---|---|
| Manual curation (seed data) | Structured Q&A | Use the provided examples as seed |
| Books: "Home Remedies" by Vasant Lad | Text extraction | Ayurvedic resource |
| AYUSH Ministry portal (ayush.gov.in) | Web scraping | Government-approved remedies |
| NDTV Food / Times of India Health | Web scraping | Popular Indian health content |
| YouTube transcripts (health channels) | Transcript extraction | Spoken remedy advice |
| Reddit r/Ayurveda, r/IndianFood | Community Q&A | Real user queries |
| NuskheInHindi.com, Boldsky.com | Web scraping | Hindi-English remedy sites |

### Step 4.2 — Data Format (JSONL)

Each sample must follow this **instruction-tuning format**:

```json
{
  "instruction": "What can I take at home if I have a cold?",
  "input": "",
  "output": "Drink warm tulsi-ginger tea with honey. It helps reduce congestion and soothes the throat. You can also try steam inhalation with hot water and eucalyptus oil to clear nasal passages."
}
```

For symptom-based input with extra context:

```json
{
  "instruction": "Suggest a home remedy for the following symptom.",
  "input": "I have acidity after eating spicy food.",
  "output": "Drink a glass of cold milk or chew fennel seeds (saunf) after meals. You can also try a mixture of jeera (cumin) water to soothe the stomach lining."
}
```

### Step 4.3 — Dataset Categories to Cover

```
├── Cold & Cough (🤧)
├── Digestion Issues (🍽️)
├── Fever & Immunity (🤒)
├── Toothache (🦷)
├── Headache (😖)
├── Skin Care (✨)
├── Joint & Muscle Pain (💪)
├── Eye Care (👁️)
├── Hair & Scalp (💆)
├── Throat & Tonsils
├── Diabetes (mild management)
├── Menstrual Cramps
├── Sleep & Stress
├── Burns & Cuts (minor)
├── Allergies & Rashes
└── Infant / Child Remedies
```

### Step 4.4 — Data Cleaning Script

```python
# scripts/clean_dataset.py
import json, re

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def validate_sample(sample):
    required = ['instruction', 'output']
    return all(k in sample and len(sample[k]) > 10 for k in required)

def clean_dataset(input_path, output_path):
    cleaned = []
    with open(input_path) as f:
        for line in f:
            sample = json.loads(line)
            sample['instruction'] = clean_text(sample['instruction'])
            sample['output'] = clean_text(sample['output'])
            if validate_sample(sample):
                cleaned.append(sample)
    
    with open(output_path, 'w') as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Cleaned: {len(cleaned)} samples saved.")
```

### Step 4.5 — Dataset Size Target

| Phase | Samples | Notes |
|---|---|---|
| Minimum Viable | 500 | Proof of concept only |
| Good Quality | 2,000–5,000 | Decent domain coverage |
| Production Ready | 10,000+ | Robust generalization |

### Step 4.6 — Train/Validation/Test Split

```
train:      80%   (e.g., 4,000 samples)
validation: 10%   (e.g., 500 samples)
test:       10%   (e.g., 500 samples)
```

---

## 5. Base Model Selection

Choose based on hardware and quality requirements:

| Model | Parameters | Min VRAM | Quality | Recommended For |
|---|---|---|---|---|
| `TinyLlama-1.1B` | 1.1B | 4 GB | ⭐⭐⭐ | CPU or very low-end GPU |
| `microsoft/phi-2` | 2.7B | 6 GB | ⭐⭐⭐⭐ | Mid-range GPU, good reasoning |
| `mistralai/Mistral-7B-v0.1` | 7B | 16 GB (4-bit: 8GB) | ⭐⭐⭐⭐⭐ | Best quality, production use |
| `google/gemma-2b` | 2B | 6 GB | ⭐⭐⭐⭐ | Google ecosystem |

**Recommended Starting Point:** `microsoft/phi-2` or `TinyLlama-1.1B` for fast iteration.

---

## 6. Fine-Tuning Strategy

### Use LoRA (Low-Rank Adaptation)

LoRA adds small trainable matrices to the model instead of updating all weights. This means:
- 10–100x fewer trainable parameters
- Fits in less than 8GB VRAM
- Training completes in hours, not days

### LoRA Configuration

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,                      # Rank of update matrices
    lora_alpha=32,             # Scaling factor
    target_modules=[           # Which layers to fine-tune
        "q_proj", "v_proj",
        "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### Prompt Template

Wrap each sample in a consistent prompt template:

```python
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
```

---

## 7. Training Pipeline

### Step 7.1 — Prepare Dataset for Training

```python
# scripts/prepare_data.py
from datasets import load_dataset

def format_prompt(sample):
    input_text = sample.get('input', '')
    return {
        "text": f"""### Instruction:
{sample['instruction']}

### Input:
{input_text}

### Response:
{sample['output']}"""
    }

dataset = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/val.jsonl"
})

dataset = dataset.map(format_prompt)
```

### Step 7.2 — Training Script

```python
# train.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

MODEL_NAME = "microsoft/phi-2"    # Change to your chosen base model
OUTPUT_DIR = "./outputs/remedies-slm"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=3,
    fp16=True,
    report_to="wandb",
    run_name="remedies-slm-v1",
    max_seq_length=512,
    dataset_text_field="text"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
```

### Step 7.3 — Training Hyperparameters Guide

| Parameter | Recommended Value | Notes |
|---|---|---|
| `learning_rate` | `2e-4` | Standard LoRA LR |
| `num_train_epochs` | `3–5` | Start with 3, increase if underfitting |
| `batch_size` | `4` (per device) | Increase if VRAM allows |
| `gradient_accumulation` | `4` | Effective batch = 16 |
| `max_seq_length` | `512` | Sufficient for remedies |
| `lora_r` | `16` | Higher = more capacity |
| `lora_alpha` | `32` | Usually 2x rank |

---

## 8. Evaluation & Testing

### Step 8.1 — Quantitative Metrics

```python
# evaluate.py
from evaluate import load
rouge = load("rouge")
bleu  = load("bleu")

# Run on test set
predictions = [model.generate(prompt) for prompt in test_prompts]
references  = [sample['output'] for sample in test_samples]

rouge_scores = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
```

| Metric | What It Measures | Target |
|---|---|---|
| ROUGE-L | Overlap with reference output | > 0.35 |
| BLEU | N-gram precision | > 0.20 |
| Manual Score (1–5) | Medical accuracy + helpfulness | > 4.0 |

### Step 8.2 — Manual Evaluation Checklist

For every test output, verify:
- [ ] Remedy is **actually an Indian home remedy** (not a pharmaceutical)
- [ ] Remedy matches the **symptom category** correctly
- [ ] **Ingredients are mentioned** (tulsi, ginger, ajwain, etc.)
- [ ] Response is **safe** (no harmful advice)
- [ ] No **hallucinated medical claims**
- [ ] Response **disclaimer** is present if deployed publicly

### Step 8.3 — Quick Inference Test

```python
# inference.py
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./outputs/remedies-slm",
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

prompt = """### Instruction:
What home remedy can I use for a headache?

### Input:


### Response:"""

result = pipe(prompt)
print(result[0]['generated_text'].split("### Response:")[-1].strip())
```

---

## 9. Deployment

### Option A — Gradio Web App (Recommended for Demo)

```python
# app.py
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="./outputs/remedies-slm", ...)

def get_remedy(symptom):
    prompt = f"### Instruction:\n{symptom}\n\n### Input:\n\n### Response:\n"
    output = pipe(prompt, max_new_tokens=200)[0]['generated_text']
    remedy = output.split("### Response:")[-1].strip()
    disclaimer = "\n\n⚠️ *This is a home remedy suggestion only. Consult a doctor for serious conditions.*"
    return remedy + disclaimer

demo = gr.Interface(
    fn=get_remedy,
    inputs=gr.Textbox(label="Describe your symptom or health issue"),
    outputs=gr.Textbox(label="Suggested Home Remedy"),
    title="🌿 India Home Remedies Assistant",
    description="Get traditional Indian home remedy suggestions based on your symptoms."
)

demo.launch(share=True)
```

```bash
python app.py
```

### Option B — FastAPI REST API

```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="India Remedies SLM API")

class SymptomRequest(BaseModel):
    symptom: str

@app.post("/remedy")
def get_remedy(req: SymptomRequest):
    result = generate_remedy(req.symptom)
    return {"symptom": req.symptom, "remedy": result}
```

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Option C — GGUF Export for llama.cpp (CPU Inference)

```bash
# Convert model to GGUF for ultra-light CPU deployment
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

python convert.py ./outputs/remedies-slm --outfile remedies-slm.gguf

./main -m remedies-slm.gguf -p "What remedy helps with acidity?" -n 200
```

### Option D — Push to Hugging Face Hub

```python
model.push_to_hub("your-username/india-home-remedies-slm")
tokenizer.push_to_hub("your-username/india-home-remedies-slm")
```

---

## 10. Project Folder Structure

```
india-remedies-slm/
│
├── data/
│   ├── raw/                      # Raw scraped/collected data
│   │   ├── cold_cough.jsonl
│   │   ├── digestion.jsonl
│   │   ├── fever.jsonl
│   │   └── ...
│   ├── processed/
│   │   ├── train.jsonl           # 80% split
│   │   ├── val.jsonl             # 10% split
│   │   └── test.jsonl            # 10% split
│   └── seed_data.jsonl           # Initial hand-curated examples
│
├── scripts/
│   ├── scrape_data.py            # Web scrapers
│   ├── clean_dataset.py          # Data cleaning
│   ├── prepare_data.py           # Format into prompt template
│   └── evaluate.py               # Evaluation metrics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_experiment.ipynb
│   └── 03_inference_demo.ipynb
│
├── train.py                      # Main training script
├── inference.py                  # Quick test inference
├── app.py                        # Gradio frontend
├── api.py                        # FastAPI backend
│
├── outputs/
│   └── remedies-slm/             # Saved model checkpoints
│       ├── adapter_config.json
│       ├── adapter_model.bin
│       └── tokenizer files
│
├── requirements.txt
├── .env                          # HF_TOKEN, WANDB_API_KEY
├── .gitignore
└── README.md
```

---

## 11. Cost Estimate

### If Using Cloud GPU (e.g., Google Colab Pro / RunPod / Lambda Labs)

| Item | Cost |
|---|---|
| GPU training (A100 40GB, ~5 hours) | $10–25 |
| Storage (model + dataset) | $1–5/month |
| Hosting (optional, HuggingFace Spaces) | Free tier available |
| **Total MVP** | **~$15–30** |

### If Using Local GPU (RTX 3090 / 4090)

| Item | Cost |
|---|---|
| Electricity (~8 hours training) | $0.50–2.00 |
| **Total MVP** | **< $5** |

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Dataset too small / low quality | High | High | Start with manual curation; use data augmentation |
| Model hallucinates harmful remedies | Medium | High | Add safety evaluation step; filter outputs |
| Overfitting on small dataset | Medium | Medium | Use dropout, fewer epochs, validation tracking |
| VRAM out of memory during training | Medium | Medium | Use 4-bit quantization + gradient checkpointing |
| Copyrighted data in scraping | Low | High | Use only public domain / CC-licensed sources |
| Model ignores symptom context | Medium | Medium | Improve prompt template; add few-shot examples |

---

## 13. Disclaimer Requirement

> ⚠️ **IMPORTANT**: This model provides **traditional Indian home remedy suggestions only**.
> It is **NOT a medical diagnosis tool** and should **NOT replace professional medical advice**.
> Always consult a licensed physician for serious or persistent health conditions.
> The model may make mistakes. Use at your own discretion.

This disclaimer **must be included** in:
- Every API response
- The Gradio/Streamlit UI
- The model card on HuggingFace Hub
- The README.md

---

## 14. Master Project Checklist

Work through each phase in order. Every checkbox is a concrete, completable action.

---

### 🖥️ Phase 1 — Environment Setup

- [ ] Install Python 3.10+ and verify with `python --version`
- [ ] Install CUDA 11.8+ (skip if using CPU or Colab)
- [ ] Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Create conda environment: `conda create -n remedies-slm python=3.10 -y`
- [ ] Activate environment: `conda activate remedies-slm`
- [ ] Install all pip dependencies from `requirements.txt`
- [ ] Create HuggingFace account and generate API token
- [ ] Run `huggingface-cli login` and paste token
- [ ] Create Weights & Biases account and generate API key
- [ ] Run `wandb login` and paste API key
- [ ] Create `.env` file with `HF_TOKEN` and `WANDB_API_KEY`
- [ ] Create project folder structure as defined in Section 10
- [ ] Initialize Git repo and add `.gitignore` (include `outputs/`, `.env`, `__pycache__/`)

---

### 📦 Phase 2 — Dataset Collection

- [ ] Write 50+ hand-curated Q&A pairs covering Cold & Cough category
- [ ] Write 50+ hand-curated Q&A pairs covering Digestion Issues
- [ ] Write 50+ hand-curated Q&A pairs covering Fever & Immunity
- [ ] Write 50+ hand-curated Q&A pairs covering Toothache
- [ ] Write 50+ hand-curated Q&A pairs covering Headache
- [ ] Write 50+ hand-curated Q&A pairs covering Skin Care
- [ ] Write 30+ pairs each for remaining 10 categories (Joint Pain, Eye Care, Hair, etc.)
- [ ] Format all samples as JSONL with `instruction`, `input`, `output` keys
- [ ] Save combined seed data to `data/raw/seed_data.jsonl`
- [ ] Set up web scraper for at least 2 public sources (AYUSH portal, health blogs)
- [ ] Run scrapers and collect additional samples into `data/raw/`
- [ ] Verify all scraped content is from public domain or CC-licensed sources

---

### 🧹 Phase 3 — Data Cleaning & Preparation

- [ ] Run `scripts/clean_dataset.py` on all raw JSONL files
- [ ] Verify no sample has `output` shorter than 30 characters
- [ ] Remove any duplicate Q&A pairs
- [ ] Check for and remove any pharmaceutical drug recommendations (keep only home remedies)
- [ ] Ensure every `output` mentions at least one named ingredient (tulsi, ginger, etc.)
- [ ] Perform 80/10/10 train-val-test split
- [ ] Save splits to `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`
- [ ] Confirm final counts: at least 400 train / 50 val / 50 test samples
- [ ] Run `scripts/prepare_data.py` to wrap samples in prompt template
- [ ] Spot-check 20 random samples from each split manually

---

### 🤖 Phase 4 — Base Model & LoRA Setup

- [ ] Decide on base model (`microsoft/phi-2` recommended for first run)
- [ ] Download base model via `AutoModelForCausalLM.from_pretrained()`
- [ ] Confirm model loads without errors in 4-bit quantization mode
- [ ] Set `tokenizer.pad_token = tokenizer.eos_token`
- [ ] Define `LoraConfig` with `r=16`, `lora_alpha=32`, target modules set
- [ ] Apply LoRA with `get_peft_model()` and print trainable parameter count
- [ ] Confirm trainable params are < 2% of total model params
- [ ] Run a single forward pass on 1 sample to verify no shape/dtype errors

---

### 🏋️ Phase 5 — Training

- [ ] Configure `SFTTrainer` with training arguments from Section 7
- [ ] Set `report_to="wandb"` and verify W&B dashboard is receiving logs
- [ ] Launch first training run with 1 epoch as a smoke test
- [ ] Confirm training loss is decreasing (check W&B loss curve)
- [ ] Confirm no CUDA OOM errors; if OOM reduce `per_device_train_batch_size` to 2
- [ ] Run full training for 3 epochs on complete dataset
- [ ] Monitor validation loss — stop early if val loss increases for 2+ eval steps
- [ ] Save final LoRA adapter to `outputs/remedies-slm/`
- [ ] Confirm adapter files exist: `adapter_config.json`, `adapter_model.bin`
- [ ] Commit trained adapter to Git (or push to HuggingFace Hub as private repo)

---

### 🧪 Phase 6 — Evaluation & Testing

- [ ] Run `inference.py` on 10 manual test prompts and read outputs
- [ ] Verify outputs are in English and mention Indian ingredients
- [ ] Run `evaluate.py` on `data/processed/test.jsonl`
- [ ] Record ROUGE-L score (target: > 0.35)
- [ ] Record BLEU score (target: > 0.20)
- [ ] Manually score 20 outputs on a 1–5 scale for accuracy and helpfulness
- [ ] Check: does the model respond correctly to at least 15/20 test inputs?
- [ ] Test edge cases: vague input ("I feel sick"), multi-symptom input, non-health input
- [ ] Verify model does NOT output pharmaceutical drug names or dosages
- [ ] Verify model does NOT claim to diagnose diseases
- [ ] Document all evaluation results in `notebooks/03_inference_demo.ipynb`
- [ ] If ROUGE-L < 0.30, go back to Phase 3 and expand dataset before retraining

---

### 🚀 Phase 7 — Deployment

- [ ] Add medical disclaimer string to a shared `utils/disclaimer.py` constant
- [ ] Build `app.py` with Gradio — symptom input → remedy output + disclaimer
- [ ] Test Gradio app locally at `http://localhost:7860`
- [ ] Verify disclaimer appears on every Gradio response
- [ ] Build `api.py` with FastAPI `/remedy` POST endpoint
- [ ] Test API with `curl -X POST http://localhost:8000/remedy -d '{"symptom":"headache"}'`
- [ ] Confirm API response includes `remedy` and `disclaimer` fields
- [ ] (Optional) Export model to GGUF format for CPU-only deployment via llama.cpp
- [ ] (Optional) Deploy Gradio app to HuggingFace Spaces (free tier)
- [ ] Push final model adapter to HuggingFace Hub with a complete model card

---

### 📄 Phase 8 — Documentation & Release

- [ ] Write `README.md` with: project description, setup instructions, usage examples
- [ ] Add disclaimer prominently at the top of `README.md`
- [ ] Write model card on HuggingFace Hub (training data, limitations, use cases)
- [ ] Add `requirements.txt` with pinned versions
- [ ] Add example inference code snippet to README
- [ ] Final Git commit with tag `v1.0`
- [ ] Share repo / HuggingFace Space link

---

### 🔒 Safety Final Check (Do Before Any Public Release)

- [ ] Model never outputs content suggesting self-medication for serious conditions
- [ ] Model never names specific pharmaceutical drugs or dosages
- [ ] Disclaimer is visible in UI, API response, model card, and README
- [ ] Tested with adversarial inputs (e.g., "how do I cure cancer at home")
- [ ] All data sources documented and confirmed copyright-safe

---

*Built with ❤️ to preserve and share India's traditional healing wisdom.*
