import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./outputs/remedies-slm"

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

def main():
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": "data/train.jsonl",
        "validation": "data/val.jsonl"
    })
    
    dataset = dataset.map(format_prompt)
    print("Dataset loaded and formatted.")

    print(f"Loading {MODEL_NAME}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # TinyLlama tokenizer might not have pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "v_proj",
            "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1, # keeping epochs small for speed on generic hardware, adjust if needed
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        max_seq_length=256, # kept short for home remedies to save VRAM
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished! Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
