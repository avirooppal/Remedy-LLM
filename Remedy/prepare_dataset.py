import csv
import json
import random
import os

CSV_FILE = "Home Remedies.csv"
TRAIN_OUTPUT = "data/train.jsonl"
VAL_OUTPUT = "data/val.jsonl"

def augment_data(row):
    item, issue, remedy, yogasan = row
    if not issue or not remedy:
        return []

    # Clean text
    issue = issue.strip().lower()
    remedy = remedy.strip()
    
    variations = [
        f"Suggest a home remedy for {issue}.",
        f"What can I take at home if I have {issue}?",
        f"I am suffering from {issue}. How can I treat it naturally?",
        f"Provide an Ayurvedic or home remedy for {issue}.",
        f"What is a good natural cure for {issue}?"
    ]
    
    samples = []
    for var in variations:
        sample = {
            "instruction": var,
            "input": "",
            "output": remedy
        }
        samples.append(sample)
    
    return samples

def main():
    os.makedirs("data", exist_ok=True)
    all_samples = []
    
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if len(row) >= 3:
                samples = augment_data(row)
                all_samples.extend(samples)
                
    # Shuffle the dataset
    random.shuffle(all_samples)
    
    # Split 80/20
    split_idx = int(len(all_samples) * 0.8)
    train_data = all_samples[:split_idx]
    val_data = all_samples[split_idx:]
    
    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for sample in val_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(train_data)} training samples and {len(val_data)} validation samples.")

if __name__ == "__main__":
    main()
