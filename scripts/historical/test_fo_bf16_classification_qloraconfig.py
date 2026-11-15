"""
FO + BF16 + Classification Head Control Experiment

Purpose: Descriptive control with major settings matched to QLoRA
- Same data: 2000-example SST-2 train subset
- Same batch size: 4
- Same learning rate: 2e-5
- Same max_length: 128
- Same max_steps: 2000
- Same LoRA config: r=8, q/k/v_proj targets
- Difference: BF16 precision (not 4-bit), classification head (not generative)

Heads, objectives, precision and runtime differ; quantization is not isolated.
SEED is a recorded label only: this script does not initialize random generators.
Historical execution logic is preserved; this content edit does not rerun training.
"""

import json
import os
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import sys

# Get seed from command line argument
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

MODEL_ID = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = f"results/fo_bf16_classification_qloraconfig_seed{SEED}"
NUM_STEPS = 2000
EVAL_EVERY = 100
BATCH_SIZE = 4
MAX_LENGTH = 128
ACC_EVAL_SAMPLES = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print(f"FO + BF16 + Classification Head (QLoRA Config, seed={SEED})")
print("=" * 70)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load SST-2 dataset
print("\nLoading SST-2 dataset...")
ds = load_dataset("glue", "sst2")

def prepare_example(ex):
    return {
        "sentence": ex["sentence"].strip(),
        "label": ex["label"]
    }

train_ds = ds["train"].select(range(2000)).map(prepare_example)
val_ds = ds["validation"].map(prepare_example)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# Tokenization for classification
def tokenize_batch(examples):
    encoded = tokenizer(
        examples["sentence"],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"].to("xpu"),
        "attention_mask": encoded["attention_mask"].to("xpu"),
        "labels": torch.tensor(examples["label"]).to("xpu")
    }

# Load model with classification head
print("\nLoading model for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = model.to("xpu")

# CRITICAL FIX: Set pad_token_id in model config (prevents ValueError with batch>1)
model.config.pad_token_id = tokenizer.pad_token_id

# Apply LoRA config (same as QLoRA: r=8, q/k/v_proj)
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(model, config)

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {n_trainable:,} / {n_total:,}")

model.train()

# Setup optimizer (PyTorch's AdamW, same lr as QLoRA: 2e-5)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-5,
)

# Helper: batched validation loss
def evaluate_val_loss():
    model.eval()
    total_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for i in range(0, min(200, len(val_ds)), BATCH_SIZE):
            val_examples = val_ds[i:i+BATCH_SIZE]
            val_batch = tokenize_batch(val_examples)
            val_outputs = model(**val_batch)
            total_loss += val_outputs.loss.item() * len(val_examples["label"])
            n_val += len(val_examples["label"])
    model.train()
    return total_loss / max(n_val, 1)

# Helper: batched validation accuracy
def evaluate_val_acc(n_samples=ACC_EVAL_SAMPLES):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, min(n_samples, len(val_ds)), BATCH_SIZE):
            val_examples = val_ds[i:i+BATCH_SIZE]
            val_batch = tokenize_batch(val_examples)
            val_outputs = model(**val_batch)
            preds = val_outputs.logits.argmax(dim=-1)
            labels = torch.tensor(val_examples["label"]).to("xpu")
            correct += (preds == labels).sum().item()
            total += len(val_examples["label"])
    model.train()
    return correct / total if total > 0 else 0

# Training loop
torch.xpu.reset_peak_memory_stats()
print(f"\n--- Training {NUM_STEPS} steps (seed={SEED}) ---")
t_start = time.perf_counter()
losses = []
step_times = []

for step in range(1, NUM_STEPS + 1):
    start = ((step - 1) * BATCH_SIZE) % len(train_ds)
    examples = train_ds[start:start+BATCH_SIZE]
    batch = tokenize_batch(examples)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    torch.xpu.synchronize()

    losses.append(loss.item())
    step_times.append(time.perf_counter() - t_start)

    if step % EVAL_EVERY == 0:
        val_loss = evaluate_val_loss()
        val_acc = evaluate_val_acc()
        elapsed = time.perf_counter() - t_start
        print(f"  step {step:4d}: train_loss={loss.item():.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2%} time={elapsed:.0f}s")

# Final evaluation
print("\n--- Final evaluation ---")
final_val_loss = evaluate_val_loss()
final_val_acc = evaluate_val_acc()

peak_mem = torch.xpu.max_memory_allocated() / 1e9
total_time = time.perf_counter() - t_start

# Save adapter
adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"Adapter saved: {adapter_dir}")

# Save result.json
result = {
    "run_id": f"fo_bf16_clf_qloraconfig_seed{SEED}",
    "method": "first_order",
    "precision": "bfloat16",
    "model": MODEL_ID,
    "task": "SST-2",
    "training_objective": "classification_head",
    "seed": SEED,
    "phase": "control_experiment",
    "lora_rank": 8,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj"],
    "num_steps": NUM_STEPS,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "learning_rate": 2e-5,
    "trainable_parameter_count": n_trainable,
    "total_model_parameters": n_total,
    "initial_train_loss": losses[0],
    "final_train_loss": losses[-1],
    "final_validation_loss": final_val_loss,
    "final_validation_accuracy": final_val_acc,
    "validation_samples": ACC_EVAL_SAMPLES,
    "peak_gpu_memory_gb": peak_mem,
    "total_elapsed_time_sec": total_time,
    "mean_step_time_sec": total_time / NUM_STEPS if NUM_STEPS > 0 else None,
    "adapter_dir": adapter_dir,
    "status": "completed",
}
result_path = os.path.join(OUTPUT_DIR, "result.json")
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)

# Save loss curve
curve_path = os.path.join(OUTPUT_DIR, "train_loss.csv")
with open(curve_path, "w") as f:
    f.write("step,train_loss,elapsed_sec\n")
    for i, (l, t) in enumerate(zip(losses, step_times), 1):
        f.write(f"{i},{l},{t}\n")

print("\n" + "=" * 70)
print("CONTROL EXPERIMENT RESULT (FO + BF16 at QLoRA Config)")
print("=" * 70)
print(json.dumps(result, indent=2))
print(f"\nSaved to: {result_path}")
print("=" * 70)