"""QLoRA 7B boundary probe (1 step) — Phase 3 feasibility test."""
import json
import os
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from ipex_llm.transformers import AutoModelForCausalLM
from ipex_llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training
from peft import LoraConfig
import sys

# Get seed from command line argument
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

MODEL_ID = "Qwen/Qwen2.5-7B"
OUTPUT_DIR = f"results/qb4_7b_boundary_probe_seed{SEED}"
NUM_STEPS = 1
EVAL_EVERY = 100
BATCH_SIZE = 2
MAX_LENGTH = 128
ACC_EVAL_SAMPLES = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print(f"QLoRA 7B BOUNDARY PROBE (1 step, seed={SEED})")
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
    sentence = ex["sentence"].strip()
    label_str = "positive" if ex["label"] == 1 else "negative"
    return {"sentence": sentence, "label_str": label_str, "label": ex["label"]}

train_ds = ds["train"].select(range(2000)).map(prepare_example)
val_ds = ds["validation"].map(prepare_example)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# Helper functions for prompts
def make_prompt(s):
    return f"Review: {s}\nSentiment:"

def make_full(s, l):
    return f"Review: {s}\nSentiment: {l}"

def make_batch(examples):
    sentences = list(examples["sentence"])
    labels_text = list(examples["label_str"])
    prompts = [make_prompt(s) for s in sentences]
    full_texts = [make_full(s, l) for s, l in zip(sentences, labels_text)]
    prompt_enc = tokenizer(prompts, padding=False, truncation=True, max_length=MAX_LENGTH)
    full_enc = tokenizer(full_texts, padding=True, truncation=True,
                          max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = full_enc["input_ids"].to("xpu")
    attention_mask = full_enc["attention_mask"].to("xpu")
    labels = input_ids.clone()
    for i in range(len(sentences)):
        prompt_len = len(prompt_enc["input_ids"][i])
        labels[i, :prompt_len] = -100
        labels[i, attention_mask[i] == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def cast_lm_head_input(module, inputs):
    hidden = inputs[0]
    if hidden.dtype != module.weight.dtype:
        return (hidden.to(module.weight.dtype), *inputs[1:])
    return None

def generate_predictions(model_to_use, n_samples=ACC_EVAL_SAMPLES):
    was_training = model_to_use.training
    model_to_use.eval()
    predictions = []
    try:
        with torch.no_grad():
            for i in range(min(n_samples, len(val_ds))):
                example = val_ds[i]
                prompt = make_prompt(example["sentence"])
                inputs = tokenizer(prompt, return_tensors="pt").to("xpu")
                generated = model_to_use.generate(
                    **inputs, max_new_tokens=3, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new_tokens = generated[0, inputs["input_ids"].shape[1]:]
                answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
                first_word = answer.split()[0].strip(".,!?") if answer else ""
                if first_word == "positive":
                    predictions.append("positive")
                elif first_word == "negative":
                    predictions.append("negative")
                else:
                    predictions.append("unknown")
    finally:
        if was_training:
            model_to_use.train()
    return predictions

def evaluate_loss():
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, min(200, len(val_ds)), BATCH_SIZE):
            examples = val_ds[i:i+BATCH_SIZE]
            batch = make_batch(examples)
            out = model(**batch)
            total_loss += out.loss.item() * len(examples["label"])
            n += len(examples["label"])
    model.train()
    return total_loss / max(n, 1)

def accuracy(preds):
    correct = sum(1 for i, p in enumerate(preds) if p == val_ds[i]["label_str"])
    return correct / max(len(preds), 1)

# Load NF4 quantized model
print("\nLoading NF4 4-bit model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    trust_remote_code=True,
)
model = model.to("xpu")
model = prepare_model_for_kbit_training(model)
if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()
model.config.use_cache = False

# Apply LoRA config
config = LoraConfig(
    r=8, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {n_trainable:,} / {n_total:,}")

# Register hook for dtype casting
base_model = model.get_base_model()
hook = base_model.lm_head.register_forward_pre_hook(cast_lm_head_input)
model.train()

# Setup optimizer
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], 
    lr=2e-5,
)

# Training loop
torch.xpu.reset_peak_memory_stats()
print(f"\n--- Training {NUM_STEPS} step(s) (seed={SEED}) ---")
t_start = time.perf_counter()
losses = []
step_times = []

for step in range(1, NUM_STEPS + 1):
    start = ((step - 1) * BATCH_SIZE) % len(train_ds)
    examples = train_ds[start:start+BATCH_SIZE]
    batch = make_batch(examples)
    optimizer.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss
    loss.backward()
    optimizer.step()
    torch.xpu.synchronize()
    losses.append(loss.item())
    step_times.append(time.perf_counter() - t_start)
    
    if step % EVAL_EVERY == 0:
        val_loss = evaluate_loss()
        elapsed = time.perf_counter() - t_start
        print(f"  step {step:4d}: train={loss.item():.4f} val={val_loss:.4f} time={elapsed:.0f}s")

# Final evaluation
print("\n--- Final evaluation ---")
final_val_loss = evaluate_loss()
final_preds = generate_predictions(model, n_samples=ACC_EVAL_SAMPLES)
final_acc = accuracy(final_preds)
peak_mem = torch.xpu.max_memory_allocated() / 1e9
total_time = time.perf_counter() - t_start
n_unknown = sum(1 for p in final_preds if p == "unknown")

# Save adapter
adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"Adapter saved: {adapter_dir}")

# Save result.json
result = {
    "run_id": f"qb4_7b_boundary_seed{SEED}",
    "method": "qlora",
    "precision": "nf4_4bit",
    "model": MODEL_ID,
    "task": "SST-2",
    "seed": SEED,
    "phase": "phase_3_boundary_probe",
    "lora_rank": 8,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj"],
    "num_steps": NUM_STEPS,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "learning_rate": 2e-5,
    "trainable_parameter_count": n_trainable,
    "total_model_parameters": n_total,
    "initial_train_loss": losses[0] if losses else None,
    "final_train_loss": losses[-1] if losses else None,
    "final_validation_loss": final_val_loss,
    "final_validation_accuracy": final_acc,
    "validation_samples": ACC_EVAL_SAMPLES,
    "unknown_predictions": n_unknown,
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

hook.remove()

print("\n" + "=" * 70)
print("PHASE 3 BOUNDARY PROBE RESULT")
print("=" * 70)
print(json.dumps(result, indent=2))
print(f"\nSaved to: {result_path}")
print("=" * 70)