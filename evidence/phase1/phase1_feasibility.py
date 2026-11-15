"""
Phase 1: Feasibility Matrix (native PyTorch XPU, no IPEX-LLM)
Zeroth-Order (ZO) vs First-Order LoRA on Qwen2.5-1.5B (bf16) on Intel Arc XPU
"""

import gc
import time
import traceback
from datetime import datetime, timezone

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EPSILON = 1e-3
LR = 1e-4
OUTPUT_CSV = "feasibility_matrix_results.csv"

CONFIGURATIONS = [
    (1,  "ZO",    4, ["q_proj"],                      1, 512),
    (2,  "ZO",    4, ["q_proj", "k_proj", "v_proj"],  1, 512),
    (3,  "ZO",    8, ["q_proj"],                      1, 512),
    (4,  "ZO",    8, ["q_proj", "k_proj", "v_proj"],  1, 512),
    (5,  "ZO",    4, ["q_proj", "k_proj", "v_proj"],  2, 256),
    (6,  "ZO",    8, ["q_proj", "k_proj", "v_proj"],  2, 256),
    (7,  "QLoRA", 4, ["q_proj"],                      1, 512),
    (8,  "QLoRA", 4, ["q_proj", "k_proj", "v_proj"],  1, 512),
    (9,  "QLoRA", 8, ["q_proj"],                      1, 512),
    (10, "QLoRA", 8, ["q_proj", "k_proj", "v_proj"],  1, 512),
    (11, "QLoRA", 4, ["q_proj", "k_proj", "v_proj"],  2, 256),
    (12, "QLoRA", 8, ["q_proj", "k_proj", "v_proj"],  2, 256),
]

def validate_targets(model, target_modules):
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    missing = [t for t in target_modules if t not in present]
    if missing:
        raise RuntimeError(
            f"Target modules not found: {missing}\n"
            f"Available proj-like: {sorted(n for n in present if 'proj' in n.lower())}"
        )

def is_oom_error(e):
    msg = str(e).lower()
    return "out of memory" in msg or "outofmemory" in msg or "xpu out of memory" in msg

def xpu_peak_mem_gb():
    try:
        return torch.xpu.max_memory_allocated() / 1e9
    except Exception:
        return float("nan")

def clear_xpu():
    gc.collect()
    try:
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()
    except Exception:
        pass

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lora_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]

@torch.no_grad()
def zo_perturb(lora_params, direction, eps):
    for (name, p), z in zip(lora_params, direction):
        p.add_(z, alpha=eps)

def sample_zo_direction(lora_params):
    return [torch.randn_like(p) for _, p in lora_params]

def make_batch(tokenizer, batch_size, seq_length, device="xpu"):
    text = "This is a sample text for fine-tuning a language model on a downstream task."
    enc = tokenizer(
        [text] * batch_size,
        return_tensors="pt",
        padding="max_length",
        max_length=seq_length,
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def forward_loss(model, inputs):
    out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["labels"],
    )
    return out.loss

def zo_step(model, inputs, lora_params, epsilon=EPSILON):
    z = sample_zo_direction(lora_params)
    zo_perturb(lora_params, z, +epsilon)
    loss_plus = forward_loss(model, inputs).item()
    zo_perturb(lora_params, z, -2 * epsilon)
    loss_minus = forward_loss(model, inputs).item()
    zo_perturb(lora_params, z, +epsilon)
    coeff = (loss_plus - loss_minus) / (2 * epsilon)
    with torch.no_grad():
        for (name, p), zi in zip(lora_params, z):
            p.add_(zi, alpha=-LR * coeff)
    return loss_plus, loss_minus, coeff

def qlora_step(model, inputs, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = forward_loss(model, inputs)
    loss.backward()
    optimizer.step()
    return loss.item()

def load_base_model():
    print(f"[load] Loading {MODEL_NAME} in bf16 (native PyTorch XPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )
    model.config.use_cache = False
    model = model.to("xpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[load] Done in {time.time()-t0:.1f}s.")
    return model, tokenizer

def run_config(run_id, method, rank, target_modules, batch_size, seq_length,
               base_model, tokenizer):
    print(f"\n{'='*70}")
    print(f"[run {run_id}] method={method} rank={rank} "
          f"modules={target_modules} bs={batch_size} seq={seq_length}")
    print(f"{'='*70}")

    clear_xpu()
    record = {
        "run_id": run_id,
        "model_name": "Qwen2.5-1.5B bf16 (native XPU)",
        "optimization_method": method,
        "lora_rank": f"r={rank}",
        "target_modules": ",".join(target_modules),
        "batch_size": batch_size,
        "seq_length": seq_length,
        "oom_occurred": False,
        "peak_vram_gb": float("nan"),
        "single_step_time_sec": float("nan"),
        "trainable_params": 0,
        "success_status": "UNKNOWN",
        "datetime_run": datetime.now(timezone.utc).isoformat(),
        "notes": "",
    }

    model_with_lora = None
    try:
        validate_targets(base_model, target_modules)
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model_with_lora = get_peft_model(base_model, lora_cfg)
        model_with_lora = model_with_lora.to("xpu")
        model_with_lora.train()

        record["trainable_params"] = count_trainable(model_with_lora)
        print(f"[run {run_id}] trainable params: {record['trainable_params']:,}")

        inputs = make_batch(tokenizer, batch_size, seq_length)
        lora_params = get_lora_params(model_with_lora)
        optimizer = torch.optim.AdamW([p for _, p in lora_params], lr=LR)

        try:
            if method == "QLoRA":
                _ = qlora_step(model_with_lora, inputs, optimizer)
            else:
                _ = zo_step(model_with_lora, inputs, lora_params)
            torch.xpu.synchronize()
        except Exception as e:
            if is_oom_error(e):
                raise
            print(f"[run {run_id}] warmup failed (non-OOM): {e}")

        clear_xpu()
        torch.xpu.synchronize()
        t0 = time.time()

        if method == "QLoRA":
            qlora_step(model_with_lora, inputs, optimizer)
        else:
            zo_step(model_with_lora, inputs, lora_params)

        torch.xpu.synchronize()
        step_time = time.time() - t0

        record["single_step_time_sec"] = round(step_time, 3)
        record["peak_vram_gb"] = round(xpu_peak_mem_gb(), 3)
        record["success_status"] = "PASS"
        record["notes"] = "Completed without error"
        print(f"[run {run_id}] PASS  time={step_time:.2f}s  peak={record['peak_vram_gb']:.2f} GB")

    except Exception as e:
        if is_oom_error(e):
            record["oom_occurred"] = True
            record["success_status"] = "OOM"
            record["peak_vram_gb"] = round(xpu_peak_mem_gb(), 3)
            record["notes"] = f"OOM: {str(e)[:120]}"
            print(f"[run {run_id}] OOM at peak {record['peak_vram_gb']:.2f} GB")
        else:
            record["success_status"] = f"ERROR: {type(e).__name__}"
            record["notes"] = str(e)[:200]
            record["peak_vram_gb"] = round(xpu_peak_mem_gb(), 3)
            print(f"[run {run_id}] ERROR: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
    finally:
        if model_with_lora is not None:
            del model_with_lora
        clear_xpu()
    return record

def main():
    if not torch.xpu.is_available():
        raise RuntimeError("XPU not available.")
    print(f"XPU available: {torch.xpu.get_device_name(0)}")
    base_model, tokenizer = load_base_model()
    results = []
    for (run_id, method, rank, modules, bs, seq) in CONFIGURATIONS:
        try:
            rec = run_config(run_id, method, rank, modules, bs, seq, base_model, tokenizer)
        except Exception as e:
            rec = {
                "run_id": run_id, "model_name": "Qwen2.5-1.5B bf16 (native XPU)",
                "optimization_method": method, "lora_rank": f"r={rank}",
                "target_modules": ",".join(modules), "batch_size": bs,
                "seq_length": seq, "oom_occurred": False,
                "peak_vram_gb": round(xpu_peak_mem_gb(), 3),
                "single_step_time_sec": float("nan"), "trainable_params": 0,
                "success_status": f"FATAL: {type(e).__name__}",
                "datetime_run": datetime.now(timezone.utc).isoformat(),
                "notes": str(e)[:200],
            }
            print(f"[run {run_id}] FATAL: {e}")
        results.append(rec)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    df = pd.DataFrame(results)
    print(f"\nSaved {OUTPUT_CSV}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()