
"""Phase 2 training loop for ZO vs FO LoRA.

Adds:
- Initial/final validation prediction diagnostics.
- Prediction-change rate between initial and final models.
- Saving the final PEFT adapter and tokenizer.
- Structured diagnostic output in prediction_diagnostics.json.
"""

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from src.model_setup import (
    load_model_and_tokenizer,
    apply_lora,
    count_trainable,
    get_trainable_params,
    assert_base_frozen,
)
from src.zo_optimizer import ZOOptimizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def tokenize(ds, tok, max_len):
    def _fn(ex):
        return tok(
            ex["sentence"],
            truncation=True,
            max_length=max_len,
        )

    return ds.map(_fn, batched=True)


def make_loader(cfg, tok, split, seed):
    ds = load_dataset(
        cfg["task"]["dataset"],
        cfg["task"]["subset"],
        split=split,
    )
    ds = tokenize(ds, tok, cfg["model"]["max_seq_length"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    collator = DataCollatorWithPadding(tok)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(split == "train"),
        collate_fn=collator,
        generator=generator,
    )


def fp32_ce_loss(out, labels):
    return F.cross_entropy(out.logits.float(), labels)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Return mean cross-entropy loss and accuracy."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        batch = {
            k: v.to(device) for k, v in batch.items()
        }

        outputs = model(**batch)
        loss = fp32_ce_loss(outputs, batch["labels"])

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size

        predictions = outputs.logits.float().argmax(dim=-1)
        correct += (
            predictions == batch["labels"]
        ).sum().item()
        total += batch_size

    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
    )


@torch.no_grad()
def prediction_diagnostics(model, loader, device):
    """Measure prediction distribution and retain predictions for comparison."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    predicted_counts = None
    true_counts = None
    probability_sum = None
    margin_sum = 0.0
    all_predictions = []

    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items()
        }

        outputs = model(**batch)
        logits = outputs.logits.float()
        labels = batch["labels"]

        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        loss = F.cross_entropy(
            logits,
            labels,
            reduction="sum",
        )

        num_classes = logits.shape[-1]

        if predicted_counts is None:
            predicted_counts = torch.zeros(
                num_classes, dtype=torch.long
            )
            true_counts = torch.zeros(
                num_classes, dtype=torch.long
            )
            probability_sum = torch.zeros(
                num_classes, dtype=torch.float64
            )

        predicted_counts += torch.bincount(
            predictions.cpu(),
            minlength=num_classes,
        )
        true_counts += torch.bincount(
            labels.cpu(),
            minlength=num_classes,
        )
        probability_sum += (
            probabilities.double().sum(dim=0).cpu()
        )

        margin_sum += (
            logits[:, 1] - logits[:, 0]
        ).sum().item()

        total_loss += loss.item()
        total_correct += (
            predictions == labels
        ).sum().item()
        total_examples += labels.numel()

        all_predictions.extend(predictions.cpu().tolist())

    if total_examples == 0:
        raise ValueError("The evaluation loader contains no examples.")

    return {
        "examples": total_examples,
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "predicted_class_counts": predicted_counts.tolist(),
        "true_class_counts": true_counts.tolist(),
        "mean_class_probabilities": (
            probability_sum / total_examples
        ).tolist(),
        "mean_logit_margin_class1_minus_class0": (
            margin_sum / total_examples
        ),
        "predictions": all_predictions,
    }


def lr_lambda_factory(warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)

        progress = (
            (step - warmup_steps)
            / max(total_steps - warmup_steps, 1)
        )

        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return fn


def train_one(
    cfg,
    config_id,
    method,
    seed,
    run_dir,
    max_steps=None,
    phase="pilot",
):
    set_seed(seed)
    device = "xpu" if torch.xpu.is_available() else "cpu"

    print(
        f"[{config_id}/{method}/seed{seed}] "
        "loading model..."
    )
    print(f"[device] {device}")

    model, tok = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg, config_id)
    model.to(device)
    model.train()

    assert_base_frozen(model)
    trainable, total = count_trainable(model)

    print(
        f"[{config_id}/{method}/seed{seed}] "
        f"trainable={trainable:,} total={total:,}"
    )

    train_loader = make_loader(
        cfg, tok, "train", seed
    )
    val_loader = make_loader(
        cfg, tok, "validation", seed
    )

    # Fixed, deterministic training-split probe.
    probe_ds = load_dataset(
        cfg["task"]["dataset"],
        cfg["task"]["subset"],
        split="train",
    )
    probe_ds = tokenize(
        probe_ds,
        tok,
        cfg["model"]["max_seq_length"],
    )
    probe_ds = probe_ds.select(
        range(min(128, len(probe_ds)))
    )
    probe_ds = probe_ds.rename_column(
        "label", "labels"
    )
    probe_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    probe_loader = DataLoader(
        probe_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tok),
    )

    steps = (
        max_steps
        if max_steps is not None
        else cfg["training"]["max_steps"]
    )
    target = cfg["training"]["target_loss"]
    eval_every = cfg["training"]["eval_every"]
    target_consec = cfg["training"].get(
        "target_consecutive_steps", 3
    )

    trainable_params = get_trainable_params(model)
    param_list = [p for _, p in trainable_params]

    if method == "fo":
        method_cfg = cfg["methods"]["fo"]

        opt = torch.optim.AdamW(
            param_list,
            lr=method_cfg["lr"],
            weight_decay=method_cfg["weight_decay"],
        )

        warmup = method_cfg.get("warmup_steps", 0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda_factory(warmup, steps),
        )
        use_zo = False

    else:
        method_cfg = cfg["methods"]["zo"]

        opt = ZOOptimizer(
            trainable_params,
            lr=method_cfg["lr"],
            eps=method_cfg["eps"],
            momentum=method_cfg["momentum"],
            normalize_z=method_cfg.get(
                "normalize_z", True
            ),
            grad_clip=method_cfg.get(
                "grad_clip", 10.0
            ),
        )

        opt.set_seed(seed * 1000 + 1)
        sched = None
        use_zo = True

    # Output files.
    csv_path = os.path.join(run_dir, "train_loss.csv")
    probe_csv_path = os.path.join(
        run_dir, "probe_loss.csv"
    )

    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    probe_csv = open(
        probe_csv_path, "w", newline="", encoding="utf-8"
    )

    writer = csv.writer(csv_f)
    writer.writerow([
        "step",
        "train_loss",
        "elapsed_sec",
        "peak_mem_gb",
        "lr",
    ])

    probe_writer = csv.writer(probe_csv)
    probe_writer.writerow([
        "step",
        "probe_loss",
        "probe_accuracy",
    ])

    if torch.xpu.is_available():
        torch.xpu.reset_peak_memory_stats()

    # Capture initial validation predictions before training.
    print("[diagnostics] evaluating initial model...")
    initial_diag = prediction_diagnostics(
        model, val_loader, device
    )

    print(
        "[initial_validation] "
        f"loss={initial_diag['loss']:.4f} "
        f"accuracy={initial_diag['accuracy']:.4f} "
        f"predicted_counts="
        f"{initial_diag['predicted_class_counts']} "
        f"mean_probs="
        f"{[round(x, 4) for x in initial_diag['mean_class_probabilities']]}"
    )

    initial_predictions = initial_diag.pop("predictions")

    it = iter(train_loader)
    step = 0
    target_step = None
    consecutive_below = 0
    t0 = time.time()
    val_log = []
    train_loss_current = None

    # Probe at step 0, before any updates.
    probe_loss, probe_acc = evaluate(
        model, probe_loader, device
    )
    probe_writer.writerow([
        0, probe_loss, probe_acc
    ])
    probe_csv.flush()

    print(
        f"[probe] step=0 loss={probe_loss:.4f} "
        f"accuracy={probe_acc:.4f}"
    )

    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        batch = {
            k: v.to(device) for k, v in batch.items()
        }

        if use_zo:
            model.eval()

            try:
                def loss_fn():
                    with torch.no_grad():
                        outputs = model(**batch)
                        return fp32_ce_loss(
                            outputs, batch["labels"]
                        )

                lp, lm = opt.step(loss_fn)
                train_loss_current = loss_fn().item()

                # Detailed ZO diagnostic for the first five updates.
                if step < 5:
                    diag = opt.last_diagnostics

                    print(
                        f"[zo_diag] iter={step + 1} "
                        f"loss_plus={lp:.6f} "
                        f"loss_minus={lm:.6f} "
                        f"raw_proj_grad="
                        f"{diag['raw_proj_grad']:.6e} "
                        f"clipped_proj_grad="
                        f"{diag['clipped_proj_grad']:.6e} "
                        f"was_clipped="
                        f"{diag['was_clipped']} "
                        f"update_norm="
                        f"{diag['update_norm']:.6e} "
                        f"post_update_loss="
                        f"{train_loss_current:.6f}"
                    )

            finally:
                model.train()

        else:
            model.train()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                param_list,
                cfg["training"]["grad_clip"],
            )

            opt.step()
            opt.zero_grad()

            if sched is not None:
                sched.step()

            train_loss_current = loss.item()

        step += 1

        # Probe evaluation at steps 5, 10, and 20.
        if use_zo and step in (5, 10, 20):
            probe_loss, probe_acc = evaluate(
                model, probe_loader, device
            )

            probe_writer.writerow([
                step, probe_loss, probe_acc
            ])
            probe_csv.flush()

            print(
                f"[probe] step={step} "
                f"loss={probe_loss:.4f} "
                f"accuracy={probe_acc:.4f}"
            )

        elapsed = time.time() - t0

        peak_mem = (
            torch.xpu.max_memory_allocated() / 1e9
            if torch.xpu.is_available()
            else 0.0
        )

        current_lr = (
            opt.param_groups[0]["lr"]
            if not use_zo
            else opt.lr
        )

        writer.writerow([
            step,
            train_loss_current,
            elapsed,
            peak_mem,
            current_lr,
        ])
        csv_f.flush()

        if train_loss_current < target:
            consecutive_below += 1

            if (
                target_step is None
                and consecutive_below >= target_consec
            ):
                target_step = step - target_consec + 1
                print(
                    f"  -> target {target} reached "
                    f"at step {target_step}"
                )
        else:
            consecutive_below = 0

        if step % eval_every == 0:
            val_loss, val_acc = evaluate(
                model, val_loader, device
            )

            val_log.append({
                "step": step,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss_current": train_loss_current,
            })

            print(
                f"  step {step}: "
                f"train={train_loss_current:.4f} "
                f"val={val_loss:.4f} "
                f"acc={val_acc:.4f} "
                f"lr={current_lr:.2e}"
            )

    total_elapsed = time.time() - t0

    # Full final validation evaluation and prediction diagnostics.
    print("[diagnostics] evaluating final model...")
    final_diag = prediction_diagnostics(
        model, val_loader, device
    )
    final_predictions = final_diag.pop("predictions")

    prediction_changes = sum(
        before != after
        for before, after in zip(
            initial_predictions, final_predictions
        )
    )

    prediction_change_rate = (
        prediction_changes / len(initial_predictions)
        if initial_predictions
        else 0.0
    )

    peak_mem = (
        torch.xpu.max_memory_allocated() / 1e9
        if torch.xpu.is_available()
        else 0.0
    )

    # Save the final PEFT adapter and tokenizer.
    adapter_dir = os.path.join(run_dir, "final_adapter")
    print(f"[save] saving adapter to: {adapter_dir}")

    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    # Persist diagnostic results without dumping thousands of predictions.
    diagnostics = {
        "initial_validation": initial_diag,
        "final_validation": final_diag,
        "prediction_comparison": {
            "examples_compared": len(initial_predictions),
            "predictions_changed": prediction_changes,
            "prediction_change_rate": prediction_change_rate,
        },
    }

    diagnostics_path = os.path.join(
        run_dir, "prediction_diagnostics.json"
    )

    with open(
        diagnostics_path, "w", encoding="utf-8"
    ) as f:
        json.dump(diagnostics, f, indent=2)

    csv_f.close()
    probe_csv.close()

    result = {
        "run_id": f"{method}_{config_id}_seed{seed}",
        "phase": phase,
        "config_id": config_id,
        "method": method,
        "seed": seed,
        "trainable_parameter_count": trainable,
        "total_model_parameters": total,
        "steps_completed": step,
        "steps_to_target": target_step,
        "target_reached": target_step is not None,
        "target_training_loss": target,
        "target_consecutive_steps": target_consec,
        "final_training_loss": train_loss_current,
        "final_validation_loss": final_diag["loss"],
        "final_validation_metric": final_diag["accuracy"],
        "initial_validation_loss": initial_diag["loss"],
        "initial_validation_accuracy": initial_diag["accuracy"],
        "prediction_change_rate": prediction_change_rate,
        "predictions_changed": prediction_changes,
        "peak_gpu_memory_gb": peak_mem,
        "total_elapsed_time_sec": total_elapsed,
        "mean_step_time_sec": (
            total_elapsed / step if step else None
        ),
        "validation_log": val_log,
        "adapter_dir": adapter_dir,
        "prediction_diagnostics_file": diagnostics_path,
        "status": "completed",
    }

    with open(
        os.path.join(run_dir, "result.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    print("[final_validation]")
    print(
        f"  loss: {final_diag['loss']:.6f}\n"
        f"  accuracy: {final_diag['accuracy']:.6f}\n"
        f"  predicted class counts: "
        f"{final_diag['predicted_class_counts']}\n"
        f"  true class counts: "
        f"{final_diag['true_class_counts']}\n"
        f"  mean class probabilities: "
        f"{[round(x, 6) for x in final_diag['mean_class_probabilities']]}\n"
        f"  mean logit margin (class 1 - class 0): "
        f"{final_diag['mean_logit_margin_class1_minus_class0']:.6f}\n"
        f"  prediction changes: "
        f"{prediction_changes}/{len(initial_predictions)} "
        f"({prediction_change_rate:.2%})"
    )

    print(json.dumps(
        {
            k: v
            for k, v in result.items()
            if k != "validation_log"
        },
        indent=2,
    ))

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_id",
        required=True,
        choices=["C1", "C2", "C3", "C4"],
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["zo", "fo"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--phase",
        default="pilot",
        choices=["pilot", "formal"],
    )
    parser.add_argument(
        "--out_dir",
        default="results/pilot",
    )

    args = parser.parse_args()

    with open(
        "configs/phase2_config.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    run_dir = os.path.join(
        args.out_dir,
        f"{args.method}_{args.config_id}_seed{args.seed}",
    )
    os.makedirs(run_dir, exist_ok=True)

    train_one(
        cfg,
        args.config_id,
        args.method,
        args.seed,
        run_dir,
        max_steps=args.max_steps,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
