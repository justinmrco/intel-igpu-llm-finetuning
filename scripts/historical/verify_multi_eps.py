"""Multi-direction x multi-epsilon verification.

20 independent directions, 3 eps values, fixed batch, fixed snapshot.
Reports mean/median loss change, success rate, and g_hat stability.

Batch construction matches diagnose_zo_fixed_batch.py:
    set_seed(SEED) -> load_model_and_tokenizer -> apply_lora
    -> make_loader -> next(iter(loader))
so the baseline loss should be ~0.8751.
"""
import statistics
import torch
import torch.nn.functional as F
import yaml

from src.model_setup import (
    load_model_and_tokenizer, apply_lora, get_trainable_params,
)
from src.train import make_loader, set_seed

CONFIG_PATH = "configs/phase2_config.yaml"
CONFIG_ID = "C1"
SEED = 42
NUM_DIRECTIONS = 20
EPS_LIST = [1e-3, 3e-4, 1e-4]
LR_FIXED = 1e-5


def fp32_ce(out, labels):
    return F.cross_entropy(out.logits.float(), labels)


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # CRITICAL: set_seed BEFORE model loading, matching diagnose_zo_fixed_batch.py.
    set_seed(SEED)
    device = "xpu" if torch.xpu.is_available() else "cpu"

    print("=" * 80)
    print("MULTI-DIRECTION x MULTI-EPSILON VERIFICATION (fixed batch)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Directions: {NUM_DIRECTIONS}")
    print(f"Epsilons: {EPS_LIST}")
    print(f"Fixed LR: {LR_FIXED}")

    model, tok = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg, CONFIG_ID)
    model.to(device)
    model.eval()

    trainable = get_trainable_params(model)

    loader = make_loader(cfg, tok, "train", SEED)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    def loss_on_current():
        with torch.no_grad():
            out = model(**batch)
            return fp32_ce(out, batch["labels"]).item()

    originals = {n: p.detach().clone() for n, p in trainable}
    baseline = loss_on_current()
    print(f"\nBaseline loss f(theta): {baseline:.10f}")
    if abs(baseline - 0.8751) < 0.01:
        print("[OK] Baseline matches diagnose_zo_fixed_batch.py (~0.8751)")
    else:
        print(f"[NOTE] Baseline differs from 0.8751 (got {baseline:.6f}).")

    all_zs = []
    for d_idx in range(NUM_DIRECTIONS):
        rng = torch.Generator(device=device)
        rng.manual_seed(SEED * 1000 + d_idx)
        zs = {}
        for n, p in trainable:
            zs[n] = torch.randn(
                p.shape, generator=rng, device=p.device, dtype=p.dtype
            )
        all_zs.append(zs)

    for eps in EPS_LIST:
        print(f"\n{'='*80}")
        print(f"EPS = {eps:.0e}")
        print(f"{'='*80}")
        print(f"{'dir':>5} {'g_hat':>12} {'f_after':>15} {'delta':>15} {'status':>10}")
        print("-" * 70)

        deltas = []
        n_decrease = 0
        n_indistinguishable = 0
        n_increase = 0

        for d_idx, zs in enumerate(all_zs):
            with torch.no_grad():
                for n, p in trainable:
                    p.data.copy_(originals[n] + eps * zs[n])
            lp = loss_on_current()

            with torch.no_grad():
                for n, p in trainable:
                    p.data.copy_(originals[n] - eps * zs[n])
            lm = loss_on_current()

            with torch.no_grad():
                for n, p in trainable:
                    p.data.copy_(originals[n])

            g_hat = (lp - lm) / (2 * eps)

            with torch.no_grad():
                for n, p in trainable:
                    g = g_hat * zs[n]
                    p.data.add_(g, alpha=-LR_FIXED)

            f_after = loss_on_current()
            delta = f_after - baseline

            with torch.no_grad():
                for n, p in trainable:
                    p.data.copy_(originals[n])

            deltas.append(delta)

            if abs(delta) < 1e-7:
                status = "FLAT"
                n_indistinguishable += 1
            elif delta < 0:
                status = "DECREASE"
                n_decrease += 1
            else:
                status = "INCREASE"
                n_increase += 1

            print(f"{d_idx:>5} {g_hat:>+12.4f} {f_after:>15.8f} "
                  f"{delta:>+15.8f} {status:>10}")

        print(f"\n--- Summary (eps={eps:.0e}) ---")
        print(f"Mean delta:   {statistics.mean(deltas):+.8f}")
        print(f"Median delta: {statistics.median(deltas):+.8f}")
        print(f"Std delta:    {statistics.stdev(deltas):.8f}")
        print(f"Min delta:    {min(deltas):+.8f}")
        print(f"Max delta:    {max(deltas):+.8f}")
        print(f"Decrease: {n_decrease}/{NUM_DIRECTIONS} "
              f"({100*n_decrease/NUM_DIRECTIONS:.0f}%)")
        print(f"Increase: {n_increase}/{NUM_DIRECTIONS} "
              f"({100*n_increase/NUM_DIRECTIONS:.0f}%)")
        print(f"Flat:     {n_indistinguishable}/{NUM_DIRECTIONS} "
              f"({100*n_indistinguishable/NUM_DIRECTIONS:.0f}%)")

    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()