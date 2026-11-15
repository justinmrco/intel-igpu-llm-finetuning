"""Model and LoRA setup for Phase 2."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_model_and_tokenizer(cfg):
    model_name = cfg["model"]["name"]
    revision = cfg["model"].get("revision", "main")

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
        num_labels=2,
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = tok.pad_token_id

    # Re-initialize classifier head with small std for stable training
    if hasattr(model, "score"):
        torch.nn.init.normal_(model.score.weight, mean=0.0, std=0.02)
        if model.score.bias is not None:
            torch.nn.init.zeros_(model.score.bias)

    return model, tok


def apply_lora(model, cfg, config_id):
    c = cfg["configs"][config_id]
    lora_cfg = LoraConfig(
        r=c["lora_rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=c["target_modules"],
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )
    return get_peft_model(model, lora_cfg)


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_trainable_params(model):
    """Return list of (name, param) for trainable params only."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def assert_base_frozen(model):
    """
    Ensure only LoRA params (and the classification head 'score') are trainable.
    """
    allowed_non_lora = ("lora_", "modules_to_save", "score")
    bad = []
    for n, p in model.named_parameters():
        if p.requires_grad and not any(k in n for k in allowed_non_lora):
            bad.append(n)
    if bad:
        raise AssertionError(f"Unexpected trainable params: {bad[:5]}")