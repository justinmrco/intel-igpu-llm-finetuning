# Sources and redistribution

Core tools and methods used or discussed in this project:

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Fine-Tuning Language Models with Just Forward Passes (MeZO)](https://arxiv.org/abs/2305.17333)
- [GLUE benchmark](https://gluebenchmark.com/) and its SST-2 task
- [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) and [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [PyTorch](https://pytorch.org/), [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft) and [IPEX-LLM](https://github.com/intel/ipex-llm)

These links identify upstream work; they are not a systematic literature review or a current audit of each upstream license. Historical reference PDFs remain in the private preserved workspace rather than being redistributed here. The Phase 1 source names an Instruct variant, while main result records name the base 1.5B model; do not conflate those workflows.

This repository contains experiment code, recorded scalar results, diagnostics and derived figures. Base weights, adapters, tokenizer packages and dataset copies are not published. Downloads required for future training remain subject to the exact upstream model, dataset and library terms. Different Qwen variants need their own license review; do not assume one license covers all model sizes.

No blanket license has been selected for the author's code. Public visibility does not itself grant unrestricted reuse, nor does academic use automatically establish redistribution rights. If a software license is selected later, apply it only to material the author has authority to license.
