"""Evaluate a model on GSM8K test problems: accuracy, time per answer and memory use.

Usage:
    python eval_gsm8k.py --model Qwen/Qwen2.5-0.5B-Instruct --n 50
    python eval_gsm8k.py --model Qwen/Qwen3-0.6B --n 50
    python eval_gsm8k.py --model Qwen/Qwen3-1.7B --n 50 --thinking

Models download automatically on first use.
Qwen3 models have a "thinking" mode; it is OFF by default here (shorter answers,
fair comparison with Qwen2.5). Use --thinking to turn it on.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = "Solve the problem step by step. Put the final number at the end."


def extract_last_number(text):
    """Return the last number in the text as a float, or None if there is none."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def gold_answer(answer_field):
    """GSM8K answers end with '#### <number>'; return that number."""
    return extract_last_number(answer_field.split("####")[-1])


def gb(num_bytes):
    return round(num_bytes / 1024**3, 2)


def ram_used_gb():
    """System RAM used by this Python process (resident set size), read from /proc."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024**2, 2)  # value is in kB
    return None


_device_peak = 0.0


def memory_snapshot():
    """Current memory use.

    vram_total_used_gb: whole-GPU usage, same as nvidia-smi (includes CUDA context
    and other programs). The rest are PyTorch's own numbers, plus process RAM.
    """
    global _device_peak
    free, total = torch.cuda.mem_get_info()
    device_used = gb(total - free)
    _device_peak = max(_device_peak, device_used)
    return {
        "vram_total_used_gb": device_used,
        "vram_total_peak_gb": _device_peak,
        "vram_allocated_gb": gb(torch.cuda.memory_allocated()),
        "vram_reserved_gb": gb(torch.cuda.memory_reserved()),
        "vram_peak_gb": gb(torch.cuda.max_memory_allocated()),
        "ram_gb": ram_used_gb(),
    }


def print_memory(label):
    m = memory_snapshot()
    print(
        f"[memory] {label}: VRAM total used={m['vram_total_used_gb']} GB | PyTorch allocated={m['vram_allocated_gb']} GB, "
        f"reserved={m['vram_reserved_gb']} GB, peak={m['vram_peak_gb']} GB | RAM={m['ram_gb']} GB"
    )
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--n", type=int, default=50, help="number of test problems")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode")
    args = parser.parse_args()

    torch.cuda.reset_peak_memory_stats()
    print_memory("start")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    after_load = print_memory("after model load")

    test_set = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))

    records = []
    correct = 0
    total_time = 0.0

    for i, example in enumerate(test_set):
        messages = [{"role": "user", "content": f"{example['question']}\n\n{INSTRUCTION}"}]
        # enable_thinking is read by Qwen3 chat templates and ignored by Qwen2.5
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=args.thinking,
        ).to("cuda")
        input_ids = inputs["input_ids"]

        start = time.time()
        with torch.no_grad():
            # Greedy decoding: deterministic, so runs are repeatable
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        elapsed = time.time() - start
        total_time += elapsed

        new_tokens = output[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        hit_limit = len(new_tokens) >= args.max_new_tokens

        predicted = extract_last_number(response)
        expected = gold_answer(example["answer"])
        is_correct = predicted is not None and abs(predicted - expected) < 1e-6
        correct += is_correct

        mem = memory_snapshot()
        records.append({
            "index": i,
            "question": example["question"],
            "response": response,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "seconds": round(elapsed, 2),
            "new_tokens": len(new_tokens),
            "hit_token_limit": hit_limit,
            **mem,
        })
        limit_flag = " CUT OFF" if hit_limit else ""
        print(
            f"[{i + 1}/{args.n}] correct={is_correct} time={elapsed:.1f}s tokens={len(new_tokens)}{limit_flag} "
            f"| VRAM total={mem['vram_total_used_gb']} GB (peak {mem['vram_total_peak_gb']}) RAM={mem['ram_gb']} GB"
        )

    final = print_memory("end")

    summary = {
        "model": args.model,
        "thinking": args.thinking,
        "n": args.n,
        "accuracy": correct / args.n,
        "avg_seconds_per_answer": round(total_time / args.n, 2),
        "answers_cut_off": sum(r["hit_token_limit"] for r in records),
        "max_new_tokens": args.max_new_tokens,
        "vram_total_after_load_gb": after_load["vram_total_used_gb"],
        "vram_total_peak_gb": final["vram_total_peak_gb"],
        "vram_pytorch_peak_gb": final["vram_peak_gb"],
        "ram_peak_gb": max(r["ram_gb"] for r in records),
    }
    print("\n" + json.dumps(summary, indent=2))

    # Save summary and every answer, so wrong gradings can be inspected later
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    suffix = "_thinking" if args.thinking else ""
    out_file = out_dir / f"eval_{args.model.split('/')[-1]}{suffix}_n{args.n}.json"
    out_file.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
