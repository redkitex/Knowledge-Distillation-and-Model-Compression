import time
import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the cleaning function and the prompt from the data module
from kd.data import prompt_ids
# Import the function to extract the number
from kd.answer_extraction import extract_last_number, gold_answer

@torch.no_grad()
def generate_batch(model, tokenizer, device, prompts, eos_ids, max_new_tokens, dtype):
    """
    Generates responses in batches using greedy decoding.
    If a batch doesn't fit in memory (OOM), it splits it in half and retries.
    """

    if not prompts:
        return []

    # Uses left padding so the model writes at the end of the tensor
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    width = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    
    for i, p in enumerate(prompts):
        input_ids[i, width - len(p):] = torch.tensor(p, device=device)
        attention_mask[i, width - len(p):] = 1

    try:
        with torch.autocast("cuda", dtype=dtype):
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # Greedy: deterministic evaluation
                pad_token_id=pad_id,
                use_cache=True,
            )
    except torch.cuda.OutOfMemoryError:
        if len(prompts) == 1:
            raise
        
        # Lifesaver: if the batch is too large, split it in half
        gc.collect()
        torch.cuda.empty_cache()
        mid = len(prompts) // 2
        return (generate_batch(model, tokenizer, device, prompts[:mid], eos_ids, max_new_tokens, dtype)
                + generate_batch(model, tokenizer, device, prompts[mid:], eos_ids, max_new_tokens, dtype))

    responses = []
    # We trim the padding that remains after the end-of-sequence token
    for row in out[:, width:].tolist():
        end = len(row)
        for j, tok in enumerate(row):
            if tok in eos_ids:
                end = j + 1
                break
        responses.append(row[:end])
    return responses

def evaluate_model(model_name_or_path, device, problems, batch_size, max_new_tokens, dtype, tokenizer_name=None):
    """
    Evaluates a model on a set of GSM8K problems.
    """
    if tokenizer_name is None:
        tokenizer_name = model_name_or_path

    print(f"\nLoading model {model_name_or_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Identify the tokens that indicate the model has finished responding
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    # In Qwen, <|im_end|> is often used to close messages
    eos_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    records = []
    t_start = time.time()

    for batch_start in range(0, len(problems), batch_size):
        batch = problems[batch_start: batch_start + batch_size]
        
        # Important: we use prompt_ids from kd.data (the same as in training)
        prompts = [prompt_ids(tokenizer, ex["question"]) for ex in batch]
        
        responses = generate_batch(model, tokenizer, device, prompts, eos_ids, max_new_tokens, dtype)

        for ex, resp_ids in zip(batch, responses):
            response_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            
            # We use our dedicated functions from kd.answer_extraction
            predicted = extract_last_number(response_text)
            expected = gold_answer(ex["answer"])
            
            is_correct = predicted is not None and abs(predicted - expected) < 1e-6
            truncated = (len(resp_ids) == 0) or (resp_ids[-1] not in eos_ids)
            
            records.append({
                "question": ex["question"],
                "response": response_text,
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct,
                "response_tokens": len(resp_ids),
                "truncated": truncated,
            })

    elapsed_total = time.time() - t_start
    correct = sum(r["correct"] for r in records)
    n = len(records)
    accuracy = correct / max(n, 1)

    summary = {
        "model": model_name_or_path,
        "n": n,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "mean_response_tokens": round(sum(r["response_tokens"] for r in records) / max(n, 1), 1),
        "truncated_fraction": round(sum(r["truncated"] for r in records) / max(n, 1), 4),
        "elapsed_seconds": round(elapsed_total, 1),
        "records": records,
    }

    # Clean up VRAM before returning the result (crucial for Kaggle)
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return summary