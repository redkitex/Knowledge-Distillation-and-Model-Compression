import argparse
import json
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from kd.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the GSM8K test set")
    parser.add_argument("--config", required=True, help="Path to the config.yaml file")
    parser.add_argument("--run", required=True, help="Name of the run")
    args = parser.parse_args()

    # Load the centralized configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Determine which model we are going to evaluate
    if args.run == "ref_teacher":
        model_path = cfg["teacher"]
    elif args.run == "ref_student0":
        model_path = cfg["student"]
    else:
        # If it's a trained run, it will load the weights from the results folder
        model_path = f"results/runs/{args.run}/checkpoint"

    # The first available GPU is used
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load the evaluation dataset
    test_set = load_dataset("openai/gsm8k", "main", split="test")

    print(f"Starting evaluation for run: {args.run}")
    
    # Call the evaluation engine
    summary = evaluate_model(
        model_name_or_path=model_path,
        device=device,
        problems=test_set,
        batch_size=8,  # Default batch; it will be automatically split if there is OOM
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        dtype=torch.float16,
        tokenizer_name=cfg["student"] # The student tokenizer is used
    )

    # Save the results
    out_dir = Path(f"results/runs/{args.run}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions.jsonl with each question and answer
    predictions_file = out_dir / "predictions.jsonl"
    with open(predictions_file, "w", encoding="utf-8") as f:
        for rec in summary["records"]:
            f.write(json.dumps(rec) + "\n")
            
    # Save a quick summary
    sum_copy = summary.copy()
    del sum_copy["records"]
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(sum_copy, f, indent=2)

    print(f"\nEvaluation completed.")
    print(f"Accuracy achieved: {summary['accuracy']:.2%}")
    print(f"Truncated answers: {summary['truncated_fraction']:.2%}")
    print(f"Results saved in {out_dir}/")

if __name__ == "__main__":
    main()