import argparse
import json
import yaml
import torch
import random
from pathlib import Path
from kd.data import load_test
from kd.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the GSM8K test set")
    parser.add_argument("--config", required=True, help="Path to the config.yaml file")
    parser.add_argument("--run", required=True, help="Name of the run")
    parser.add_argument("--harness", action="store_true", help="Recreates the 300-question headroom test")
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

    #We extract the arm and the seed from run_id
    if args.run.startswith("ref_"):
        arm = args.run
        seed = 0
    else:
        #Divides by the last "_s"
        arm_part, seed_part = args.run.rsplit("_s", 1)
        arm = arm_part
        seed = int(seed_part)

    # The first available GPU is used
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load the evaluation dataset
    test_set = load_test()

    if args.harness:
        # Recreates the exact shuffle from the original headroom notebook
        rng = random.Random(42)
        indexes = list(range(len(test_set)))
        rng.shuffle(indexes)
        test_set = [test_set[i] for i in indexes[:300]]
        print("HARNESS CHECK: evaluating random subset of 300 questions with seed 42.")
    else:
        # Limits the number of questions if specified in the config
        n_questions = cfg["eval"].get("n_questions")
        if n_questions is not None:
            test_set = test_set[:n_questions]

    batch_size = cfg["eval"].get("batch_size", 8)

    print(f"Starting evaluation for run: {args.run} with {len(test_set)} questions.")
    
    # Call the evaluation engine
    summary = evaluate_model(
        model_name_or_path=model_path,
        device=device,
        problems=test_set,
        batch_size=batch_size,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        dtype=getattr(torch, cfg["train"]["autocast_dtype"], torch.float16),
        tokenizer_name=cfg["student"], # The student tokenizer is used
        run_id=args.run,
        arm=arm,
        seed=seed
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