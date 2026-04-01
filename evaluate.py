#!/usr/bin/env python3
"""Evaluate a trained context compression model."""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from context_compression import ContextCompressionEvaluator, load_model_for_inference
from context_compression.utils import extract_summary, get_default_config, load_config, load_jsonl


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Context Compression Model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data (JSON or JSONL)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to evaluate (-1 for all)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    
    return parser.parse_args()


def load_test_data(data_path: str):
    """Load JSON or JSONL evaluation data."""
    path = Path(data_path)

    if path.suffix == ".jsonl":
        return load_jsonl(data_path)
    if path.suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("samples", [])
    raise ValueError(f"Unsupported file format: {path.suffix}")


def main():
    """Run evaluation."""
    args = parse_args()

    print("=" * 60)
    print("Context Compression Evaluation")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config:
        config = load_config(args.config)
    else:
        config_path = Path(args.model_path) / "config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            config = get_default_config()

    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Output dir: {args.output_dir}")

    print("\nLoading model...")
    model, tokenizer = load_model_for_inference(args.model_path, config)

    print("\nLoading test data...")
    test_data = load_test_data(args.test_data)

    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]

    print(f"Test samples: {len(test_data)}")

    evaluator = ContextCompressionEvaluator(
        tokenizer=tokenizer,
        embedding_model=config.get("rewards", {}).get("embedding_model", "BAAI/bge-large-en-v1.5")
    )

    contexts = []
    predictions = []
    references = []

    print("\nGenerating summaries...")
    for item in tqdm(test_data, desc="Processing"):
        context = item.get("context", "")
        reference = item.get("gold_summary", "")

        prompt = f"""You are an expert at summarizing conversations. Your task is to create a concise summary of the following conversation context while preserving all important information.

Context:
{context}

Please provide a concise summary that captures the key points, decisions, and important information from the conversation. Format your response as:

<reasoning>
Briefly explain what key information needs to be preserved.
</reasoning>

<summary>
Your concise summary here.
</summary>

Summary:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        raw_prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        contexts.append(context)
        predictions.append(extract_summary(raw_prediction))
        references.append(reference)

    print("\nEvaluating...")
    metrics = evaluator.evaluate(contexts, predictions, references)

    print("\n" + str(metrics))

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics.to_dict(),
            "samples": [
                {
                    "context": c[:500] + "..." if len(c) > 500 else c,
                    "reference": r,
                    "prediction": p,
                }
                for c, r, p in zip(contexts[:10], references[:10], predictions[:10])
            ]
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_file}")

    detailed_file = output_dir / "detailed_results.jsonl"
    with open(detailed_file, "w", encoding="utf-8") as f:
        for c, r, p in zip(contexts, references, predictions):
            entry = {
                "context": c,
                "reference": r,
                "prediction": p,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Detailed results saved to {detailed_file}")


if __name__ == "__main__":
    main()
