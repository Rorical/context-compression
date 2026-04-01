#!/usr/bin/env python3
"""Run inference with a trained context compression model."""

import argparse
from pathlib import Path

from context_compression import load_model_for_inference
from context_compression.utils import extract_summary, get_default_config, load_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Context Compression Inference")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input file containing context"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output file"
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Direct context input (for quick testing)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max_summary_length",
        type=int,
        default=1024,
        help="Maximum summary length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    
    return parser.parse_args()


def generate_summary(
    model,
    tokenizer,
    context: str,
    max_summary_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a summary for the provided context."""
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
            max_new_tokens=max_summary_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    summary = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return summary.strip()


def interactive_mode(model, tokenizer, args):
    """Run interactive inference mode."""
    print("\n" + "=" * 60)
    print("Context Compression - Interactive Mode")
    print("=" * 60)
    print("Enter your context (press Ctrl+D or type 'END' on a new line to finish):")
    print("-" * 60)
    
    while True:
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
        except EOFError:
            pass
        
        if not lines:
            break
        
        context = "\n".join(lines)
        
        print("\n" + "-" * 60)
        print("Compressing...")
        print("-" * 60)
        
        output = generate_summary(
            model,
            tokenizer,
            context,
            args.max_summary_length,
            args.temperature,
            args.top_p,
        )
        
        summary = extract_summary(output)
        
        print("\nGenerated Output:")
        print("=" * 60)
        print(output)
        print("=" * 60)
        
        print("\nExtracted Summary:")
        print("=" * 60)
        print(summary)
        print("=" * 60)
        
        # 计算压缩比例
        context_tokens = len(tokenizer.encode(context))
        summary_tokens = len(tokenizer.encode(summary))
        compression_ratio = summary_tokens / context_tokens if context_tokens > 0 else 0
        
        print(f"\nCompression ratio: {compression_ratio:.2%}")
        print(f"Original: {context_tokens} tokens")
        print(f"Summary: {summary_tokens} tokens")
        print("\n" + "-" * 60)
        print("Enter next context (or Ctrl+C to exit):")


def main():
    """Run inference."""
    args = parse_args()

    print("=" * 60)
    print("Context Compression Inference")
    print("=" * 60)

    if args.config:
        config = load_config(args.config)
    else:
        config_path = Path(args.model_path) / "config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            config = get_default_config()

    print(f"Model path: {args.model_path}")

    print("\nLoading model...")
    model, tokenizer = load_model_for_inference(args.model_path, config)

    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.context:
        print("\nCompressing context...")
        output = generate_summary(
            model,
            tokenizer,
            args.context,
            args.max_summary_length,
            args.temperature,
            args.top_p,
        )
        
        summary = extract_summary(output)
        
        print("\nGenerated Output:")
        print("=" * 60)
        print(output)
        print("=" * 60)
        
        print("\nExtracted Summary:")
        print("=" * 60)
        print(summary)
        print("=" * 60)
        
        # 计算压缩比例
        context_tokens = len(tokenizer.encode(args.context))
        summary_tokens = len(tokenizer.encode(summary))
        compression_ratio = summary_tokens / context_tokens if context_tokens > 0 else 0
        
        print(f"\nCompression ratio: {compression_ratio:.2%}")
        
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"\nSummary saved to {args.output_file}")
    elif args.input_file:
        print(f"\nReading context from {args.input_file}...")
        with open(args.input_file, "r", encoding="utf-8") as f:
            context = f.read()

        print("Compressing context...")
        output = generate_summary(
            model,
            tokenizer,
            context,
            args.max_summary_length,
            args.temperature,
            args.top_p,
        )
        
        summary = extract_summary(output)
        
        print("\nGenerated Output:")
        print("=" * 60)
        print(output)
        print("=" * 60)
        
        print("\nExtracted Summary:")
        print("=" * 60)
        print(summary)
        print("=" * 60)
        
        # 计算压缩比例
        context_tokens = len(tokenizer.encode(context))
        summary_tokens = len(tokenizer.encode(summary))
        compression_ratio = summary_tokens / context_tokens if context_tokens > 0 else 0
        
        print(f"\nCompression ratio: {compression_ratio:.2%}")
        print(f"Original: {context_tokens} tokens")
        print(f"Summary: {summary_tokens} tokens")
        
        output_file = args.output_file or args.input_file + ".summary"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\nSummary saved to {output_file}")
    else:
        print("Error: Please provide --context, --input_file, or use --interactive mode")


if __name__ == "__main__":
    main()
