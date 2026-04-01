#!/usr/bin/env python3
"""Train the context compression model."""

import argparse
from pathlib import Path

from context_compression import (
    ContextCompressionTrainer,
    DataPipeline,
    get_config_for_gpu,
    get_default_config,
    load_config,
)
from context_compression.utils import get_gpu_info


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Context Compression Model")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    # 模型配置
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name or path"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Maximum sequence length"
    )
    
    # LoRA配置
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank"
    )
    
    # 训练配置
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps"
    )
    
    # 数据配置
    parser.add_argument(
        "--use_synthetic_data",
        action="store_true",
        help="Use synthetic data for testing"
    )
    parser.add_argument(
        "--synthetic_samples",
        type=int,
        default=100,
        help="Number of synthetic samples"
    )
    
    # 输出配置
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory"
    )
    
    # 其他配置
    parser.add_argument(
        "--gpu_memory_gb",
        type=float,
        default=None,
        help="GPU memory in GB (for auto config)"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    return parser.parse_args()


def update_config_from_args(config: dict, args: argparse.Namespace) -> dict:
    """Update configuration using CLI overrides."""
    if args.model_name:
        config["model"]["name"] = args.model_name
    
    if args.max_seq_length:
        config["model"]["max_seq_length"] = args.max_seq_length
    
    if args.lora_rank:
        config["lora"]["rank"] = args.lora_rank
        config["lora"]["alpha"] = args.lora_rank
    
    if args.num_train_epochs:
        config["training"]["num_train_epochs"] = args.num_train_epochs
    
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    
    return config


def prepare_data(config: dict, args: argparse.Namespace):
    """Load and preprocess the training and evaluation datasets."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipeline = DataPipeline(tokenizer=tokenizer)

    if args.use_synthetic_data:
        print(f"Generating {args.synthetic_samples} synthetic samples...")
        pipeline.generate_synthetic_data(
            num_samples=args.synthetic_samples,
            scenarios=["customer_service", "technical_discussion", "educational_tutoring"]
        )
    else:
        datasets_config = config.get("data", {}).get("real_datasets", [])
        if datasets_config:
            print("Loading real datasets...")
            pipeline.load_real_datasets(datasets_config)
        else:
            print("No datasets configured, using synthetic data...")
            pipeline.generate_synthetic_data(num_samples=100)

    print("Preprocessing data...")
    pipeline.preprocess(
        filter_by_length=True,
        min_tokens=100,
        max_tokens=config["model"]["max_seq_length"]
    )
    
    train_dataset = pipeline.create_hf_dataset(split="train")
    eval_dataset = pipeline.create_hf_dataset(split="eval")

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def main():
    """Run the training workflow."""
    args = parse_args()

    print("=" * 60)
    print("Context Compression Training")
    print("=" * 60)

    if Path(args.config).exists():
        print(f"Loading config from {args.config}")
        config = load_config(args.config)
    elif args.gpu_memory_gb:
        print(f"Using auto config for {args.gpu_memory_gb}GB GPU")
        config = get_config_for_gpu(args.gpu_memory_gb)
    else:
        print("Using default config")
        config = get_default_config()

    config = update_config_from_args(config, args)

    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        for device in gpu_info["devices"]:
            print(f"GPU {device['id']}: {device['name']} ({device['total_memory_gb']:.1f} GB)")
    else:
        print("Warning: No GPU available!")

    train_dataset, eval_dataset = prepare_data(config, args)

    trainer = ContextCompressionTrainer(config)

    print("Setting up model...")
    trainer.setup_model()

    print("Setting up trainer...")
    trainer.setup_trainer(train_dataset, eval_dataset)

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving model...")
    trainer.save_model()

    print("Evaluating model...")
    trainer.evaluate(eval_dataset)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {config['output']['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
