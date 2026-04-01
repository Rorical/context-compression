"""
Context Compression - Utility Functions
工具函数模块
"""

import os
import re
import json
import yaml
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np


# =============================================================================
# Logging Utilities
# =============================================================================

def setup_logging(
    log_dir: str = "./logs",
    experiment_name: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """设置日志记录"""
    
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成实验名称
    if experiment_name is None:
        from datetime import datetime
        experiment_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 配置日志
    log_file = Path(log_dir) / f"{experiment_name}.log"
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("ContextCompression")
    logger.info(f"Logging initialized: {log_file}")
    
    return logger


# =============================================================================
# Configuration Utilities
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件 (YAML或JSON)"""
    
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """保存配置文件"""
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """递归合并配置字典"""
    
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    
    return {
        "model": {
            "name": "unsloth/Qwen2.5-7B-Instruct",
            "max_seq_length": 8192,
            "load_in_4bit": True,
        },
        "lora": {
            "rank": 64,
            "alpha": 64,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "dropout": 0.0,
        },
        "training": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-6,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "optim": "adamw_8bit",
            "beta": 0.04,
            "num_generations": 4,
            "epsilon": 0.2,
            "max_prompt_length": 4096,
            "max_completion_length": 1024,
            "save_steps": 100,
            "logging_steps": 10,
            "eval_steps": 50,
            "use_vllm": True,
            "vllm_gpu_memory_utilization": 0.7,
        },
        "rewards": {
            "task_weight": 0.4,
            "reproduce_weight": 0.3,
            "format_weight": 0.1,
            "compression_weight": 0.2,
            "target_compression_ratio": 0.20,
            "embedding_model": "BAAI/bge-large-en-v1.5",
        },
        "output": {
            "output_dir": "./checkpoints",
            "log_dir": "./logs",
        },
        "logging": {
            "use_wandb": False,
            "wandb_project": "context-compression",
        },
    }


def get_config_for_gpu(memory_gb: float) -> Dict[str, Any]:
    """根据GPU显存获取推荐配置"""
    
    config = get_default_config()
    
    if memory_gb < 16:
        # 8-16GB GPU
        config["model"]["load_in_4bit"] = True
        config["lora"]["rank"] = 32
        config["training"]["per_device_train_batch_size"] = 1
        config["training"]["gradient_accumulation_steps"] = 16
        config["training"]["num_generations"] = 4
        config["training"]["max_prompt_length"] = 2048
        config["training"]["max_completion_length"] = 512
    elif memory_gb < 32:
        # 16-32GB GPU
        config["model"]["load_in_4bit"] = True
        config["lora"]["rank"] = 64
        config["training"]["per_device_train_batch_size"] = 1
        config["training"]["gradient_accumulation_steps"] = 8
        config["training"]["num_generations"] = 8
        config["training"]["max_prompt_length"] = 4096
        config["training"]["max_completion_length"] = 1024
    elif memory_gb < 48:
        # 32-48GB GPU
        config["model"]["load_in_4bit"] = True
        config["lora"]["rank"] = 128
        config["training"]["per_device_train_batch_size"] = 2
        config["training"]["gradient_accumulation_steps"] = 4
        config["training"]["num_generations"] = 8
        config["training"]["max_prompt_length"] = 8192
        config["training"]["max_completion_length"] = 2048
    elif memory_gb < 90:
        # 48-90GB GPU
        config["model"]["load_in_4bit"] = True
        config["lora"]["rank"] = 128
        config["training"]["per_device_train_batch_size"] = 2
        config["training"]["gradient_accumulation_steps"] = 4
        config["training"]["num_generations"] = 8
        config["training"]["max_prompt_length"] = 16384
        config["training"]["max_completion_length"] = 4096
    else:
        # 90GB+ GPU
        config["model"]["load_in_4bit"] = True
        config["lora"]["rank"] = 128
        config["training"]["per_device_train_batch_size"] = 4
        config["training"]["gradient_accumulation_steps"] = 2
        config["training"]["num_generations"] = 8
        config["training"]["max_prompt_length"] = 24576
        config["training"]["max_completion_length"] = 4096
        config["training"]["vllm_gpu_memory_utilization"] = 0.8
    
    return config


# =============================================================================
# Text Processing Utilities
# =============================================================================

def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """计算token数量"""
    
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    
    # 简单估算：按空格分词
    return len(text.split())


def truncate_text(text: str, max_tokens: int, tokenizer: Optional[Any] = None) -> str:
    """截断文本到指定token数"""
    
    if tokenizer is not None:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return tokenizer.decode(tokens[:max_tokens])
    
    # 简单估算
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def extract_summary(text: str) -> str:
    """从模型输出中提取摘要内容"""
    
    # 尝试提取<summary>标签内容
    pattern = r"<summary>(.*?)</summary>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 尝试提取"Summary:"之后的内容
    pattern = r"Summary:\s*(.+?)(?:\n\n|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 返回整个文本
    return text.strip()


def extract_reasoning(text: str) -> str:
    """从模型输出中提取推理内容"""
    
    pattern = r"<reasoning>(.*?)</reasoning>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# =============================================================================
# GPU Utilities
# =============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """获取GPU信息"""
    
    if not torch.cuda.is_available():
        return {"available": False}
    
    gpu_info = {
        "available": True,
        "count": torch.cuda.device_count(),
        "devices": [],
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_info["devices"].append({
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": props.total_memory / 1e9,
        })
    
    return gpu_info


def print_gpu_memory():
    """打印GPU显存使用情况"""
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        print(f"GPU {i}: {allocated:.2f}GB / {total:.2f}GB (reserved: {reserved:.2f}GB)")


def clear_gpu_cache():
    """清理GPU缓存"""
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Checkpoint Utilities
# =============================================================================

def save_checkpoint(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    output_dir: str,
    step: Optional[int] = None
) -> str:
    """保存训练检查点"""
    
    # 创建输出目录
    if step is not None:
        output_dir = Path(output_dir) / f"checkpoint-{step}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存LoRA权重
    lora_path = output_dir / "lora_adapter"
    model.save_lora(str(lora_path))
    
    # 保存Tokenizer
    tokenizer_path = output_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_path))
    
    # 保存配置
    config_path = output_dir / "config.yaml"
    save_config(config, str(config_path))
    
    return str(output_dir)


def load_checkpoint(
    checkpoint_dir: str,
    model: Any,
    tokenizer: Any
) -> Dict[str, Any]:
    """加载训练检查点"""
    
    checkpoint_path = Path(checkpoint_dir)
    
    # 加载LoRA权重
    lora_path = checkpoint_path / "lora_adapter"
    if lora_path.exists():
        model.load_adapter(str(lora_path))
    
    # 加载配置
    config_path = checkpoint_path / "config.yaml"
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {}
    
    return config


# =============================================================================
# Metrics Utilities
# =============================================================================

def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """格式化指标输出"""
    
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    
    return "\n".join(lines)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """计算统计信息"""
    
    if not values:
        return {}
    
    arr = np.array(values)
    
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


# =============================================================================
# Data Utilities
# =============================================================================

def save_jsonl(data: List[Dict], output_path: str):
    """保存JSONL文件"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(input_path: str) -> List[Dict]:
    """加载JSONL文件"""
    
    data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    return data


# =============================================================================
# Wandb Utilities
# =============================================================================

def init_wandb(
    project: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Any:
    """初始化Wandb"""
    
    try:
        import wandb
        
        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=tags,
        )
        
        return run
    
    except ImportError:
        print("Warning: wandb not installed")
        return None


def log_metrics_to_wandb(metrics: Dict[str, float], step: Optional[int] = None):
    """记录指标到Wandb"""
    
    try:
        import wandb
        
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    
    except ImportError:
        pass
