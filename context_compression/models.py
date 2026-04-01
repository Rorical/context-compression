"""
Context Compression - Model Definitions
模型定义模块

包含：
1. 模型加载和管理
2. Embedding模型管理
3. 模型推理封装
"""

# Import Unsloth before any transformers usage elsewhere in the process.
import unsloth  # noqa: F401

import torch
from typing import Optional, Dict, Any, List
from pathlib import Path

from unsloth import FastLanguageModel


# =============================================================================
# Model Loader
# =============================================================================

def load_tokenizer_for_model(model_name: str) -> Any:
    """
    Load a text tokenizer for the configured model.

    For newer remote-code or processor-backed models such as Qwen3.5,
    AutoTokenizer may fail unless a newer transformers build is installed.
    This helper falls back to AutoProcessor.tokenizer when available and raises
    a clearer error for unsupported local transformers versions.
    """

    from transformers import AutoProcessor, AutoTokenizer

    load_kwargs = {
        "trust_remote_code": True,
    }

    tokenizer_error = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as exc:
        tokenizer_error = exc

    try:
        processor = AutoProcessor.from_pretrained(model_name, **load_kwargs)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
    except Exception:
        pass

    model_name_lower = model_name.lower()
    if "qwen3.5" in model_name_lower:
        raise RuntimeError(
            "Failed to load the tokenizer for Qwen3.5. The Qwen3.5 model card "
            "currently requires the latest transformers from main. In Colab, run:\n"
            '  pip uninstall -y transformers tokenizers\n'
            '  pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"\n'
            "then restart the runtime and try again."
        ) from tokenizer_error

    raise RuntimeError(
        f"Failed to load tokenizer for model '{model_name}'."
    ) from tokenizer_error

class ModelLoader:
    """模型加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(
        self,
        model_name: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        load_in_4bit: Optional[bool] = None,
    ) -> tuple:
        """
        加载Unsloth模型
        
        Args:
            model_name: 模型名称
            max_seq_length: 最大序列长度
            load_in_4bit: 是否使用4-bit量化
            
        Returns:
            (model, tokenizer) 元组
        """
        model_name = model_name or self.config["model"]["name"]
        max_seq_length = max_seq_length or self.config["model"]["max_seq_length"]
        load_in_4bit = load_in_4bit if load_in_4bit is not None else self.config["model"]["load_in_4bit"]
        max_lora_rank = self.config.get("training", {}).get(
            "max_lora_rank",
            self.config.get("lora", {}).get("rank", 64),
        )
        
        print(f"Loading model: {model_name}")
        print(f"Max sequence length: {max_seq_length}")
        print(f"4-bit quantization: {load_in_4bit}")
        print(f"Max LoRA rank: {max_lora_rank}")
        
        # 加载模型
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=self.config["training"].get("use_vllm", True),
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=self.config["training"].get("vllm_gpu_memory_utilization", 0.7),
        )
        
        print("Model loaded successfully!")
        
        return self.model, self.tokenizer
    
    def apply_lora(
        self,
        rank: Optional[int] = None,
        alpha: Optional[int] = None,
        target_modules: Optional[List[str]] = None,
        dropout: float = 0.0,
    ) -> Any:
        """
        应用LoRA
        
        Args:
            rank: LoRA秩
            alpha: LoRA alpha
            target_modules: 目标模块列表
            dropout: Dropout率
            
        Returns:
            应用LoRA后的模型
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        rank = rank or self.config["lora"]["rank"]
        alpha = alpha or self.config["lora"]["alpha"]
        target_modules = target_modules or self.config["lora"]["target_modules"]
        
        print(f"Applying LoRA (rank={rank}, alpha={alpha})...")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=rank,
            target_modules=target_modules,
            lora_alpha=alpha,
            lora_dropout=dropout,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        # 打印可训练参数
        self._print_trainable_parameters()
        
        return self.model
    
    def _print_trainable_parameters(self):
        """打印可训练参数信息"""
        
        if self.model is None:
            return
        
        trainable_params = 0
        all_params = 0
        
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Trainable parameters: {trainable_params:,} / {all_params:,} "
              f"({100 * trainable_params / all_params:.2f}%)")
    
    def save_model(self, output_dir: str):
        """保存模型"""
        
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存LoRA权重
        lora_path = output_path / "lora_adapter"
        self.model.save_lora(str(lora_path))
        print(f"LoRA adapter saved to {lora_path}")
        
        # 保存Tokenizer
        if self.tokenizer is not None:
            tokenizer_path = output_path / "tokenizer"
            self.tokenizer.save_pretrained(str(tokenizer_path))
            print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_adapter(self, adapter_path: str):
        """加载LoRA适配器"""
        
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        print(f"Loading LoRA adapter from {adapter_path}...")
        self.model.load_adapter(adapter_path)
        print("Adapter loaded!")


# =============================================================================
# Embedding Model Manager
# =============================================================================

class EmbeddingModelManager:
    """
    Embedding模型管理器（单例模式）
    
    避免重复加载embedding模型
    """
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Any:
        """
        获取embedding模型
        
        Args:
            model_name: 模型名称
            device: 运行设备
            
        Returns:
            embedding模型
        """
        cache_key = f"{model_name}_{device}"
        
        if cache_key not in self._models:
            print(f"Loading embedding model: {model_name}")
            
            try:
                from sentence_transformers import SentenceTransformer
                
                self._models[cache_key] = SentenceTransformer(model_name, device=device)
                print(f"Embedding model loaded on {device}")
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        
        return self._models[cache_key]
    
    def clear_cache(self):
        """清理模型缓存"""
        self._models.clear()
        print("Embedding model cache cleared")


# =============================================================================
# Model Inference Wrapper
# =============================================================================

class ModelInference:
    """模型推理封装"""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化推理封装
        
        Args:
            model: 模型
            tokenizer: 分词器
            config: 配置字典
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        
        # 设置设备
        self.device = next(model.parameters()).device
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: Top-p采样参数
            top_k: Top-k采样参数
            repetition_penalty: 重复惩罚
            do_sample: 是否采样
            
        Returns:
            生成的文本
        """
        # Tokenize输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("max_prompt_length", 4096)
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_summary(
        self,
        context: str,
        max_summary_length: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        生成摘要
        
        Args:
            context: 上下文文本
            max_summary_length: 最大摘要长度
            temperature: 采样温度
            top_p: Top-p采样参数
            
        Returns:
            生成的摘要
        """
        # 构建提示
        prompt = self._build_compression_prompt(context, max_summary_length)
        
        # 生成
        max_length = max_summary_length or self.config.get("max_completion_length", 1024)
        
        summary = self.generate(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
        )
        
        return summary
    
    def _build_compression_prompt(
        self,
        context: str,
        target_length: Optional[int] = None
    ) -> str:
        """
        构建压缩提示
        
        Args:
            context: 上下文文本
            target_length: 目标摘要长度
            
        Returns:
            提示文本
        """
        length_hint = ""
        if target_length is not None:
            length_hint = f"The summary should be approximately {target_length} tokens long.\n\n"
        
        prompt = f"""You are an expert at summarizing conversations. Your task is to create a concise summary of the following conversation context while preserving all important information.

{length_hint}Context:
{context}

Please provide a concise summary that captures the key points, decisions, and important information from the conversation. Format your response as:

<reasoning>
Briefly explain what key information needs to be preserved.
</reasoning>

<summary>
Your concise summary here.
</summary>

Summary:"""
        
        return prompt
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 4,
    ) -> List[str]:
        """
        批量生成
        
        Args:
            prompts: 提示列表
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: Top-p采样参数
            batch_size: 批次大小
            
        Returns:
            生成文本列表
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("max_prompt_length", 4096)
            ).to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # 解码
            for j, output in enumerate(outputs):
                generated = self.tokenizer.decode(
                    output[inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                results.append(generated.strip())
        
        return results


# =============================================================================
# Helper Functions
# =============================================================================

def load_model_for_training(config: Dict[str, Any]) -> tuple:
    """
    加载训练用模型
    
    Args:
        config: 配置字典
        
    Returns:
        (model, tokenizer) 元组
    """
    loader = ModelLoader(config)
    model, tokenizer = loader.load_model()
    model = loader.apply_lora()
    
    return model, tokenizer


def load_model_for_inference(
    checkpoint_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    加载推理用模型
    
    Args:
        checkpoint_dir: 检查点目录
        config: 配置字典（可选）
        
    Returns:
        (model, tokenizer) 元组
    """
    from .utils import load_config
    
    # 加载配置
    if config is None:
        config_path = Path(checkpoint_dir) / "config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            config = get_default_config()
    
    # 加载模型
    loader = ModelLoader(config)
    model, tokenizer = loader.load_model()
    
    # 加载LoRA适配器
    lora_path = Path(checkpoint_dir) / "lora_adapter"
    if lora_path.exists():
        loader.load_adapter(str(lora_path))
    
    return model, tokenizer


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    from .utils import get_default_config as utils_get_default_config

    return utils_get_default_config()
