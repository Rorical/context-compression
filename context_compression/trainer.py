"""
Context Compression - Training Module
训练模块

基于Unsloth + GRPO实现强化学习训练
"""

import torch
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from datasets import Dataset

from .utils import setup_logging, save_config, print_gpu_memory
from .models import ModelLoader
from .rewards import set_reward_config


# =============================================================================
# Context Compression Trainer
# =============================================================================

class ContextCompressionTrainer:
    """
    Context Compression训练器
    
    基于Unsloth GRPO实现强化学习训练
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.logger = None
        
        # 初始化日志
        self._setup_logger()
        
        # 设置奖励配置
        self._setup_reward_config()
    
    def _setup_logger(self):
        """设置日志"""
        
        log_dir = self.config.get("output", {}).get("log_dir", "./logs")
        run_name = self.config.get("logging", {}).get("run_name", None)
        
        self.logger = setup_logging(log_dir, run_name)
        self.logger.info("Trainer initialized")
        
        # 记录配置
        self.logger.info(f"Config: {self.config}")
    
    def _setup_reward_config(self):
        """设置奖励配置"""
        
        reward_config = self.config.get("rewards", {})
        set_reward_config(reward_config)
    
    def setup_model(self):
        """设置模型"""
        
        self.logger.info("Setting up model...")
        
        # 创建模型加载器
        model_loader = ModelLoader(self.config)
        
        # 加载模型
        self.model, self.tokenizer = model_loader.load_model()
        
        # 应用LoRA
        self.model = model_loader.apply_lora()
        
        self.logger.info("Model setup complete!")
    
    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ):
        """
        设置GRPO训练器
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集（可选）
        """
        
        self.logger.info("Setting up GRPO trainer...")

        try:
            from unsloth import is_bfloat16_supported
            from trl import GRPOTrainer, GRPOConfig
        except ImportError as exc:
            raise ImportError(
                "GRPO training dependencies are incomplete. Install the latest "
                "training stack from requirements.txt, including mergekit."
            ) from exc
        
        # 创建奖励函数
        reward_funcs = self._create_reward_functions()
        
        # GRPO配置
        training_config = self.config.get("training", {})
        
        grpo_kwargs = dict(
            # 输出目录
            output_dir=self.config.get("output", {}).get("output_dir", "./checkpoints"),
            
            # 学习率配置
            learning_rate=training_config.get("learning_rate", 1e-6),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            warmup_ratio=training_config.get("warmup_ratio", 0.1),
            
            # 批次配置
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
            num_train_epochs=training_config.get("num_train_epochs", 3),
            
            # 优化器配置
            optim=training_config.get("optim", "adamw_8bit"),
            weight_decay=training_config.get("weight_decay", 0.01),
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            
            # GRPO特定配置
            beta=training_config.get("beta", 0.04),
            num_generations=training_config.get("num_generations", 4),
            
            # 序列长度
            max_prompt_length=training_config.get("max_prompt_length", 4096),
            max_completion_length=training_config.get("max_completion_length", 1024),
            
            # vLLM配置
            use_vllm=training_config.get("use_vllm", True),
            vllm_gpu_memory_utilization=training_config.get("vllm_gpu_memory_utilization", 0.7),
            
            # Unsloth优化
            unsloth_logit_chunk_multiplier=training_config.get("unsloth_logit_chunk_multiplier", 4),
            unsloth_grpo_mini_batch=training_config.get("unsloth_grpo_mini_batch", 1),
            
            # 日志和保存
            logging_steps=training_config.get("logging_steps", 10),
            save_steps=training_config.get("save_steps", 100),
            save_total_limit=training_config.get("save_total_limit", 3),

            # 报告
            report_to="wandb" if self.config.get("logging", {}).get("use_wandb", False) else None,
        )

        if eval_dataset is not None:
            train_batch_size = training_config.get("per_device_train_batch_size", 1)
            num_generations = training_config.get("num_generations", 4)
            global_eval_batch_size = train_batch_size

            if global_eval_batch_size % num_generations != 0:
                self.logger.warning(
                    "Disabling eval during GRPO setup because global eval batch size "
                    "(%s) is not divisible by num_generations (%s).",
                    global_eval_batch_size,
                    num_generations,
                )
            else:
                grpo_kwargs["eval_steps"] = training_config.get("eval_steps", 50)
                grpo_kwargs["eval_strategy"] = "steps"
                grpo_kwargs["do_eval"] = True

        grpo_config = GRPOConfig(**grpo_kwargs)
        
        # 创建GRPOTrainer
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        self.logger.info("GRPO trainer setup complete!")
    
    def _create_reward_functions(self) -> List:
        """创建奖励函数列表"""
        
        from .rewards import (
            format_reward_func,
            compression_reward_func,
            task_success_reward_func,
            reproduce_reward_func,
        )
        
        # 返回所有奖励函数
        return [
            format_reward_func,
            compression_reward_func,
            task_success_reward_func,
            reproduce_reward_func,
        ]
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        执行训练
        
        Args:
            resume_from_checkpoint: 从检查点恢复训练
        """
        
        self.logger.info("Starting training...")
        
        # 打印显存信息
        if torch.cuda.is_available():
            self.logger.info(f"GPU memory before training:")
            print_gpu_memory()
        
        # 执行训练
        if resume_from_checkpoint:
            self.logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            self.trainer.train()
        
        # 打印显存信息
        if torch.cuda.is_available():
            self.logger.info(f"GPU memory after training:")
            print_gpu_memory()
        
        self.logger.info("Training complete!")
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        保存模型
        
        Args:
            output_dir: 输出目录
        """
        
        if output_dir is None:
            output_dir = self.config.get("output", {}).get("output_dir", "./checkpoints")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving model to {output_dir}...")
        
        # 保存LoRA权重
        lora_path = output_path / "lora_adapter"
        self.model.save_lora(str(lora_path))
        self.logger.info(f"LoRA adapter saved to {lora_path}")
        
        # 保存Tokenizer
        tokenizer_path = output_path / "tokenizer"
        self.tokenizer.save_pretrained(str(tokenizer_path))
        self.logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        # 保存配置
        config_path = output_path / "config.yaml"
        save_config(self.config, str(config_path))
        self.logger.info(f"Config saved to {config_path}")
    
    def compress_context(
        self,
        context: str,
        max_summary_length: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        压缩上下文
        
        Args:
            context: 原始上下文
            max_summary_length: 最大摘要长度
            temperature: 采样温度
            top_p: Top-p采样参数
            
        Returns:
            压缩后的摘要
        """
        
        # 构建提示
        prompt = self._build_compression_prompt(context, max_summary_length)
        
        # Tokenize输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 设置最大长度
        max_length = max_summary_length or self.config.get("training", {}).get("max_completion_length", 1024)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        
        # 解码摘要
        summary = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return summary.strip()
    
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
    
    def evaluate(
        self,
        eval_dataset: Dataset,
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            eval_dataset: 评估数据集
            batch_size: 批次大小
            
        Returns:
            评估指标字典
        """
        
        self.logger.info("Starting evaluation...")
        
        from .evaluator import ContextCompressionEvaluator
        
        # 创建评估器
        evaluator = ContextCompressionEvaluator(
            tokenizer=self.tokenizer,
            embedding_model=self.config.get("rewards", {}).get("embedding_model", "BAAI/bge-large-en-v1.5")
        )
        
        # 收集数据
        contexts = []
        predictions = []
        references = []
        
        for i in tqdm(range(0, len(eval_dataset), batch_size), desc="Evaluating"):
            batch = eval_dataset[i:i+batch_size]
            
            for context in batch.get("context", []):
                # 生成摘要
                summary = self.compress_context(context)
                
                contexts.append(context)
                predictions.append(summary)
            
            # 收集参考摘要
            if "gold_summary" in batch:
                references.extend(batch["gold_summary"])
        
        # 执行评估
        metrics = evaluator.evaluate(contexts, predictions, references if references else None)
        
        self.logger.info(f"Evaluation results:\n{metrics}")
        
        return metrics.to_dict()


# =============================================================================
# Helper Functions
# =============================================================================

def create_trainer(config: Dict[str, Any]) -> ContextCompressionTrainer:
    """
    创建训练器
    
    Args:
        config: 配置字典
        
    Returns:
        训练器实例
    """
    return ContextCompressionTrainer(config)


def train_model(
    config: Dict[str, Any],
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None
) -> ContextCompressionTrainer:
    """
    训练模型（便捷函数）
    
    Args:
        config: 配置字典
        train_dataset: 训练数据集
        eval_dataset: 评估数据集（可选）
        
    Returns:
        训练后的训练器
    """
    # 创建训练器
    trainer = ContextCompressionTrainer(config)
    
    # 设置模型
    trainer.setup_model()
    
    # 设置训练器
    trainer.setup_trainer(train_dataset, eval_dataset)
    
    # 训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    
    return trainer
