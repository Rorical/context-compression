"""
Context Compression - Reward Functions
奖励计算模块

实现双目标奖励架构：
R_total = 0.4 * R_task + 0.3 * R_reproduce + 0.1 * R_format + 0.2 * R_compression
"""

import re
import numpy as np
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# =============================================================================
# Reward Configuration
# =============================================================================

@dataclass
class RewardConfig:
    """奖励配置"""
    task_weight: float = 0.4
    reproduce_weight: float = 0.3
    format_weight: float = 0.1
    compression_weight: float = 0.2
    
    target_compression_ratio: float = 0.20
    min_compression_ratio: float = 0.05
    max_compression_ratio: float = 0.50
    
    embedding_model: str = "BAAI/bge-large-en-v1.5"


# =============================================================================
# Embedding Model Manager (Singleton)
# =============================================================================

class EmbeddingManager:
    """Embedding模型管理器（单例模式）"""
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """获取embedding模型"""
        
        if model_name not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {model_name}")
                self._models[model_name] = SentenceTransformer(model_name)
            except ImportError:
                print("Warning: sentence-transformers not installed")
                self._models[model_name] = None
        
        return self._models[model_name]
    
    def encode(self, texts: Union[str, List[str]], model_name: str = "BAAI/bge-large-en-v1.5"):
        """编码文本"""
        
        model = self.get_model(model_name)
        
        if model is None:
            return None
        
        if isinstance(texts, str):
            texts = [texts]
        
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


# =============================================================================
# Reward Functions
# =============================================================================

class RewardFunctions:
    """奖励函数集合"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        初始化奖励函数
        
        Args:
            config: 奖励配置
        """
        self.config = config or RewardConfig()
        self.embedding_manager = EmbeddingManager()
    
    # -------------------------------------------------------------------------
    # Format Reward
    # -------------------------------------------------------------------------
    
    def format_reward(self, completions: List[Any], **kwargs) -> List[float]:
        """
        格式奖励：确保输出包含正确的标签
        
        期望格式：
        <reasoning>...</reasoning>
        <summary>...</summary>
        
        Args:
            completions: 模型输出列表
            
        Returns:
            奖励值列表
        """
        rewards = []
        
        for completion in completions:
            content = self._extract_content(completion)
            
            has_reasoning = "<reasoning>" in content and "</reasoning>" in content
            has_summary = "<summary>" in content and "</summary>" in content
            
            if has_reasoning and has_summary:
                # 检查内容非空
                reasoning = self._extract_tag_content(content, "reasoning")
                summary = self._extract_tag_content(content, "summary")
                
                if len(reasoning) > 10 and len(summary) > 10:
                    rewards.append(1.0)
                else:
                    rewards.append(0.5)
            elif has_summary:
                rewards.append(0.3)
            else:
                rewards.append(0.0)
        
        return rewards
    
    # -------------------------------------------------------------------------
    # Compression Ratio Reward
    # -------------------------------------------------------------------------
    
    def compression_reward(
        self,
        prompts: List[Any],
        completions: List[Any],
        original_lengths: Optional[List[int]] = None,
        **kwargs
    ) -> List[float]:
        """
        压缩率奖励
        
        鼓励将上下文压缩到目标比例附近
        
        Args:
            prompts: 提示列表
            completions: 模型输出列表
            original_lengths: 原始上下文长度列表
            
        Returns:
            奖励值列表
        """
        rewards = []
        target_ratio = self.config.target_compression_ratio
        
        for i, completion in enumerate(completions):
            content = self._extract_content(completion)
            summary = self._extract_tag_content(content, "summary")
            
            summary_len = len(summary)
            
            # 获取原始长度
            if original_lengths and i < len(original_lengths):
                orig_len = original_lengths[i]
            else:
                # 从prompt估算
                orig_len = len(self._extract_content(prompts[i])) if i < len(prompts) else 1000
            
            if orig_len > 0:
                actual_ratio = summary_len / orig_len
                
                # 高斯奖励函数：越接近目标比例奖励越高
                sigma = 0.05  # 标准差
                reward = np.exp(-0.5 * ((actual_ratio - target_ratio) / sigma) ** 2)
                
                # 惩罚过度压缩
                if actual_ratio < self.config.min_compression_ratio:
                    reward *= 0.5
                
                # 惩罚压缩不足
                if actual_ratio > self.config.max_compression_ratio:
                    reward *= 0.5
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards
    
    # -------------------------------------------------------------------------
    # Task Success Reward
    # -------------------------------------------------------------------------
    
    def task_success_reward(
        self,
        prompts: List[Any],
        completions: List[Any],
        answers: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """
        任务成功率奖励
        
        评估摘要是否包含关键信息，能够支持任务继续
        
        Args:
            prompts: 提示列表
            completions: 模型输出列表
            answers: 参考答案列表
            
        Returns:
            奖励值列表
        """
        rewards = []
        
        if answers is None:
            answers = [""] * len(completions)
        
        for prompt, completion, answer in zip(prompts, completions, answers):
            content = self._extract_content(completion)
            summary = self._extract_tag_content(content, "summary")
            
            if answer and len(answer) > 0:
                # 计算关键词重叠
                answer_keywords = set(str(answer).lower().split())
                summary_words = set(summary.lower().split())
                
                if len(answer_keywords) > 0:
                    overlap = len(answer_keywords & summary_words)
                    reward = min(1.0, overlap / len(answer_keywords))
                else:
                    reward = 0.5
            else:
                # 没有答案时，基于摘要质量给予基础奖励
                reward = 0.5 if len(summary) > 50 else 0.2
            
            rewards.append(reward)
        
        return rewards
    
    # -------------------------------------------------------------------------
    # Reproduce Reward (Embedding Similarity)
    # -------------------------------------------------------------------------
    
    def reproduce_reward(
        self,
        prompts: List[Any],
        completions: List[Any],
        original_contexts: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """
        模型复现率奖励（使用embedding相似度）
        
        计算摘要与原上下文的embedding相似度
        作为信息保留程度的代理指标
        
        Args:
            prompts: 提示列表
            completions: 模型输出列表
            original_contexts: 原始上下文列表
            
        Returns:
            奖励值列表
        """
        rewards = []
        
        if original_contexts is None:
            # 从prompts提取原始上下文
            original_contexts = [self._extract_content(p) for p in prompts]
        
        try:
            for completion, orig_ctx in zip(completions, original_contexts):
                content = self._extract_content(completion)
                summary = self._extract_tag_content(content, "summary")
                
                # 将原始上下文转换为文本
                if isinstance(orig_ctx, list):
                    orig_text = "\n".join([str(item) for item in orig_ctx])
                else:
                    orig_text = str(orig_ctx)
                
                if len(orig_text) > 0 and len(summary) > 0:
                    # 计算embedding
                    embeddings = self.embedding_manager.encode(
                        [orig_text, summary],
                        self.config.embedding_model
                    )
                    
                    if embeddings is not None:
                        orig_emb, summary_emb = embeddings[0], embeddings[1]
                        
                        # 余弦相似度
                        similarity = np.dot(orig_emb, summary_emb) / (
                            np.linalg.norm(orig_emb) * np.linalg.norm(summary_emb) + 1e-8
                        )
                        
                        # 映射到[0, 1]
                        reward = max(0.0, similarity)
                    else:
                        reward = 0.5
                else:
                    reward = 0.0
                
                rewards.append(reward)
        
        except Exception as e:
            print(f"Error in reproduce_reward: {e}")
            rewards = [0.5] * len(completions)
        
        return rewards
    
    # -------------------------------------------------------------------------
    # Combined Reward
    # -------------------------------------------------------------------------
    
    def compute_total_reward(
        self,
        prompts: List[Any],
        completions: List[Any],
        **kwargs
    ) -> List[float]:
        """
        计算总奖励
        
        R_total = w_task * R_task + w_reproduce * R_reproduce + 
                  w_format * R_format + w_compression * R_compression
        
        Args:
            prompts: 提示列表
            completions: 模型输出列表
            **kwargs: 额外参数
            
        Returns:
            总奖励值列表
        """
        # 计算各维度奖励
        r_format = np.array(self.format_reward(completions, **kwargs))
        r_compression = np.array(self.compression_reward(prompts, completions, **kwargs))
        r_task = np.array(self.task_success_reward(prompts, completions, **kwargs))
        r_reproduce = np.array(self.reproduce_reward(prompts, completions, **kwargs))
        
        # 加权求和
        total_reward = (
            self.config.format_weight * r_format +
            self.config.compression_weight * r_compression +
            self.config.task_weight * r_task +
            self.config.reproduce_weight * r_reproduce
        )
        
        return total_reward.tolist()
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _extract_content(self, completion: Any) -> str:
        """从completion中提取内容"""
        
        if isinstance(completion, list) and len(completion) > 0:
            if isinstance(completion[0], dict):
                return completion[0].get("content", "")
            return str(completion[0])
        elif isinstance(completion, dict):
            return completion.get("content", "")
        return str(completion)
    
    def _extract_tag_content(self, text: str, tag: str) -> str:
        """提取标签内容"""
        
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""


# =============================================================================
# Reward Function Factory (for GRPO)
# =============================================================================

def create_reward_functions(config: Optional[Dict[str, Any]] = None):
    """
    创建奖励函数（用于GRPO训练）
    
    Args:
        config: 配置字典
        
    Returns:
        奖励函数列表
    """
    
    # 创建奖励配置
    reward_config = RewardConfig()
    
    if config:
        reward_config.task_weight = config.get("task_weight", 0.4)
        reward_config.reproduce_weight = config.get("reproduce_weight", 0.3)
        reward_config.format_weight = config.get("format_weight", 0.1)
        reward_config.compression_weight = config.get("compression_weight", 0.2)
        reward_config.target_compression_ratio = config.get("target_compression_ratio", 0.20)
        reward_config.embedding_model = config.get("embedding_model", "BAAI/bge-large-en-v1.5")
    
    # 创建奖励函数实例
    reward_funcs = RewardFunctions(reward_config)
    
    # 返回各个奖励函数
    return {
        "format": reward_funcs.format_reward,
        "compression": reward_funcs.compression_reward,
        "task_success": reward_funcs.task_success_reward,
        "reproduce": reward_funcs.reproduce_reward,
        "total": reward_funcs.compute_total_reward,
    }


# =============================================================================
# Standalone Reward Functions (for TRL GRPOTrainer)
# =============================================================================

# 全局奖励函数实例（用于TRL）
_global_reward_config = None
_global_reward_funcs = None


def _get_global_reward_funcs():
    """获取全局奖励函数实例"""
    global _global_reward_funcs, _global_reward_config
    
    if _global_reward_funcs is None:
        if _global_reward_config is None:
            _global_reward_config = RewardConfig()
        _global_reward_funcs = RewardFunctions(_global_reward_config)
    
    return _global_reward_funcs


def format_reward_func(completions, **kwargs):
    """格式奖励函数（用于TRL）"""
    return _get_global_reward_funcs().format_reward(completions, **kwargs)


def compression_reward_func(prompts, completions, **kwargs):
    """压缩率奖励函数（用于TRL）"""
    return _get_global_reward_funcs().compression_reward(prompts, completions, **kwargs)


def task_success_reward_func(prompts, completions, **kwargs):
    """任务成功率奖励函数（用于TRL）"""
    return _get_global_reward_funcs().task_success_reward(prompts, completions, **kwargs)


def reproduce_reward_func(prompts, completions, **kwargs):
    """复现率奖励函数（用于TRL）"""
    return _get_global_reward_funcs().reproduce_reward(prompts, completions, **kwargs)


def combined_reward_func(prompts, completions, **kwargs):
    """组合奖励函数（用于TRL）"""
    return _get_global_reward_funcs().compute_total_reward(prompts, completions, **kwargs)


def set_reward_config(config: Dict[str, Any]):
    """设置全局奖励配置"""
    global _global_reward_config, _global_reward_funcs
    
    _global_reward_config = RewardConfig()
    _global_reward_config.task_weight = config.get("task_weight", 0.4)
    _global_reward_config.reproduce_weight = config.get("reproduce_weight", 0.3)
    _global_reward_config.format_weight = config.get("format_weight", 0.1)
    _global_reward_config.compression_weight = config.get("compression_weight", 0.2)
    _global_reward_config.target_compression_ratio = config.get("target_compression_ratio", 0.20)
    _global_reward_config.embedding_model = config.get("embedding_model", "BAAI/bge-large-en-v1.5")
    
    # 重置实例
    _global_reward_funcs = None
