"""
Context Compression - Data Pipeline
数据管线模块

功能：
1. 多数据源加载（DialogSum, SAMSum, UltraChat, 合成数据）
2. 数据预处理和清洗
3. 压缩点检测和样本构造
4. PyTorch Dataset实现
5. Online RL数据迭代器
"""

import json
import random
import re
from typing import List, Dict, Optional, Iterator, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer
import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DialogueTurn:
    """对话轮次数据结构"""
    speaker: str  # 说话者角色: "user" 或 "assistant"
    content: str  # 对话内容
    timestamp: Optional[str] = None  # 可选时间戳
    metadata: Dict = field(default_factory=dict)  # 额外元数据


@dataclass
class CompressionSample:
    """Context Compression训练样本格式"""
    
    sample_id: str
    dialogue_history: List[DialogueTurn]
    compression_point: int
    context_before: List[DialogueTurn]
    gold_summary: Optional[str] = None
    instruction: str = ""
    domain: str = "general"
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "sample_id": self.sample_id,
            "dialogue_history": [
                {"speaker": t.speaker, "content": t.content}
                for t in self.dialogue_history
            ],
            "compression_point": self.compression_point,
            "context_before": [
                {"speaker": t.speaker, "content": t.content}
                for t in self.context_before
            ],
            "gold_summary": self.gold_summary,
            "instruction": self.instruction,
            "domain": self.domain,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CompressionSample":
        """从字典创建实例"""
        dialogue_history = [
            DialogueTurn(**turn) for turn in data.get("dialogue_history", [])
        ]
        context_before = [
            DialogueTurn(**turn) for turn in data.get("context_before", [])
        ]
        return cls(
            sample_id=data["sample_id"],
            dialogue_history=dialogue_history,
            compression_point=data["compression_point"],
            context_before=context_before,
            gold_summary=data.get("gold_summary"),
            instruction=data.get("instruction", ""),
            domain=data.get("domain", "general"),
            metadata=data.get("metadata", {})
        )


# =============================================================================
# Shared Helpers
# =============================================================================

def _resolve_hf_split(split: str, sample_limit: Optional[int] = None) -> str:
    """Build a Hugging Face split expression with an optional row cap."""
    if sample_limit is None or sample_limit <= 0:
        return split
    return f"{split}[:{sample_limit}]"


def _stringify_message_content(content: Any) -> str:
    """Normalize HF chat message content into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def _split_reasoning_trace(text: str) -> Tuple[str, str]:
    """Split a reasoning trace into reasoning text and final answer text."""
    text = (text or "").strip()
    if not text:
        return "", ""

    patterns = [
        r"<think>(.*?)</think>\s*(.*)",
        r"<Thinking>(.*?)</Thinking>\s*(.*)",
        r"<reasoning>(.*?)</reasoning>\s*(.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return reasoning, answer

    marker_patterns = [
        r"^(.*?)(?:\n+##\s*Final Answer[:\s]*)(.*)$",
        r"^(.*?)(?:\n+Final Answer[:\s]*)(.*)$",
        r"^(.*?)(?:\n+\*\*Answer:\*\*\s*)(.*)$",
        r"^(.*?)(?:\n+Answer:\s*)(.*)$",
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return reasoning, answer

    return text, ""


# =============================================================================
# Dataset Loaders
# =============================================================================

class DialogSumLoader:
    """DialogSum数据集加载器"""
    
    DATASET_NAME = "knkarthick/dialogsum"
    
    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        sample_limit: Optional[int] = None
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.sample_limit = sample_limit
        self.data = []
    
    def load(self) -> List[Dict]:
        """从HuggingFace加载DialogSum数据"""
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                self.DATASET_NAME,
                split=_resolve_hf_split(self.split, self.sample_limit),
                cache_dir=self.cache_dir
            )
            self.data = list(dataset)
            print(f"Loaded {len(self.data)} samples from DialogSum ({self.split})")
            return self.data
        except Exception as e:
            print(f"Failed to load DialogSum: {e}")
            return []
    
    def parse_dialogue(self, dialogue_text: str) -> List[DialogueTurn]:
        """解析DialogSum格式的对话文本"""
        turns = []
        pattern = r'#([^#]+)#:\s*(.*?)(?=\n#|$)'
        matches = re.findall(pattern, dialogue_text, re.DOTALL)
        
        for speaker, content in matches:
            turns.append(DialogueTurn(
                speaker=speaker.strip(),
                content=content.strip()
            ))
        return turns
    
    def to_compression_samples(
        self,
        min_turns: int = 6,
        max_turns: int = 50
    ) -> List[CompressionSample]:
        """转换为CompressionSample格式"""
        
        if not self.data:
            self.load()
        
        samples = []
        for idx, item in enumerate(self.data):
            dialogue = self.parse_dialogue(item.get("dialogue", ""))
            
            if len(dialogue) < min_turns or len(dialogue) > max_turns:
                continue
            
            compression_point = len(dialogue) // 2
            context_before = dialogue[:compression_point]
            gold_summary = item.get("summary", "")
            
            sample = CompressionSample(
                sample_id=f"dialogsum_{self.split}_{idx}",
                dialogue_history=dialogue,
                compression_point=compression_point,
                context_before=context_before,
                gold_summary=gold_summary,
                instruction="Please summarize the following conversation.",
                domain=item.get("topic", "general"),
                metadata={"source": "dialogsum", "topic": item.get("topic", "")}
            )
            samples.append(sample)
        
        print(f"Created {len(samples)} compression samples from DialogSum")
        return samples


class SAMSumLoader:
    """SAMSum数据集加载器"""
    
    DATASET_NAME = "Samsung/samsum"
    
    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        sample_limit: Optional[int] = None
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.sample_limit = sample_limit
        self.data = []
    
    def load(self) -> List[Dict]:
        """从HuggingFace加载SAMSum数据"""
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                self.DATASET_NAME,
                split=_resolve_hf_split(self.split, self.sample_limit),
                cache_dir=self.cache_dir
            )
            self.data = list(dataset)
            print(f"Loaded {len(self.data)} samples from SAMSum ({self.split})")
            return self.data
        except Exception as e:
            print(f"Failed to load SAMSum: {e}")
            return []
    
    def parse_dialogue(self, dialogue_text: str) -> List[DialogueTurn]:
        """解析SAMSum格式的对话文本"""
        turns = []
        for line in dialogue_text.strip().split('\n'):
            if ':' in line:
                speaker, content = line.split(':', 1)
                turns.append(DialogueTurn(
                    speaker=speaker.strip(),
                    content=content.strip()
                ))
        return turns
    
    def to_compression_samples(
        self,
        min_turns: int = 5,
        max_turns: int = 30
    ) -> List[CompressionSample]:
        """转换为CompressionSample格式"""
        
        if not self.data:
            self.load()
        
        samples = []
        for idx, item in enumerate(self.data):
            dialogue = self.parse_dialogue(item.get("dialogue", ""))
            
            if len(dialogue) < min_turns or len(dialogue) > max_turns:
                continue
            
            compression_point = len(dialogue) // 2
            context_before = dialogue[:compression_point]
            gold_summary = item.get("summary", "")
            
            sample = CompressionSample(
                sample_id=f"samsum_{self.split}_{idx}",
                dialogue_history=dialogue,
                compression_point=compression_point,
                context_before=context_before,
                gold_summary=gold_summary,
                instruction="Summarize this chat conversation briefly.",
                domain="chat",
                metadata={"source": "samsum"}
            )
            samples.append(sample)
        
        print(f"Created {len(samples)} compression samples from SAMSum")
        return samples


class UltraChatLoader:
    """UltraChat数据集加载器"""
    
    DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
    
    def __init__(
        self,
        split: str = "train_sft",
        cache_dir: Optional[str] = None,
        sample_limit: Optional[int] = None
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.sample_limit = sample_limit
        self.data = []
    
    def load(self) -> List[Dict]:
        """从HuggingFace加载UltraChat数据"""
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                self.DATASET_NAME,
                split=_resolve_hf_split(self.split, self.sample_limit),
                cache_dir=self.cache_dir
            )
            self.data = list(dataset)
            print(f"Loaded {len(self.data)} samples from UltraChat ({self.split})")
            return self.data
        except Exception as e:
            print(f"Failed to load UltraChat: {e}")
            return []
    
    def parse_dialogue(self, messages: List[Dict]) -> List[DialogueTurn]:
        """解析UltraChat格式的对话"""
        turns = []
        for msg in messages:
            speaker = "user" if msg.get("role") == "user" else "assistant"
            turns.append(DialogueTurn(
                speaker=speaker,
                content=msg.get("content", "")
            ))
        return turns
    
    def to_compression_samples(
        self,
        min_turns: int = 10,
        max_turns: int = 50
    ) -> List[CompressionSample]:
        """转换为CompressionSample格式"""
        
        if not self.data:
            self.load()
        
        samples = []
        for idx, item in enumerate(self.data):
            messages = item.get("messages", [])
            dialogue = self.parse_dialogue(messages)
            
            if len(dialogue) < min_turns or len(dialogue) > max_turns:
                continue
            
            compression_point = len(dialogue) // 2
            context_before = dialogue[:compression_point]
            
            sample = CompressionSample(
                sample_id=f"ultrachat_{self.split}_{idx}",
                dialogue_history=dialogue,
                compression_point=compression_point,
                context_before=context_before,
                gold_summary=None,  # UltraChat没有预标注摘要
                instruction="Summarize our conversation so far.",
                domain="general",
                metadata={"source": "ultrachat", "prompt": item.get("prompt", "")}
            )
            samples.append(sample)
        
        print(f"Created {len(samples)} compression samples from UltraChat")
        return samples


class OpusReasoningLoader:
    """Opus reasoning trace dataset loader."""

    DATASET_NAME = "nohurry/Opus-4.6-Reasoning-3000x-filtered"

    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        sample_limit: Optional[int] = None
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.sample_limit = sample_limit
        self.data = []

    def load(self) -> List[Dict]:
        """Load the dataset from Hugging Face."""
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                self.DATASET_NAME,
                split=_resolve_hf_split(self.split, self.sample_limit),
                cache_dir=self.cache_dir
            )
            self.data = list(dataset)
            print(f"Loaded {len(self.data)} samples from Opus reasoning ({self.split})")
            return self.data
        except Exception as e:
            print(f"Failed to load Opus reasoning dataset: {e}")
            return []

    def to_compression_samples(self) -> List[CompressionSample]:
        """Convert rows into compression samples."""
        if not self.data:
            self.load()

        samples = []
        for idx, item in enumerate(self.data):
            problem = (item.get("problem") or "").strip()
            thinking = (item.get("thinking") or "").strip()
            solution = (item.get("solution") or "").strip()

            if not problem:
                continue

            dialogue = [DialogueTurn(speaker="user", content=problem)]
            context_before = [DialogueTurn(speaker="user", content=problem)]

            if thinking:
                thinking_turn = DialogueTurn(speaker="assistant", content=thinking)
                dialogue.append(thinking_turn)
                context_before.append(thinking_turn)

            if solution:
                dialogue.append(DialogueTurn(speaker="assistant", content=solution))

            sample = CompressionSample(
                sample_id=f"opus_reasoning_{self.split}_{idx}",
                dialogue_history=dialogue,
                compression_point=len(context_before),
                context_before=context_before,
                gold_summary=solution or None,
                instruction="Summarize the problem and the reasoning trace so the task can be resumed later.",
                domain=item.get("category", "reasoning"),
                metadata={
                    "source": "opus_reasoning",
                    "difficulty": item.get("difficulty"),
                    "timestamp": item.get("timestamp"),
                    "row_id": item.get("id"),
                    "source_problem_id": item.get("source_id") or item.get("dataset_id"),
                },
            )
            samples.append(sample)

        print(f"Created {len(samples)} compression samples from Opus reasoning")
        return samples


class KimiK25Loader:
    """KIMI reasoning trace dataset loader."""

    DATASET_NAME = "ianncity/KIMI-K2.5-450000x"

    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        sample_limit: Optional[int] = None
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.sample_limit = sample_limit
        self.data = []

    def load(self) -> List[Dict]:
        """Load the dataset from Hugging Face."""
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                self.DATASET_NAME,
                split=_resolve_hf_split(self.split, self.sample_limit),
                cache_dir=self.cache_dir
            )
            self.data = list(dataset)
            print(f"Loaded {len(self.data)} samples from KIMI K2.5 ({self.split})")
            return self.data
        except Exception as e:
            print(f"Failed to load KIMI K2.5 dataset: {e}")
            return []

    def to_compression_samples(self) -> List[CompressionSample]:
        """Convert chat rows into compression samples."""
        if not self.data:
            self.load()

        samples = []
        for idx, item in enumerate(self.data):
            messages = item.get("messages", [])
            if not messages:
                continue

            dialogue = []
            context_before = []
            answer_segments = []

            for message in messages:
                role = message.get("role", "user")
                content = _stringify_message_content(message.get("content", ""))
                if not content:
                    continue

                if role == "assistant":
                    reasoning, answer = _split_reasoning_trace(content)
                    if reasoning:
                        reasoning_turn = DialogueTurn(speaker="assistant", content=reasoning)
                        dialogue.append(reasoning_turn)
                        context_before.append(reasoning_turn)
                    else:
                        assistant_turn = DialogueTurn(speaker="assistant", content=content)
                        dialogue.append(assistant_turn)
                        context_before.append(assistant_turn)

                    if answer:
                        answer_turn = DialogueTurn(speaker="assistant", content=answer)
                        dialogue.append(answer_turn)
                        answer_segments.append(answer)
                else:
                    speaker = "user" if role == "user" else role
                    turn = DialogueTurn(speaker=speaker, content=content)
                    dialogue.append(turn)
                    context_before.append(turn)

            if not context_before:
                continue

            # If no explicit answer segment was found, keep the last turn in the compressible context.
            if len(dialogue) == len(context_before) and dialogue:
                compression_point = len(dialogue)
            else:
                compression_point = len(context_before)

            sample = CompressionSample(
                sample_id=f"kimi_k25_{self.split}_{idx}",
                dialogue_history=dialogue,
                compression_point=compression_point,
                context_before=context_before,
                gold_summary="\n\n".join(answer_segments) or None,
                instruction="Summarize the coding or reasoning trace so the task can be resumed later.",
                domain="coding_reasoning",
                metadata={
                    "source": "kimi_k25",
                    "num_messages": len(messages),
                    "has_explicit_answer_segment": bool(answer_segments),
                },
            )
            samples.append(sample)

        print(f"Created {len(samples)} compression samples from KIMI K2.5")
        return samples


# =============================================================================
# Synthetic Data Generator
# =============================================================================

class SyntheticDialogueGenerator:
    """合成对话数据生成器"""
    
    SCENARIOS = {
        "customer_service": {
            "topics": ["product inquiry", "refund request", "technical support", 
                      "account issue", "shipping question"],
        },
        "technical_discussion": {
            "topics": ["code review", "architecture design", "bug investigation",
                      "performance optimization", "feature planning"],
        },
        "educational_tutoring": {
            "topics": ["concept explanation", "homework help", "exam preparation",
                      "study plan", "resource recommendation"],
        },
    }
    
    def __init__(self):
        self.generated_count = 0
    
    def generate_dialogue(
        self,
        scenario: str,
        num_turns: int = 20
    ) -> List[DialogueTurn]:
        """生成合成对话"""
        
        scenario_info = self.SCENARIOS.get(scenario, self.SCENARIOS["customer_service"])
        topic = random.choice(scenario_info["topics"])
        
        turns = []
        
        # 开场
        turns.append(DialogueTurn(
            speaker="user",
            content=f"Hi, I need help with {topic}."
        ))
        
        # 中间轮次
        for i in range(1, num_turns - 1):
            speaker = "user" if i % 2 == 0 else "assistant"
            if speaker == "user":
                content = f"Can you tell me more about {topic}? Question {i}."
            else:
                content = f"Certainly! Regarding {topic}, here's what you need to know for turn {i}. "
                content += "This is additional information to make the response longer. " * 3
            turns.append(DialogueTurn(speaker=speaker, content=content))
        
        # 结束
        turns.append(DialogueTurn(
            speaker="assistant",
            content=f"Is there anything else I can help you with regarding {topic}?"
        ))
        
        return turns
    
    def generate_summary(self, dialogue: List[DialogueTurn]) -> str:
        """生成简单摘要"""
        
        user_turns = [t for t in dialogue if t.speaker == "user"]
        assistant_turns = [t for t in dialogue if t.speaker == "assistant"]
        
        summary_parts = []
        
        if user_turns:
            first_user = user_turns[0].content[:100]
            summary_parts.append(f"The user asked about: {first_user}...")
        
        if assistant_turns:
            summary_parts.append(f"The assistant provided {len(assistant_turns)} responses.")
        
        return " ".join(summary_parts)
    
    def generate_batch(
        self,
        num_samples: int,
        scenarios: Optional[List[str]] = None,
        min_turns: int = 10,
        max_turns: int = 50
    ) -> List[CompressionSample]:
        """批量生成合成数据"""
        
        if scenarios is None:
            scenarios = list(self.SCENARIOS.keys())
        
        samples = []
        
        for i in range(num_samples):
            scenario = random.choice(scenarios)
            num_turns = random.randint(min_turns, max_turns)
            
            dialogue = self.generate_dialogue(scenario, num_turns)
            compression_point = len(dialogue) // 2
            context_before = dialogue[:compression_point]
            gold_summary = self.generate_summary(dialogue)
            
            sample = CompressionSample(
                sample_id=f"synthetic_{self.generated_count}",
                dialogue_history=dialogue,
                compression_point=compression_point,
                context_before=context_before,
                gold_summary=gold_summary,
                instruction="Please summarize the conversation so far.",
                domain=scenario,
                metadata={"source": "synthetic", "scenario": scenario}
            )
            samples.append(sample)
            self.generated_count += 1
        
        print(f"Generated {len(samples)} synthetic samples")
        return samples


# =============================================================================
# PyTorch Dataset
# =============================================================================

class ContextCompressionDataset(Dataset):
    """Context Compression PyTorch Dataset"""
    
    def __init__(
        self,
        samples: List[CompressionSample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        summary_max_length: int = 512,
        instruction_template: Optional[str] = None,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.summary_max_length = summary_max_length
        
        # 默认指令模板
        self.instruction_template = instruction_template or (
            "Below is a conversation history. Please provide a concise summary "
            "that captures the key information and main points.\n\n"
            "Conversation:\n{context}\n\nSummary:"
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        
        sample = self.samples[idx]
        
        # 构建输入文本
        context_text = self._format_context(sample.context_before)
        input_text = self.instruction_template.format(context=context_text)
        
        # Tokenize输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "sample_id": sample.sample_id,
            "domain": sample.domain,
            "context": context_text,
        }
        
        # 如果有黄金摘要，添加标签
        if sample.gold_summary:
            summary_encoding = self.tokenizer(
                sample.gold_summary,
                max_length=self.summary_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            result["labels"] = summary_encoding["input_ids"].squeeze(0)
            result["gold_summary"] = sample.gold_summary
        
        return result
    
    def _format_context(self, turns: List[DialogueTurn]) -> str:
        """格式化对话上下文"""
        
        lines = []
        for turn in turns:
            prefix = "User" if turn.speaker == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)


# =============================================================================
# Data Pipeline Main Class
# =============================================================================

class DataPipeline:
    """Context Compression数据管线主类"""
    
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        cache_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        
        # 组件
        self.synthetic_generator = SyntheticDialogueGenerator()
        
        # 数据存储
        self.raw_samples: List[CompressionSample] = []
        self.processed_samples: List[CompressionSample] = []
    
    def load_real_datasets(
        self,
        datasets_config: List[Dict]
    ) -> List[CompressionSample]:
        """
        加载真实数据集
        
        Args:
            datasets_config: 数据集配置列表
                [
                    {"name": "dialogsum", "split": "train", "weight": 0.3},
                    {"name": "samsum", "split": "train", "weight": 0.3},
                    ...
                ]
        """
        all_samples = []
        
        for config in datasets_config:
            name = config["name"]
            split = config.get("split", "train")
            weight = config.get("weight", 1.0)
            sample_limit = config.get("max_samples")
            
            loader = None
            if name == "dialogsum":
                loader = DialogSumLoader(split, self.cache_dir, sample_limit)
            elif name == "samsum":
                loader = SAMSumLoader(split, self.cache_dir, sample_limit)
            elif name == "ultrachat":
                loader = UltraChatLoader(split, self.cache_dir, sample_limit)
            elif name == "opus_reasoning":
                loader = OpusReasoningLoader(split, self.cache_dir, sample_limit)
            elif name == "kimi_k25":
                loader = KimiK25Loader(split, self.cache_dir, sample_limit)
            
            if loader:
                samples = loader.to_compression_samples()

                if weight < 1.0:
                    num_samples = int(len(samples) * weight)
                    if num_samples > 0:
                        samples = random.sample(samples, num_samples)
                    else:
                        samples = []

                final_limit = config.get("final_max_samples")
                if final_limit and len(samples) > final_limit:
                    samples = random.sample(samples, final_limit)

                all_samples.extend(samples)
        
        self.raw_samples.extend(all_samples)
        print(f"Loaded {len(all_samples)} samples from real datasets")
        return all_samples
    
    def generate_synthetic_data(
        self,
        num_samples: int,
        scenarios: Optional[List[str]] = None
    ) -> List[CompressionSample]:
        """生成合成数据"""
        
        samples = self.synthetic_generator.generate_batch(
            num_samples=num_samples,
            scenarios=scenarios
        )
        self.raw_samples.extend(samples)
        return samples
    
    def preprocess(
        self,
        filter_by_length: bool = True,
        min_tokens: int = 100,
        max_tokens: int = 8000,
    ) -> List[CompressionSample]:
        """
        预处理数据
        
        Args:
            filter_by_length: 是否按长度过滤
            min_tokens: 最小token数
            max_tokens: 最大token数
        """
        samples = self.raw_samples.copy()
        
        # 长度过滤
        if filter_by_length and self.tokenizer:
            filtered = []
            for sample in samples:
                context_text = "\n".join([t.content for t in sample.context_before])
                token_count = len(self.tokenizer.encode(context_text))
                
                if min_tokens <= token_count <= max_tokens:
                    filtered.append(sample)
            
            samples = filtered
            print(f"Filtered by length: {len(self.raw_samples)} -> {len(samples)}")
        
        self.processed_samples = samples
        return samples
    
    def create_dataset(
        self,
        split: str = "train",
        max_length: int = 4096,
        summary_max_length: int = 512
    ) -> ContextCompressionDataset:
        """
        创建PyTorch Dataset
        
        Args:
            split: 数据集划分 ("train" 或 "eval")
            max_length: 最大输入长度
            summary_max_length: 摘要最大长度
        """
        if not self.processed_samples:
            self.preprocess()
        
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")
        
        # 划分训练/验证集
        if split == "train":
            split_ratio = 0.9
            num_train = int(len(self.processed_samples) * split_ratio)
            samples = self.processed_samples[:num_train]
        else:
            split_ratio = 0.9
            num_train = int(len(self.processed_samples) * split_ratio)
            samples = self.processed_samples[num_train:]
        
        dataset = ContextCompressionDataset(
            samples=samples,
            tokenizer=self.tokenizer,
            max_length=max_length,
            summary_max_length=summary_max_length
        )
        
        return dataset
    
    def create_hf_dataset(
        self,
        split: str = "train"
    ) -> HFDataset:
        """
        创建HuggingFace Dataset（用于TRL训练）
        
        Args:
            split: 数据集划分
        """
        if not self.processed_samples:
            self.preprocess()
        
        # 划分数据集
        if split == "train":
            split_ratio = 0.9
            num_train = int(len(self.processed_samples) * split_ratio)
            samples = self.processed_samples[:num_train]
        else:
            split_ratio = 0.9
            num_train = int(len(self.processed_samples) * split_ratio)
            samples = self.processed_samples[num_train:]
        
        # 转换为字典列表
        data = []
        for sample in samples:
            context_text = "\n".join([f"{t.speaker}: {t.content}" for t in sample.context_before])
            
            item = {
                "prompt": [
                    {"role": "system", "content": "You are an expert at summarizing conversations."},
                    {"role": "user", "content": f"Please summarize the following conversation:\n\n{context_text}"}
                ],
                "context": context_text,
                "gold_summary": sample.gold_summary or "",
                "sample_id": sample.sample_id,
                "domain": sample.domain,
            }
            data.append(item)
        
        return HFDataset.from_list(data)
    
    def save_samples(self, output_path: str):
        """保存样本到JSON文件"""
        
        data = [s.to_dict() for s in self.processed_samples]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(data)} samples to {output_path}")
    
    def load_samples(self, input_path: str):
        """从JSON文件加载样本"""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.processed_samples = [CompressionSample.from_dict(d) for d in data]
        print(f"Loaded {len(self.processed_samples)} samples from {input_path}")


# =============================================================================
# Helper Functions
# =============================================================================

def create_dataloaders(
    pipeline: DataPipeline,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证DataLoader
    
    Args:
        pipeline: 数据管线实例
        train_batch_size: 训练批次大小
        eval_batch_size: 验证批次大小
        num_workers: 数据加载线程数
    """
    train_dataset = pipeline.create_dataset(split="train")
    eval_dataset = pipeline.create_dataset(split="eval")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, eval_loader


def get_dataset_statistics(samples: List[CompressionSample]) -> Dict:
    """获取数据集统计信息"""
    
    stats = {
        "total_samples": len(samples),
        "domains": defaultdict(int),
        "avg_turns": 0,
        "avg_context_length": 0,
        "has_gold_summary": 0,
    }
    
    total_turns = 0
    total_context_len = 0
    
    for s in samples:
        stats["domains"][s.domain] += 1
        total_turns += len(s.dialogue_history)
        total_context_len += sum(len(t.content) for t in s.context_before)
        if s.gold_summary:
            stats["has_gold_summary"] += 1
    
    if samples:
        stats["avg_turns"] = total_turns / len(samples)
        stats["avg_context_length"] = total_context_len / len(samples)
    
    return dict(stats)
