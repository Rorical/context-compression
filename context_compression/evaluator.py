"""
Context Compression - Evaluation Framework
评估框架模块

功能：
1. 任务成功率评估
2. 模型复现率评估（Embedding相似度）
3. 压缩率指标
4. 传统N-gram指标（ROUGE, BLEU）
5. BERTScore评估
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# Evaluation Metrics Data Class
# =============================================================================

@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    
    # 核心指标
    task_success_rate: float = 0.0
    embedding_similarity: float = 0.0
    compression_ratio: float = 0.0
    
    # N-gram指标
    bleu_score: float = 0.0
    rouge1_f: float = 0.0
    rouge2_f: float = 0.0
    rougeL_f: float = 0.0
    
    # BERTScore
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    bertscore_f1: float = 0.0
    
    # 统计信息
    num_samples: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "task_success_rate": self.task_success_rate,
            "embedding_similarity": self.embedding_similarity,
            "compression_ratio": self.compression_ratio,
            "bleu_score": self.bleu_score,
            "rouge1_f": self.rouge1_f,
            "rouge2_f": self.rouge2_f,
            "rougeL_f": self.rougeL_f,
            "bertscore_precision": self.bertscore_precision,
            "bertscore_recall": self.bertscore_recall,
            "bertscore_f1": self.bertscore_f1,
            "num_samples": self.num_samples,
        }
    
    def __str__(self) -> str:
        """格式化输出"""
        lines = [
            "=" * 60,
            "Evaluation Metrics:",
            "=" * 60,
            f"Task Success Rate:    {self.task_success_rate:.4f}",
            f"Embedding Similarity: {self.embedding_similarity:.4f}",
            f"Compression Ratio:    {self.compression_ratio:.4f}",
            "-" * 60,
            f"BLEU Score:           {self.bleu_score:.4f}",
            f"ROUGE-1 F1:           {self.rouge1_f:.4f}",
            f"ROUGE-2 F1:           {self.rouge2_f:.4f}",
            f"ROUGE-L F1:           {self.rougeL_f:.4f}",
            "-" * 60,
            f"BERTScore Precision:  {self.bertscore_precision:.4f}",
            f"BERTScore Recall:     {self.bertscore_recall:.4f}",
            f"BERTScore F1:         {self.bertscore_f1:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# Compression Ratio Calculator
# =============================================================================

class CompressionRatioCalculator:
    """压缩率计算器"""
    
    def __init__(self, tokenizer: Optional[Any] = None):
        """
        初始化
        
        Args:
            tokenizer: 分词器（可选）
        """
        self.tokenizer = tokenizer
    
    def count_tokens(self, text: str) -> int:
        """计算token数"""
        
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        
        # 简单估算
        return len(text.split())
    
    def calculate(
        self,
        original_text: str,
        compressed_text: str
    ) -> Dict[str, float]:
        """
        计算压缩率
        
        Args:
            original_text: 原始文本
            compressed_text: 压缩后文本
            
        Returns:
            压缩率相关指标
        """
        original_tokens = self.count_tokens(original_text)
        compressed_tokens = self.count_tokens(compressed_text)
        
        # 压缩比例 = 压缩后token数 / 原始token数
        compression_rate = compressed_tokens / max(original_tokens, 1)
        
        # 压缩率 = 原始token数 / 压缩后token数
        compression_ratio = original_tokens / max(compressed_tokens, 1)
        
        # 节省比例
        savings_rate = 1 - compression_rate
        
        return {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_rate": compression_rate,
            "compression_ratio": compression_ratio,
            "savings_rate": savings_rate,
            "tokens_saved": original_tokens - compressed_tokens,
        }


# =============================================================================
# Embedding Similarity Evaluator
# =============================================================================

class EmbeddingSimilarityEvaluator:
    """Embedding相似度评估器"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    ):
        """
        初始化
        
        Args:
            model_name: Embedding模型名称
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载embedding模型"""
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        获取文本的embedding向量
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            embedding向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
        
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度 (-1 to 1)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def evaluate(
        self,
        original_text: str,
        compressed_text: str
    ) -> Dict[str, float]:
        """
        评估两个文本的embedding相似度
        
        Args:
            original_text: 原始文本
            compressed_text: 压缩后文本
            
        Returns:
            相似度评估结果
        """
        # 获取embeddings
        embeddings = self.get_embeddings([original_text, compressed_text])
        vec1, vec2 = embeddings[0], embeddings[1]
        
        # 计算余弦相似度
        similarity = self.cosine_similarity(vec1, vec2)
        
        return {
            "similarity": similarity,
            "model": self.model_name,
        }
    
    def evaluate_batch(
        self,
        original_texts: List[str],
        compressed_texts: List[str]
    ) -> List[Dict[str, float]]:
        """
        批量评估embedding相似度
        
        Args:
            original_texts: 原始文本列表
            compressed_texts: 压缩后文本列表
            
        Returns:
            评估结果列表
        """
        assert len(original_texts) == len(compressed_texts)
        
        results = []
        for orig, comp in zip(original_texts, compressed_texts):
            result = self.evaluate(orig, comp)
            results.append(result)
        
        return results


# =============================================================================
# N-gram Similarity Evaluator (ROUGE/BLEU)
# =============================================================================

class NgramSimilarityEvaluator:
    """N-gram相似度评估器 (ROUGE, BLEU)"""
    
    def __init__(self):
        """初始化"""
        self.rouge_available = self._check_rouge()
        self.bleu_available = self._check_bleu()
    
    def _check_rouge(self) -> bool:
        """检查ROUGE是否可用"""
        try:
            from rouge_score import rouge_scorer
            return True
        except ImportError:
            return False
    
    def _check_bleu(self) -> bool:
        """检查BLEU是否可用"""
        try:
            import sacrebleu
            return True
        except ImportError:
            return False
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算ROUGE分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            
        Returns:
            ROUGE分数
        """
        if not self.rouge_available:
            print("Warning: rouge_score not installed")
            return {}
        
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        scores = {
            "rouge1_f": [],
            "rouge2_f": [],
            "rougeL_f": [],
        }
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores["rouge1_f"].append(score["rouge1"].fmeasure)
            scores["rouge2_f"].append(score["rouge2"].fmeasure)
            scores["rougeL_f"].append(score["rougeL"].fmeasure)
        
        return {
            k: sum(v) / len(v) for k, v in scores.items()
        }
    
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算BLEU分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            
        Returns:
            BLEU分数
        """
        if not self.bleu_available:
            print("Warning: sacrebleu not installed")
            return {}
        
        import sacrebleu
        
        # sacrebleu需要references为列表的列表
        refs = [[r] for r in references]
        bleu = sacrebleu.corpus_bleu(predictions, refs)
        
        return {"bleu": bleu.score}


# =============================================================================
# BERTScore Evaluator
# =============================================================================

class BERTScoreEvaluator:
    """BERTScore评估器"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    ):
        """
        初始化
        
        Args:
            model_name: BERT模型名称
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        lang: str = "en"
    ) -> Dict[str, float]:
        """
        计算BERTScore
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            lang: 语言代码
            
        Returns:
            BERTScore结果
        """
        try:
            from bert_score import score
            
            P, R, F1 = score(
                predictions,
                references,
                model_type=self.model_name,
                device=self.device,
                lang=lang,
                verbose=False
            )
            
            return {
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item(),
            }
        
        except ImportError:
            print("Warning: bert-score not installed")
            return {}


# =============================================================================
# Task Success Evaluator
# =============================================================================

class TaskSuccessEvaluator:
    """任务成功率评估器"""
    
    def __init__(self, tokenizer: Optional[Any] = None):
        """
        初始化
        
        Args:
            tokenizer: 分词器
        """
        self.tokenizer = tokenizer
    
    def evaluate(
        self,
        contexts: List[str],
        summaries: List[str],
        answers: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        评估任务成功率
        
        简化的评估：检查摘要是否包含关键信息
        
        Args:
            contexts: 原始上下文列表
            summaries: 摘要列表
            answers: 参考答案列表（可选）
            
        Returns:
            任务成功率
        """
        if answers is None:
            # 如果没有参考答案，使用关键词覆盖度作为代理
            return self._evaluate_by_keyword_coverage(contexts, summaries)
        
        success_count = 0
        
        for context, summary, answer in zip(contexts, summaries, answers):
            # 检查摘要是否包含答案的关键信息
            answer_keywords = set(str(answer).lower().split())
            summary_words = set(summary.lower().split())
            
            if len(answer_keywords) > 0:
                overlap = len(answer_keywords & summary_words)
                coverage = overlap / len(answer_keywords)
                
                if coverage >= 0.5:  # 50%覆盖视为成功
                    success_count += 1
        
        success_rate = success_count / len(contexts) if contexts else 0.0
        
        return {"task_success_rate": success_rate}
    
    def _evaluate_by_keyword_coverage(
        self,
        contexts: List[str],
        summaries: List[str]
    ) -> Dict[str, float]:
        """使用关键词覆盖度评估"""
        
        coverages = []
        
        for context, summary in zip(contexts, summaries):
            # 提取关键词（简单实现）
            context_words = set(self._extract_keywords(context))
            summary_words = set(self._extract_keywords(summary))
            
            if len(context_words) > 0:
                overlap = len(context_words & summary_words)
                coverage = overlap / len(context_words)
                coverages.append(coverage)
        
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0
        
        return {"task_success_rate": avg_coverage}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        
        import re
        
        # 提取字母单词
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # 简单停用词列表
        stopwords = {
            'that', 'with', 'have', 'this', 'will', 'your', 'from', 'they',
            'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very',
            'when', 'come', 'here', 'just', 'like', 'long', 'make', 'over',
            'such', 'take', 'than', 'them', 'well', 'were', 'what', 'would',
        }
        
        return [w for w in words if w not in stopwords]


# =============================================================================
# Main Evaluator
# =============================================================================

class ContextCompressionEvaluator:
    """Context Compression主评估器"""
    
    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    ):
        """
        初始化评估器
        
        Args:
            tokenizer: 分词器
            embedding_model: Embedding模型名称
            device: 运行设备
        """
        self.tokenizer = tokenizer
        
        # 初始化各个评估器
        self.compression_calculator = CompressionRatioCalculator(tokenizer)
        self.embedding_evaluator = EmbeddingSimilarityEvaluator(embedding_model, device)
        self.ngram_evaluator = NgramSimilarityEvaluator()
        self.bertscore_evaluator = BERTScoreEvaluator(device=device)
        self.task_evaluator = TaskSuccessEvaluator(tokenizer)
    
    def evaluate(
        self,
        contexts: List[str],
        predictions: List[str],
        references: Optional[List[str]] = None,
        answers: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """
        执行完整评估
        
        Args:
            contexts: 原始上下文列表
            predictions: 预测摘要列表
            references: 参考摘要列表（可选）
            answers: 参考答案列表（可选）
            
        Returns:
            评估指标
        """
        metrics = EvaluationMetrics()
        metrics.num_samples = len(contexts)
        
        # 1. 计算压缩率
        compression_ratios = []
        for context, pred in zip(contexts, predictions):
            result = self.compression_calculator.calculate(context, pred)
            compression_ratios.append(result["compression_ratio"])
        
        metrics.compression_ratio = sum(compression_ratios) / len(compression_ratios)
        
        # 2. 计算embedding相似度
        embedding_sims = []
        for context, pred in zip(contexts, predictions):
            result = self.embedding_evaluator.evaluate(context, pred)
            embedding_sims.append(result["similarity"])
        
        metrics.embedding_similarity = sum(embedding_sims) / len(embedding_sims)
        
        # 3. 计算任务成功率
        if answers:
            task_result = self.task_evaluator.evaluate(contexts, predictions, answers)
            metrics.task_success_rate = task_result["task_success_rate"]
        else:
            task_result = self.task_evaluator.evaluate(contexts, predictions)
            metrics.task_success_rate = task_result["task_success_rate"]
        
        # 4. 计算ROUGE（如果有参考摘要）
        if references:
            rouge_scores = self.ngram_evaluator.calculate_rouge(predictions, references)
            metrics.rouge1_f = rouge_scores.get("rouge1_f", 0.0)
            metrics.rouge2_f = rouge_scores.get("rouge2_f", 0.0)
            metrics.rougeL_f = rouge_scores.get("rougeL_f", 0.0)
            
            # 5. 计算BLEU
            bleu_scores = self.ngram_evaluator.calculate_bleu(predictions, references)
            metrics.bleu_score = bleu_scores.get("bleu", 0.0)
            
            # 6. 计算BERTScore
            bert_scores = self.bertscore_evaluator.evaluate(predictions, references)
            metrics.bertscore_precision = bert_scores.get("bertscore_precision", 0.0)
            metrics.bertscore_recall = bert_scores.get("bertscore_recall", 0.0)
            metrics.bertscore_f1 = bert_scores.get("bertscore_f1", 0.0)
        
        return metrics
    
    def evaluate_single(
        self,
        context: str,
        prediction: str,
        reference: Optional[str] = None,
        answer: Optional[str] = None
    ) -> Dict[str, float]:
        """
        评估单个样本
        
        Args:
            context: 原始上下文
            prediction: 预测摘要
            reference: 参考摘要（可选）
            answer: 参考答案（可选）
            
        Returns:
            评估指标字典
        """
        contexts = [context]
        predictions = [prediction]
        references = [reference] if reference else None
        answers = [answer] if answer else None
        
        metrics = self.evaluate(contexts, predictions, references, answers)
        
        return metrics.to_dict()
