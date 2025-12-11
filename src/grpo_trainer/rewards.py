"""Reward functions for GRPO training."""

import re
import logging
from abc import ABC, abstractmethod
from typing import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def extract_xml_answer(text: str) -> str:
    """Extracts the answer from XML-formatted text."""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        logger.warning("Failed to extract answer from XML format.")
        return ""


def extract_hash_answer(text: str) -> str | None:
    """Extracts the answer from a hash-formatted string (GSM8K format)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_boxed_answer(text: str) -> str | None:
    """Extracts the answer from LaTeX boxed format (MATH dataset)."""
    pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


class BaseReward(ABC):
    """Base class for reward functions."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    @abstractmethod
    def compute(self, completions: list, **kwargs) -> list[float]:
        """Compute rewards for a batch of completions."""
        pass
    
    def __call__(self, completions: list, **kwargs) -> list[float]:
        """Apply weight to computed rewards."""
        raw_rewards = self.compute(completions, **kwargs)
        return [r * self.weight for r in raw_rewards]


class CorrectnessReward(BaseReward):
    """Rewards correct answers."""
    
    def __init__(self, weight: float = 2.0, partial_credit: bool = False):
        super().__init__(weight)
        self.partial_credit = partial_credit
    
    def compute(self, completions: list, prompts: list = None, answer: list = None, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        
        if prompts:
            q = prompts[0][-1]['content']
            logger.info(
                f"Question:\n{q}\nAnswer:\n{answer[0]}\n"
                f"Response:\n{responses[0]}\nExtracted:\n{extracted_responses[0]}"
            )
        
        rewards = []
        for r, a in zip(extracted_responses, answer):
            if r == a:
                rewards.append(1.0)
            elif self.partial_credit:
                # Partial credit for numeric proximity
                try:
                    r_num = float(r.replace(",", ""))
                    a_num = float(a.replace(",", ""))
                    if abs(r_num - a_num) / max(abs(a_num), 1e-8) < 0.01:
                        rewards.append(0.5)
                    else:
                        rewards.append(0.0)
                except (ValueError, AttributeError):
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        
        return rewards


class IntegerReward(BaseReward):
    """Rewards responses that produce integer/digit answers."""
    
    def __init__(self, weight: float = 0.5, allow_negative: bool = True):
        super().__init__(weight)
        self.allow_negative = allow_negative
    
    def compute(self, completions: list, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        
        rewards = []
        for r in extracted_responses:
            r_clean = r.lstrip("-") if self.allow_negative else r
            if r_clean.isdigit():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards


class FormatReward(BaseReward):
    """Rewards responses that follow the expected XML format."""
    
    def __init__(self, weight: float = 0.5, strict: bool = False):
        super().__init__(weight)
        self.strict = strict
        self.pattern = (
            r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
            if strict
            else r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        )
    
    def compute(self, completions: list, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(self.pattern, r, re.DOTALL) for r in responses]
        return [1.0 if match else 0.0 for match in matches]


class XMLCountReward(BaseReward):
    """Rewards based on XML tag counts and penalizes extra content."""
    
    def __init__(self, weight: float = 0.5, tag_value: float = 0.125, penalty_rate: float = 0.001):
        super().__init__(weight)
        self.tag_value = tag_value
        self.penalty_rate = penalty_rate
    
    def _count_xml(self, text: str) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += self.tag_value
        if text.count("\n</reasoning>\n") == 1:
            count += self.tag_value
        if text.count("\n<answer>\n") == 1:
            count += self.tag_value
            count -= len(text.split("\n</answer>\n")[-1]) * self.penalty_rate
        if text.count("\n</answer>") == 1:
            count += self.tag_value
            count -= (len(text.split("\n</answer>")[-1]) - 1) * self.penalty_rate
        return max(count, 0.0)
    
    def compute(self, completions: list, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [self._count_xml(c) for c in contents]


class LengthPenaltyReward(BaseReward):
    """Penalizes overly long responses."""
    
    def __init__(self, weight: float = 0.1, max_length: int = 1024, penalty_type: str = "linear"):
        super().__init__(weight)
        self.max_length = max_length
        self.penalty_type = penalty_type
    
    def compute(self, completions: list, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for r in responses:
            length = len(r)
            if length <= self.max_length:
                rewards.append(0.0)
            else:
                excess = length - self.max_length
                if self.penalty_type == "linear":
                    penalty = -excess / self.max_length
                elif self.penalty_type == "quadratic":
                    penalty = -(excess / self.max_length) ** 2
                else:
                    penalty = -1.0 if excess > 0 else 0.0
                rewards.append(penalty)
        
        return rewards


class ReasoningQualityReward(BaseReward):
    """Rewards quality indicators in reasoning (step-by-step, calculations, etc.)."""
    
    def __init__(self, weight: float = 0.3, reasoning_start: str = "<reasoning>", reasoning_end: str = "</reasoning>"):
        super().__init__(weight)
        self.reasoning_start = reasoning_start.lower()
        self.reasoning_end = reasoning_end.lower()
        self.quality_indicators = [
            (r"\d+\s*[\+\-\*\/\=]\s*\d+", 0.2),  # Math operations
            (r"(step|first|then|next|finally)", 0.15),  # Sequential reasoning
            (r"(therefore|thus|so|hence)", 0.15),  # Logical connectors
            (r"(because|since|given)", 0.1),  # Causal reasoning
        ]
    
    def compute(self, completions: list, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for r in responses:
            # Extract reasoning section
            reasoning = ""
            r_lower = r.lower()
            if self.reasoning_start in r_lower and self.reasoning_end in r_lower:
                reasoning = r_lower.split(self.reasoning_start)[1].split(self.reasoning_end)[0]
            
            score = 0.0
            for pattern, value in self.quality_indicators:
                if re.search(pattern, reasoning, re.IGNORECASE):
                    score += value
            
            rewards.append(min(score, 1.0))
        
        return rewards


class GibberishPenaltyReward(BaseReward):
    """Penalizes gibberish outputs common in VLMs (e.g., addCriterion repetition).
    
    This addresses issues in vision models like Qwen2.5-VL where the model
    can produce repetitive gibberish tokens during RL training.
    See: https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl
    """
    
    def __init__(
        self, 
        weight: float = 1.0, 
        penalty: float = -2.0,
        gibberish_patterns: list[str] = None,
        threshold: float = 0.5,
    ):
        super().__init__(weight)
        self.penalty = penalty
        self.threshold = threshold
        self.gibberish_patterns = gibberish_patterns or [
            "addCriterion",
            "\n\n\n",  # Excessive newlines
            "................",  # Repeated punctuation
        ]
    
    def compute(self, completions: list, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for r in responses:
            if len(r) == 0:
                rewards.append(0.0)
                continue
            
            # Remove all gibberish patterns
            cleaned = r
            for pattern in self.gibberish_patterns:
                cleaned = cleaned.replace(pattern, "")
            
            # Calculate ratio of removed content
            removed_ratio = (len(r) - len(cleaned)) / len(r)
            
            if removed_ratio >= self.threshold:
                rewards.append(self.penalty)
            else:
                rewards.append(0.0)
        
        return rewards


class CustomDelimiterFormatReward(BaseReward):
    """Rewards responses with custom start/end delimiters for reasoning and answer.
    
    Useful for different formatting styles like:
    - <REASONING>...</REASONING><SOLUTION>...</SOLUTION>
    - [THINK]...[/THINK][ANSWER]...[/ANSWER]
    """
    
    def __init__(
        self,
        weight: float = 0.5,
        reasoning_start: str = "<REASONING>",
        reasoning_end: str = "</REASONING>",
        answer_start: str = "<SOLUTION>",
        answer_end: str = "</SOLUTION>",
    ):
        super().__init__(weight)
        self.reasoning_start = reasoning_start
        self.reasoning_end = reasoning_end
        self.answer_start = answer_start
        self.answer_end = answer_end
    
    def compute(self, completions: list, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for r in responses:
            score = 0.0
            
            # Check reasoning delimiters
            reasoning_matches = re.findall(
                f"{re.escape(self.reasoning_start)}(.*?){re.escape(self.reasoning_end)}",
                r, re.DOTALL
            )
            if len(reasoning_matches) == 1:
                score += 0.5
            
            # Check answer delimiters
            answer_matches = re.findall(
                f"{re.escape(self.answer_start)}(.*?){re.escape(self.answer_end)}",
                r, re.DOTALL
            )
            if len(answer_matches) == 1:
                score += 0.5
            
            rewards.append(score)
        
        return rewards


@dataclass
class RewardFunction:
    """Wrapper for reward function with metadata."""
    name: str
    func: Callable
    weight: float
    enabled: bool = True


class RewardManager:
    """Manages multiple reward functions."""
    
    def __init__(self, config=None):
        self.rewards: list[BaseReward] = []
        self.config = config
        
        if config:
            self._init_from_config(config)
    
    def _init_from_config(self, config):
        """Initialize reward functions from configuration."""
        rc = config.reward
        
        if rc.correctness_weight > 0:
            self.add_reward(CorrectnessReward(weight=rc.correctness_weight))
        
        # Use custom delimiter format or standard XML format
        if rc.use_custom_delimiters and rc.format_weight > 0:
            self.add_reward(CustomDelimiterFormatReward(
                weight=rc.format_weight,
                reasoning_start=rc.reasoning_start,
                reasoning_end=rc.reasoning_end,
                answer_start=rc.answer_start,
                answer_end=rc.answer_end,
            ))
        elif rc.format_weight > 0:
            self.add_reward(FormatReward(weight=rc.format_weight, strict=rc.strict_format))
        
        if rc.integer_weight > 0:
            self.add_reward(IntegerReward(weight=rc.integer_weight))
        
        if rc.xml_count_weight > 0:
            self.add_reward(XMLCountReward(weight=rc.xml_count_weight))
        
        if rc.length_penalty_weight > 0:
            self.add_reward(LengthPenaltyReward(
                weight=rc.length_penalty_weight,
                max_length=rc.max_length_for_penalty
            ))
        
        if rc.reasoning_quality_weight > 0:
            self.add_reward(ReasoningQualityReward(
                weight=rc.reasoning_quality_weight,
                reasoning_start=rc.reasoning_start if rc.use_custom_delimiters else "<reasoning>",
                reasoning_end=rc.reasoning_end if rc.use_custom_delimiters else "</reasoning>",
            ))
        
        # VLM gibberish penalty
        if rc.gibberish_penalty_weight > 0:
            self.add_reward(GibberishPenaltyReward(
                weight=rc.gibberish_penalty_weight,
                threshold=rc.gibberish_threshold,
            ))
    
    def add_reward(self, reward: BaseReward) -> None:
        """Add a reward function."""
        self.rewards.append(reward)
    
    def get_reward_functions(self) -> list[Callable]:
        """Get list of reward functions for GRPOTrainer."""
        return [r.__call__ for r in self.rewards]
    
    def compute_all(self, completions: list, **kwargs) -> dict[str, list[float]]:
        """Compute all rewards and return as dictionary."""
        results = {}
        for reward in self.rewards:
            name = reward.__class__.__name__
            results[name] = reward(completions, **kwargs)
        return results


def create_reward_functions(config) -> list[Callable]:
    """Create reward functions from configuration (convenience function)."""
    manager = RewardManager(config)
    return manager.get_reward_functions()
