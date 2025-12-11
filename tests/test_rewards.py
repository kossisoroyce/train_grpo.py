"""Tests for reward functions."""

import pytest

from grpo_trainer.rewards import (
    extract_xml_answer,
    extract_hash_answer,
    extract_boxed_answer,
    CorrectnessReward,
    IntegerReward,
    FormatReward,
    XMLCountReward,
    LengthPenaltyReward,
    ReasoningQualityReward,
    RewardManager,
)


class TestExtractFunctions:
    """Tests for answer extraction functions."""
    
    def test_extract_xml_answer(self):
        text = "<reasoning>some reasoning</reasoning><answer>42</answer>"
        assert extract_xml_answer(text) == "42"
    
    def test_extract_xml_answer_multiline(self):
        text = """<reasoning>
        Step 1: Do something
        Step 2: Calculate
        </reasoning>
        <answer>
        100
        </answer>"""
        assert extract_xml_answer(text) == "100"
    
    def test_extract_xml_answer_missing(self):
        text = "No XML here"
        assert extract_xml_answer(text) == ""
    
    def test_extract_hash_answer(self):
        text = "The answer is calculated as follows... #### 42"
        assert extract_hash_answer(text) == "42"
    
    def test_extract_hash_answer_none(self):
        text = "No hash answer here"
        assert extract_hash_answer(text) is None
    
    def test_extract_boxed_answer(self):
        text = "Therefore, the answer is \\boxed{42}"
        assert extract_boxed_answer(text) == "42"
    
    def test_extract_boxed_answer_none(self):
        text = "No boxed answer"
        assert extract_boxed_answer(text) is None


class TestCorrectnessReward:
    """Tests for CorrectnessReward."""
    
    def test_correct_answer(self):
        reward = CorrectnessReward(weight=2.0)
        completions = [[{"content": "<reasoning>test</reasoning><answer>42</answer>"}]]
        answer = ["42"]
        
        result = reward.compute(completions, answer=answer)
        assert result[0] == 1.0
    
    def test_incorrect_answer(self):
        reward = CorrectnessReward(weight=2.0)
        completions = [[{"content": "<reasoning>test</reasoning><answer>41</answer>"}]]
        answer = ["42"]
        
        result = reward.compute(completions, answer=answer)
        assert result[0] == 0.0
    
    def test_weight_applied(self):
        reward = CorrectnessReward(weight=3.0)
        completions = [[{"content": "<reasoning>test</reasoning><answer>42</answer>"}]]
        answer = ["42"]
        
        result = reward(completions, answer=answer)
        assert result[0] == 3.0


class TestIntegerReward:
    """Tests for IntegerReward."""
    
    def test_integer_answer(self):
        reward = IntegerReward(weight=0.5)
        completions = [[{"content": "<answer>42</answer>"}]]
        
        result = reward.compute(completions)
        assert result[0] == 1.0
    
    def test_non_integer_answer(self):
        reward = IntegerReward(weight=0.5)
        completions = [[{"content": "<answer>forty-two</answer>"}]]
        
        result = reward.compute(completions)
        assert result[0] == 0.0
    
    def test_negative_integer(self):
        reward = IntegerReward(weight=0.5, allow_negative=True)
        completions = [[{"content": "<answer>-42</answer>"}]]
        
        result = reward.compute(completions)
        assert result[0] == 1.0


class TestFormatReward:
    """Tests for FormatReward."""
    
    def test_valid_format(self):
        reward = FormatReward(weight=0.5, strict=False)
        completions = [[{"content": "<reasoning>test</reasoning><answer>42</answer>"}]]
        
        result = reward.compute(completions)
        assert result[0] == 1.0
    
    def test_invalid_format(self):
        reward = FormatReward(weight=0.5, strict=False)
        completions = [[{"content": "Just a plain answer: 42"}]]
        
        result = reward.compute(completions)
        assert result[0] == 0.0


class TestXMLCountReward:
    """Tests for XMLCountReward."""
    
    def test_perfect_format(self):
        reward = XMLCountReward(weight=0.5)
        text = "<reasoning>\ntest\n</reasoning>\n<answer>\n42\n</answer>"
        completions = [[{"content": text}]]
        
        result = reward.compute(completions)
        assert result[0] > 0.0
    
    def test_missing_tags(self):
        reward = XMLCountReward(weight=0.5)
        completions = [[{"content": "No tags here"}]]
        
        result = reward.compute(completions)
        assert result[0] == 0.0


class TestLengthPenaltyReward:
    """Tests for LengthPenaltyReward."""
    
    def test_short_response(self):
        reward = LengthPenaltyReward(weight=0.1, max_length=100)
        completions = [[{"content": "Short response"}]]
        
        result = reward.compute(completions)
        assert result[0] == 0.0
    
    def test_long_response(self):
        reward = LengthPenaltyReward(weight=0.1, max_length=10)
        completions = [[{"content": "This is a much longer response that exceeds the limit"}]]
        
        result = reward.compute(completions)
        assert result[0] < 0.0


class TestRewardManager:
    """Tests for RewardManager."""
    
    def test_add_rewards(self):
        manager = RewardManager()
        manager.add_reward(CorrectnessReward(weight=2.0))
        manager.add_reward(FormatReward(weight=0.5))
        
        assert len(manager.rewards) == 2
    
    def test_get_reward_functions(self):
        manager = RewardManager()
        manager.add_reward(CorrectnessReward(weight=2.0))
        
        funcs = manager.get_reward_functions()
        assert len(funcs) == 1
        assert callable(funcs[0])
