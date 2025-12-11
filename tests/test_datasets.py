"""Tests for dataset loading module."""

import pytest
from unittest.mock import patch, MagicMock

from grpo_trainer.config import Config, DataConfig
from grpo_trainer.datasets import (
    DatasetLoader,
    SYSTEM_PROMPT,
    XML_COT_FORMAT,
    ONE_SHOT_EXAMPLES,
)


class TestDatasetLoader:
    """Tests for DatasetLoader."""
    
    @pytest.fixture
    def config(self):
        return Config()
    
    def test_supported_datasets(self, config):
        loader = DatasetLoader(config)
        assert "gsm8k" in loader.SUPPORTED_DATASETS
        assert "math" in loader.SUPPORTED_DATASETS
        assert "svamp" in loader.SUPPORTED_DATASETS
    
    def test_system_prompt_exists(self):
        assert SYSTEM_PROMPT is not None
        assert "<reasoning>" in SYSTEM_PROMPT
        assert "<answer>" in SYSTEM_PROMPT
    
    def test_xml_cot_format(self):
        formatted = XML_COT_FORMAT.format(reasoning="test", answer="42")
        assert "<reasoning>" in formatted
        assert "test" in formatted
        assert "<answer>" in formatted
        assert "42" in formatted
    
    def test_one_shot_examples_exist(self):
        assert "gsm8k" in ONE_SHOT_EXAMPLES
        assert "question" in ONE_SHOT_EXAMPLES["gsm8k"]
        assert "reasoning" in ONE_SHOT_EXAMPLES["gsm8k"]
        assert "answer" in ONE_SHOT_EXAMPLES["gsm8k"]


class TestDataExtraction:
    """Tests for data extraction methods."""
    
    @pytest.fixture
    def loader(self):
        config = Config()
        return DatasetLoader(config)
    
    def test_extract_question_gsm8k(self, loader):
        example = {"question": "What is 2+2?"}
        result = loader._extract_question(example, "gsm8k")
        assert result == "What is 2+2?"
    
    def test_extract_question_math(self, loader):
        example = {"problem": "Solve x+1=2"}
        result = loader._extract_question(example, "math")
        assert result == "Solve x+1=2"
    
    def test_extract_answer_gsm8k(self, loader):
        example = {"answer": "The sum is 2+2 = 4 #### 4"}
        result = loader._extract_answer(example, "gsm8k")
        assert result == "4"
