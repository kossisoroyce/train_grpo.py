"""Dataset loading and preprocessing for GRPO training."""

import logging
from typing import Optional
from datasets import load_dataset, Dataset, DatasetDict
from grpo_trainer.rewards import extract_hash_answer, extract_boxed_answer

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# One-shot examples for different datasets
ONE_SHOT_EXAMPLES = {
    "gsm8k": {
        "question": "What is the largest single-digit prime number?",
        "reasoning": "9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
        "answer": "7"
    },
    "math": {
        "question": "Find the value of x if 2x + 5 = 13.",
        "reasoning": "Subtract 5 from both sides: 2x = 8. Divide both sides by 2: x = 4.",
        "answer": "4"
    },
    "svamp": {
        "question": "If you have 5 apples and give away 2, how many do you have left?",
        "reasoning": "Starting with 5 apples and giving away 2 means 5 - 2 = 3 apples remain.",
        "answer": "3"
    },
    "asdiv": {
        "question": "A store has 24 toys. If they sell 8 toys, how many are left?",
        "reasoning": "The store starts with 24 toys. After selling 8: 24 - 8 = 16 toys remain.",
        "answer": "16"
    },
    "aqua": {
        "question": "What is 15% of 200?",
        "reasoning": "15% means 15/100 = 0.15. So 0.15 × 200 = 30.",
        "answer": "30"
    },
}

# Few-shot examples for more complex prompting
FEW_SHOT_EXAMPLES = {
    "gsm8k": [
        {
            "question": "What is the largest single-digit prime number?",
            "reasoning": "9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
            "answer": "7"
        },
        {
            "question": "If a rectangle has length 5 and width 3, what is its area?",
            "reasoning": "The area of a rectangle is length × width = 5 × 3 = 15.",
            "answer": "15"
        },
        {
            "question": "What is 25% of 80?",
            "reasoning": "25% means 25/100 = 0.25. So 0.25 × 80 = 20.",
            "answer": "20"
        },
    ],
}


class DatasetLoader:
    """Handles loading and preprocessing of various math datasets."""
    
    SUPPORTED_DATASETS = {
        "gsm8k": ("openai/gsm8k", "main"),
        "math": ("hendrycks/competition_math", None),
        "svamp": ("ChilleD/SVAMP", None),
        "asdiv": ("MU-NLPC/Calc-asdiv_a", None),
        "aqua": ("aqua_rat", "raw"),
        "mawps": ("MU-NLPC/Calc-mawps", None),
    }
    
    def __init__(self, config):
        self.config = config
        self.system_prompt = SYSTEM_PROMPT
    
    def load(self) -> Dataset:
        """Load and prepare the dataset."""
        data_config = self.config.data
        
        if data_config.custom_path:
            return self._load_custom(data_config.custom_path)
        
        dataset_name = data_config.name.lower()
        
        if dataset_name not in self.SUPPORTED_DATASETS:
            logger.warning(
                f"Dataset '{dataset_name}' not in supported list. "
                f"Attempting to load from Hugging Face Hub directly."
            )
            return self._load_generic(data_config.name, data_config.subset, data_config.split)
        
        loader_method = getattr(self, f"_load_{dataset_name}", None)
        if loader_method:
            return loader_method(data_config)
        
        return self._load_generic(
            self.SUPPORTED_DATASETS[dataset_name][0],
            self.SUPPORTED_DATASETS[dataset_name][1],
            data_config.split
        )
    
    def _load_custom(self, path: str) -> Dataset:
        """Load a custom dataset from local path."""
        logger.info(f"Loading custom dataset from {path}")
        data = load_dataset("json", data_files=path)["train"]
        return self._apply_formatting(data, "custom")
    
    def _load_generic(self, name: str, subset: Optional[str], split: str) -> Dataset:
        """Load a generic dataset from Hugging Face Hub."""
        logger.info(f"Loading dataset {name} (subset={subset}, split={split})")
        try:
            if subset:
                data = load_dataset(name, subset)[split]
            else:
                data = load_dataset(name)[split]
            return data
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _load_gsm8k(self, data_config) -> Dataset:
        """Load and prepare GSM8K dataset."""
        logger.info("Loading GSM8K dataset")
        
        try:
            data = load_dataset("openai/gsm8k", "main")[data_config.split]
        except Exception as e:
            logger.error(f"Failed to load GSM8K dataset: {e}")
            raise
        
        return self._apply_formatting(data, "gsm8k")
    
    def _load_math(self, data_config) -> Dataset:
        """Load and prepare MATH dataset."""
        logger.info("Loading MATH competition dataset")
        
        try:
            data = load_dataset("hendrycks/competition_math")[data_config.split]
        except Exception as e:
            logger.error(f"Failed to load MATH dataset: {e}")
            raise
        
        return self._apply_formatting(data, "math")
    
    def _load_svamp(self, data_config) -> Dataset:
        """Load and prepare SVAMP dataset."""
        logger.info("Loading SVAMP dataset")
        
        try:
            data = load_dataset("ChilleD/SVAMP")[data_config.split]
        except Exception as e:
            logger.error(f"Failed to load SVAMP dataset: {e}")
            raise
        
        return self._apply_formatting(data, "svamp")
    
    def _load_aqua(self, data_config) -> Dataset:
        """Load and prepare AQuA-RAT dataset."""
        logger.info("Loading AQuA-RAT dataset")
        
        try:
            data = load_dataset("aqua_rat", "raw")[data_config.split]
        except Exception as e:
            logger.error(f"Failed to load AQuA dataset: {e}")
            raise
        
        return self._apply_formatting(data, "aqua")
    
    def _apply_formatting(self, data: Dataset, dataset_type: str) -> Dataset:
        """Apply chat formatting to dataset."""
        data_config = self.config.data
        
        def format_example(x):
            prompt = [{"role": "system", "content": self.system_prompt}]
            
            # Add one-shot or few-shot examples
            if data_config.use_few_shot and dataset_type in FEW_SHOT_EXAMPLES:
                examples = FEW_SHOT_EXAMPLES[dataset_type][:data_config.num_few_shot_examples]
                for ex in examples:
                    prompt.extend([
                        {"role": "user", "content": ex["question"]},
                        {"role": "assistant", "content": XML_COT_FORMAT.format(
                            reasoning=ex["reasoning"],
                            answer=ex["answer"]
                        )}
                    ])
            elif data_config.use_one_shot and dataset_type in ONE_SHOT_EXAMPLES:
                ex = ONE_SHOT_EXAMPLES[dataset_type]
                prompt.extend([
                    {"role": "user", "content": ex["question"]},
                    {"role": "assistant", "content": XML_COT_FORMAT.format(
                        reasoning=ex["reasoning"],
                        answer=ex["answer"]
                    )}
                ])
            
            # Add the actual question
            question = self._extract_question(x, dataset_type)
            prompt.append({"role": "user", "content": question})
            
            # Extract answer
            answer = self._extract_answer(x, dataset_type)
            
            return {"prompt": prompt, "answer": answer}
        
        formatted_data = data.map(format_example)
        
        # Apply max samples limit
        if data_config.max_samples and len(formatted_data) > data_config.max_samples:
            if data_config.shuffle:
                formatted_data = formatted_data.shuffle(seed=data_config.seed)
            formatted_data = formatted_data.select(range(data_config.max_samples))
        elif data_config.shuffle:
            formatted_data = formatted_data.shuffle(seed=data_config.seed)
        
        return formatted_data
    
    def _extract_question(self, example: dict, dataset_type: str) -> str:
        """Extract question from example based on dataset type."""
        if dataset_type == "gsm8k":
            return example["question"]
        elif dataset_type == "math":
            return example["problem"]
        elif dataset_type == "svamp":
            return example.get("question", example.get("Body", "") + " " + example.get("Question", ""))
        elif dataset_type == "aqua":
            q = example["question"]
            options = example.get("options", [])
            if options:
                q += "\n" + "\n".join(options)
            return q
        elif dataset_type == "custom":
            return example.get("question", example.get("problem", example.get("input", "")))
        else:
            return example.get("question", "")
    
    def _extract_answer(self, example: dict, dataset_type: str) -> str:
        """Extract answer from example based on dataset type."""
        if dataset_type == "gsm8k":
            return extract_hash_answer(example["answer"])
        elif dataset_type == "math":
            return extract_boxed_answer(example["solution"]) or example.get("answer", "")
        elif dataset_type == "svamp":
            return str(example.get("Answer", ""))
        elif dataset_type == "aqua":
            return example.get("correct", "")
        elif dataset_type == "custom":
            return str(example.get("answer", example.get("solution", example.get("output", ""))))
        else:
            return str(example.get("answer", ""))


def load_dataset_for_training(config) -> Dataset:
    """Convenience function to load dataset for training."""
    loader = DatasetLoader(config)
    return loader.load()


def load_eval_dataset(config) -> Optional[Dataset]:
    """Load evaluation dataset if specified."""
    if not config.data.eval_split:
        return None
    
    # Create a modified config for eval split
    eval_config = config
    original_split = config.data.split
    config.data.split = config.data.eval_split
    
    loader = DatasetLoader(eval_config)
    eval_data = loader.load()
    
    # Restore original split
    config.data.split = original_split
    
    return eval_data
