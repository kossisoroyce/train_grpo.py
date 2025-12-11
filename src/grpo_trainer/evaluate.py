"""Evaluation utilities for GRPO-trained models."""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_trainer.rewards import extract_xml_answer, extract_hash_answer, extract_boxed_answer
from grpo_trainer.datasets import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    total_samples: int
    correct: int
    accuracy: float
    format_correct: int
    format_accuracy: float
    avg_response_length: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class ModelEvaluator:
    """Evaluator for GRPO-trained models."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(
        self,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """Generate a response for a given question."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        split: str = "test",
        num_samples: Optional[int] = None,
        max_new_tokens: int = 512,
        verbose: bool = False,
    ) -> EvaluationResult:
        """Evaluate model on a dataset."""
        # Load dataset
        logger.info(f"Loading {dataset_name} ({split})")
        
        dataset_configs = {
            "gsm8k": ("openai/gsm8k", "main", "question", lambda x: extract_hash_answer(x["answer"])),
            "math": ("hendrycks/competition_math", None, "problem", lambda x: extract_boxed_answer(x["solution"])),
        }
        
        if dataset_name.lower() in dataset_configs:
            ds_path, ds_subset, q_field, answer_fn = dataset_configs[dataset_name.lower()]
            if ds_subset:
                dataset = load_dataset(ds_path, ds_subset)[split]
            else:
                dataset = load_dataset(ds_path)[split]
        else:
            dataset = load_dataset(dataset_name)[split]
            q_field = "question"
            answer_fn = lambda x: str(x.get("answer", ""))
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        # Evaluate
        correct = 0
        format_correct = 0
        total_length = 0
        results = []
        
        for example in tqdm(dataset, desc="Evaluating"):
            question = example[q_field]
            gold_answer = answer_fn(example)
            
            response = self.generate_response(question, max_new_tokens=max_new_tokens)
            extracted = extract_xml_answer(response)
            
            is_correct = extracted == gold_answer
            has_format = "<reasoning>" in response and "</reasoning>" in response and \
                        "<answer>" in response and "</answer>" in response
            
            if is_correct:
                correct += 1
            if has_format:
                format_correct += 1
            total_length += len(response)
            
            if verbose:
                results.append({
                    "question": question,
                    "gold": gold_answer,
                    "response": response,
                    "extracted": extracted,
                    "correct": is_correct,
                    "has_format": has_format,
                })
        
        total = len(dataset)
        
        return EvaluationResult(
            total_samples=total,
            correct=correct,
            accuracy=correct / total if total > 0 else 0,
            format_correct=format_correct,
            format_accuracy=format_correct / total if total > 0 else 0,
            avg_response_length=total_length / total if total > 0 else 0,
        )


def run_evaluation(
    model_path: str,
    dataset_name: str = "gsm8k",
    split: str = "test",
    num_samples: Optional[int] = None,
    batch_size: int = 8,
    output_file: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Run evaluation and return results."""
    evaluator = ModelEvaluator(model_path)
    
    results = evaluator.evaluate_dataset(
        dataset_name=dataset_name,
        split=split,
        num_samples=num_samples,
        verbose=verbose,
    )
    
    results_dict = results.to_dict()
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return results_dict


def main():
    """CLI entry point for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a GRPO-trained model")
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("--dataset", "-d", default="gsm8k", help="Dataset name")
    parser.add_argument("--split", "-s", default="test", help="Dataset split")
    parser.add_argument("--num-samples", "-n", type=int, help="Number of samples")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    results = run_evaluation(
        model_path=args.model_path,
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        output_file=args.output,
        verbose=args.verbose,
    )
    
    print("\nEvaluation Results:")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total_samples']})")
    print(f"  Format Accuracy: {results['format_accuracy']:.4f}")
    print(f"  Avg Response Length: {results['avg_response_length']:.1f}")


if __name__ == "__main__":
    main()
