#!/usr/bin/env python3
"""
Glasgow Coma Scale VLM Evaluation Script

This script evaluates various Vision Language Models (VLMs) on their ability
to assess Glasgow Coma Scale motor responses from clinical images.
"""

import os
import json
import yaml
import base64
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv is optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from openai import OpenAI

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class GCSEvaluator:
    """Evaluates VLMs on Glasgow Coma Scale assessment."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize evaluator with configuration."""
        self.config = self._load_config(config_path)
        self.client = self._initialize_openai_client()
        self.dataset_path = Path(self.config["dataset"]["base_path"])
        self.results_dir = Path(self.config["evaluation"]["output_dir"])
        self.results_dir.mkdir(exist_ok=True)
        self.logging_cfg = self.config.get("evaluation", {}).get("logging", {})
        self.logs_enabled = self.logging_cfg.get("enabled", False)
        if self.logs_enabled:
            self.logs_dir = Path(self.logging_cfg.get("dir", "logs"))
            self.logs_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = self.logging_cfg.get("file_prefix", "llm_interactions")
            self.log_file = self.logs_dir / f"{prefix}_{ts}.jsonl"
            self.redact_base64 = bool(self.logging_cfg.get("redact_base64_images", False))
            self.truncate_chars = int(self.logging_cfg.get("truncate_response_chars", 0) or 0)
        else:
            self.logs_dir = None
            self.log_file = None
            self.redact_base64 = False
            self.truncate_chars = 0

    def _maybe_redact_messages(self, messages: List[Dict]) -> List[Dict]:
        if not self.redact_base64:
            return messages
        redacted: List[Dict] = []
        for msg in messages:
            new_msg = {"role": msg.get("role"), "content": []}
            for part in msg.get("content", []):
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:image") and "," in url:
                        b64 = url.split(",", 1)[1]
                        part = {
                            "type": "image_url",
                            "image_url": {"url": f"<redacted base64, length={len(b64)}>"}
                        }
                new_msg["content"].append(part)
            redacted.append(new_msg)
        return redacted

    def _log_interaction(self, entry: Dict) -> None:
        if not self.logs_enabled or not self.log_file:
            return
        try:
            with open(self.log_file, "a") as lf:
                lf.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def _truncate(self, text: Optional[str]) -> Optional[str]:
        if not isinstance(text, str):
            return text
        if self.truncate_chars and len(text) > self.truncate_chars:
            return text[: self.truncate_chars] + "...<truncated>"
        return text

    def _extract_message_text(self, choice) -> str:
        """Extract text content from a chat completion choice, handling list segments."""
        def join_segments(segments):
            return "".join(seg for seg in segments if isinstance(seg, str) and seg).strip()

        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif "text" in part:
                            parts.append(part["text"])
                    else:
                        part_text = getattr(part, "text", None)
                        if part_text:
                            parts.append(part_text)
                text = join_segments(parts)
                if text:
                    return text
            elif isinstance(content, str) and content.strip():
                return content.strip()
            elif hasattr(message, "text"):
                text_attr = getattr(message, "text", None)
                if isinstance(text_attr, str) and text_attr.strip():
                    return text_attr.strip()

        choice_content = getattr(choice, "content", None)
        if isinstance(choice_content, list):
            parts = []
            for part in choice_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                else:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        parts.append(part_text)
            text = join_segments(parts)
            if text:
                return text
        elif isinstance(choice_content, str) and choice_content.strip():
            return choice_content.strip()

        return ""
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle environment variable substitution
        if config.get("openai", {}).get("api_key", "").startswith("${"):
            env_var = config["openai"]["api_key"][2:-1]
            config["openai"]["api_key"] = os.getenv(env_var, "")
        
        return config
    
    def _initialize_openai_client(self) -> OpenAI:
        """Initialize OpenAI client."""
        openai_config = self.config["openai"]
        api_key = openai_config.get("api_key", "") or os.getenv("OPENAI_API_KEY", "")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in .env file or as environment variable."
            )
        
        client_kwargs = {
            "api_key": api_key,
            "timeout": openai_config.get("timeout", 60),
        }
        
        if openai_config.get("base_url"):
            client_kwargs["base_url"] = openai_config["base_url"]
        
        return OpenAI(**client_kwargs)
    
    def _encode_image(self, image_path: Path) -> Tuple[str, str]:
        """Encode image to base64 and return (base64, mime_type)."""
        ext = image_path.suffix.lower()
        mime = 'image/png'
        if ext in ['.jpg', '.jpeg']:
            mime = 'image/jpeg'
        elif ext == '.gif':
            mime = 'image/gif'
        elif ext == '.webp':
            mime = 'image/webp'
        with open(image_path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode('utf-8')
            return b64, mime
    
    def _get_image_files(self) -> List[Tuple[Path, Dict]]:
        """Get all image files with their category information."""
        image_files = []
        
        for category in self.config["dataset"]["categories"]:
            category_path = self.dataset_path / category["name"]
            if not category_path.exists():
                print(f"Warning: Category path {category_path} does not exist")
                continue
            
            # Get all image files in category
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                for img_file in category_path.glob(ext):
                    image_files.append((img_file, category))
        
        return image_files
    
    def _select_few_shot_examples(self, exclude_path: Optional[Path] = None) -> List[Tuple[Path, Dict]]:
        """Select example images for few-shot prompting."""
        examples_per_category = self.config["few_shot"]["examples_per_category"]
        random_seed = self.config["few_shot"]["random_seed"]
        
        random.seed(random_seed)
        examples = []
        
        for category in self.config["dataset"]["categories"]:
            category_path = self.dataset_path / category["name"]
            if not category_path.exists():
                continue
            
            # Get all image files in category
            category_images = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                category_images.extend(category_path.glob(ext))
            
            # Exclude the current image if provided
            if exclude_path:
                category_images = [img for img in category_images if img != exclude_path]
            
            # Randomly select examples
            selected = random.sample(
                category_images, 
                min(examples_per_category, len(category_images))
            )
            
            for img_path in selected:
                examples.append((img_path, category))
        
        random.seed()  # Reset seed
        return examples
    
    def _create_few_shot_content(self, examples: List[Tuple[Path, Dict]]) -> str:
        """Create few-shot example content for the prompt."""
        example_texts = []
        
        for idx, (img_path, category) in enumerate(examples, 1):
            example_texts.append(
                f"Example {idx}: GCS Motor Score {category['gcs_motor_score']} - {category['description']}\n"
                f"(See image {idx} below)\n"
            )
        
        return "\n".join(example_texts)
    
    def _create_messages(self, image_path: Path, use_few_shot: bool = False, task: str = "6bin") -> List[Dict]:
        """Create messages for API call.
        task: '6bin' uses standard prompts; '3bin' uses 3-bin templates.
        """
        base64_image, base_mime = self._encode_image(image_path)
        
        if use_few_shot:
            template_key = "few_shot_3bin_template" if task == "3bin" else "few_shot_template"
            template_path = Path(self.config["prompting"][template_key])
            examples = self._select_few_shot_examples(exclude_path=image_path)
            few_shot_examples = self._create_few_shot_content(examples)
            with open(template_path, 'r') as tf:
                from jinja2 import Template
                prompt = Template(tf.read()).render(few_shot_examples=few_shot_examples)
            
            content = [{"type": "text", "text": prompt}]
            
            for idx, (img_path, category) in enumerate(examples, 1):
                example_base64, example_mime = self._encode_image(img_path)
                content.append({
                    "type": "text",
                    "text": f"\n[Example Image {idx} - GCS Motor Score {category['gcs_motor_score']}: {category['description']}]"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{example_mime};base64,{example_base64}",
                        "detail": self.config.get("prompting", {}).get("image_detail", "auto")
                    }
                })
            
            content.append({
                "type": "text",
                "text": "\n\nNow evaluate the following test image:"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{base_mime};base64,{base64_image}",
                    "detail": self.config.get("prompting", {}).get("image_detail", "auto")
                }
            })
            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Return one JSON object only: {\n  gcs_motor_score: 1-6,\n  confidence: 0-1,\n  reasoning: string,\n  observed_behaviors: string\n}"}
                    ]
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        else:
            template_key = "zero_shot_3bin_template" if task == "3bin" else "zero_shot_template"
            template_path = Path(self.config["prompting"][template_key])
            with open(template_path, 'r') as tf:
                from jinja2 import Template
                prompt = Template(tf.read()).render()
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Return one JSON object only: {\n  gcs_motor_score: 1-6,\n  confidence: 0-1,\n  reasoning: string,\n  observed_behaviors: string\n}"}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{base_mime};base64,{base64_image}",
                                "detail": self.config.get("prompting", {}).get("image_detail", "auto")
                            }
                        }
                    ]
                }
            ]
        
        return messages
    
    def _call_model(self, model_name: str, messages: List[Dict], model_params: Dict) -> Dict:
        """Call OpenAI API with specified model and parameters.
        Includes adaptive normalization and a single retry if the API rejects a parameter.
        """
        # Normalize parameter names for newer models (e.g., GPT-5)
        normalized_params = dict(model_params)

        # Generic mapping: max_tokens -> max_completion_tokens
        if "max_completion_tokens" not in normalized_params and "max_tokens" in normalized_params:
            normalized_params["max_completion_tokens"] = normalized_params.pop("max_tokens")

        # Model-specific cleanups
        model_lower = model_name.lower()
        if model_lower.startswith("gpt-5"):
            # GPT-5 models may not allow overriding temperature; rely on default
            normalized_params.pop("temperature", None)
            # Some penalty params may be unsupported in certain variants
            # Keep them if present unless API complains; retry will remove them if needed

        def do_request(params: Dict):
            return self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **params
            )

        try:
            self._log_interaction({
                "ts": datetime.now().isoformat(),
                "stage": "request",
                "attempt": 1,
                "model": model_name,
                "params": normalized_params,
                "messages": self._maybe_redact_messages(messages)
            })
            response = do_request(normalized_params)
            choice = response.choices[0]
            content_text = self._extract_message_text(choice)
            if not content_text:
                return {"success": False, "error": "Empty completion content", "response": "", "usage": getattr(response, "usage", None)}
            self._log_interaction({
                "ts": datetime.now().isoformat(),
                "stage": "response",
                "attempt": 1,
                "model": model_name,
                "raw": self._truncate(content_text),
                "usage": getattr(response, "usage", None)
            })
            return {
                "success": True,
                "response": content_text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            error_text = str(e)
            return {"success": False, "error": error_text, "response": None, "usage": None}
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse JSON response from model. Expects a single JSON object."""
        if not isinstance(response_text, str):
            return {"error": "Response is not string", "raw_response": response_text}
        text = response_text.strip()
        # Strip fenced code if present
        if text.startswith("```"):
            first = text.find("\n")
            last = text.rfind("```")
            if first != -1 and last != -1:
                text = text[first+1:last].strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {"error": "Response JSON is not an object", "raw_response": text}
        except Exception as e:
            return {"error": f"JSON parse error: {str(e)}", "raw_response": text}
    
    def evaluate(self, use_few_shot: bool = False, task: str = "6bin") -> Dict:
        """Run evaluation on all images."""
        image_files = self._get_image_files()
        all_results = []
        
        print(f"\n{'='*60}")
        print(f"Starting GCS Evaluation ({task})")
        print(f"Mode: {'Few-Shot' if use_few_shot else 'Zero-Shot'}")
        print(f"Total images: {len(image_files)}")
        print(f"Models: {[m['name'] for m in self.config['models']]}")
        print(f"{'='*60}\n")
        
        for img_idx, (image_path, category) in enumerate(image_files, 1):
            print(f"[{img_idx}/{len(image_files)}] Processing: {image_path.name}")
            print(f"  Category: {category['name']} (Expected Score: {category['gcs_motor_score']})")
            
            # Create messages
            messages = self._create_messages(image_path, use_few_shot=use_few_shot, task=task)
            
            # Evaluate with each model
            for model_config in self.config["models"]:
                model_name = model_config["name"]
                model_params = model_config["parameters"]
                
                print(f"  Testing model: {model_name}...", end=" ")
                
                # Call model
                api_result = self._call_model(model_name, messages, model_params)
                
                if not api_result["success"]:
                    print(f"ERROR: {api_result['error']}")
                    result = {
                        "image_path": str(image_path),
                        "category": category["name"],
                        "expected_score": category["gcs_motor_score"],
                        "model": model_name,
                        "prompting_mode": "few_shot" if use_few_shot else "zero_shot",
                        "success": False,
                        "error": api_result["error"]
                    }
                else:
                    # Parse response
                    parsed = self._parse_response(api_result["response"])
                    # Log parsed summary
                    self._log_interaction({
                        "ts": datetime.now().isoformat(),
                        "stage": "parsed",
                        "model": model_name,
                        "image": str(image_path),
                        "category": category["name"],
                        "expected": category["gcs_motor_score"],
                        "parsed": parsed
                    })
                    
                    predicted_score = parsed.get("gcs_motor_score")
                    if isinstance(predicted_score, str):
                        try:
                            predicted_score = int(predicted_score)
                        except Exception:
                            pass

                    # If task is 3bin and model returned gcs_motor_bin, map accordingly
                    predicted_bin = parsed.get("gcs_motor_bin")
                    if isinstance(predicted_bin, str):
                        try:
                            predicted_bin = int(predicted_bin)
                        except Exception:
                            pass
                    
                    # Calculate 6-bin accuracy
                    is_correct_6bin = predicted_score == category["gcs_motor_score"] if task == "6bin" else False
                    
                    # Calculate 3-bin scores
                    predicted_3bin = predicted_bin if task == "3bin" else self._score_6bin_to_3bin(predicted_score)
                    expected_3bin = self._score_6bin_to_3bin(category["gcs_motor_score"])
                    is_correct_3bin = predicted_3bin == expected_3bin if (predicted_3bin is not None and expected_3bin is not None) else False
                    
                    # Clear, task-specific console output
                    if task == "3bin":
                        mark = '✓' if is_correct_3bin else '✗'
                        print(f"  -> 3-bin: {predicted_3bin}/{expected_3bin} {mark}")
                    else:
                        mark = '✓' if is_correct_6bin else '✗'
                        print(f"  -> 6-bin: {predicted_score}/{category['gcs_motor_score']} {mark}")
                    
                    result = {
                        "image_path": str(image_path),
                        "category": category["name"],
                        "expected_score": category["gcs_motor_score"],
                        "expected_score_3bin": expected_3bin,
                        "model": model_name,
                        "prompting_mode": "few_shot" if use_few_shot else "zero_shot",
                        "success": True,
                        "predicted_score": predicted_score,
                        "predicted_score_3bin": predicted_3bin,
                        "is_correct": is_correct_6bin,  # 6-bin accuracy
                        "is_correct_3bin": is_correct_3bin,  # 3-bin accuracy
                        "confidence": parsed.get("confidence"),
                        "reasoning": parsed.get("reasoning"),
                        "observed_behaviors": parsed.get("observed_behaviors"),
                        "raw_response": api_result["response"],
                        "usage": api_result["usage"]
                    }
                
                all_results.append(result)
            
            print()
        
        return all_results
    
    def _score_6bin_to_3bin(self, score_6bin: Optional[int]) -> Optional[int]:
        """Convert 6-bin GCS score to 3-bin simplified score.
        
        1-bin (Unresponsive/Abnormal): combines scores 1-3
        2-bin (Withdraws/Localizes): combines scores 4-5  
        3-bin (Obeys Commands/Normal): score 6 only
        """
        if score_6bin is None:
            return None
        if score_6bin in [1, 2, 3]:
            return 1  # Unresponsive/Abnormal
        elif score_6bin in [4, 5]:
            return 2  # Withdraws/Localizes
        elif score_6bin == 6:
            return 3  # Obeys Commands/Normal
        return None
    
    def _calculate_metrics(self, results: List[Dict], task: str = "6bin") -> Dict:
        """Calculate evaluation metrics."""
        # Filter successful results
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {"error": "No successful evaluations"}
        
        metrics = {}
        
        for model_config in self.config["models"]:
            model_name = model_config["name"]
            model_results = [r for r in successful_results if r["model"] == model_name]
            
            if not model_results:
                continue
            
            total = len(model_results)
            
            if task == "6bin":
                correct_6bin = sum(1 for r in model_results if r.get("is_correct", False))
                accuracy_6bin = correct_6bin / total if total > 0 else 0
            else:
                correct_6bin = None
                accuracy_6bin = None
            
            correct_3bin = sum(1 for r in model_results if r.get("is_correct_3bin", False))
            accuracy_3bin = correct_3bin / total if total > 0 else 0
            
            category_accuracies = {}
            for category in self.config["dataset"]["categories"]:
                cat_results = [r for r in model_results if r["category"] == category["name"]]
                if cat_results:
                    cat_correct = sum(1 for r in cat_results if r.get("is_correct", False))
                    category_accuracies[category["name"]] = {
                        "accuracy": cat_correct / len(cat_results),
                        "correct": cat_correct,
                        "total": len(cat_results)
                    }
            
            bin_3_accuracies = {}
            for bin_num in [1, 2, 3]:
                bin_results = [r for r in model_results if r.get("expected_score_3bin") == bin_num]
                if bin_results:
                    bin_correct = sum(1 for r in bin_results if r.get("is_correct_3bin", False))
                    bin_label = ["Unresponsive/Abnormal", "Withdraws/Localizes", "Obeys Commands/Normal"][bin_num - 1]
                    bin_3_accuracies[bin_label] = {
                        "accuracy": bin_correct / len(bin_results),
                        "correct": bin_correct,
                        "total": len(bin_results)
                    }
            
            total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in model_results)
            
            metrics[model_name] = {
                "total_images": total,
                "correct": correct_6bin,
                "accuracy": accuracy_6bin,
                "correct_3bin": correct_3bin,
                "accuracy_3bin": accuracy_3bin,
                "total_tokens": total_tokens,
                "category_accuracies": category_accuracies,
                "bin_3_accuracies": bin_3_accuracies
            }
        
        return metrics
    
    def save_results(self, results: List[Dict], use_few_shot: bool = False, task: str = "6bin"):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "few_shot" if use_few_shot else "zero_shot"
        
        # Save individual results
        if self.config["evaluation"]["save_individual_responses"]:
            results_file = self.results_dir / f"results_{mode}_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_file}")
        
        # Calculate and save summary
        if self.config["evaluation"]["save_summary"]:
            metrics = self._calculate_metrics(results, task)
            summary = {
                "timestamp": timestamp,
                "mode": mode,
                "total_images": len([r for r in results if r.get("success", False)]),
                "metrics": metrics
            }
            
            summary_file = self.results_dir / f"summary_{mode}_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Evaluation Summary ({mode.upper()})")
            print(f"{'='*60}")
            for model_name, model_metrics in metrics.items():
                print(f"\nModel: {model_name}")
                if model_metrics['accuracy'] is not None and model_metrics['correct'] is not None:
                    print(f"  6-bin Accuracy: {model_metrics['accuracy']:.2%} ({model_metrics['correct']}/{model_metrics['total_images']})")
                else:
                    print(f"  6-bin Accuracy: N/A (not evaluated for 3-bin prompt)")
                label_3 = "3-bin Accuracy (derived from 6-bin)" if task == "6bin" else "3-bin Accuracy (direct)"
                print(f"  {label_3}: {model_metrics['accuracy_3bin']:.2%} ({model_metrics['correct_3bin']}/{model_metrics['total_images']})")
                print(f"  Total Tokens: {model_metrics['total_tokens']}")
                print(f"\n  Per-Category Accuracy (6-bin):")
                for cat_name, cat_metrics in model_metrics['category_accuracies'].items():
                    print(f"    {cat_name}: {cat_metrics['accuracy']:.2%} ({cat_metrics['correct']}/{cat_metrics['total']})")
                print(f"\n  Per-Bin Accuracy (3-bin):")
                for bin_name, bin_metrics in model_metrics['bin_3_accuracies'].items():
                    print(f"    {bin_name}: {bin_metrics['accuracy']:.2%} ({bin_metrics['correct']}/{bin_metrics['total']})")
            
    def _plot_results(self, results: List[Dict], metrics: Dict, use_few_shot: bool, timestamp: str):
        """Generate visualization plots for evaluation results."""
        mode = "few_shot" if use_few_shot else "zero_shot"
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating plots for {mode} evaluation...")
        
        # 1. Overall accuracy comparison
        self._plot_overall_accuracy(metrics, mode, plots_dir, timestamp)
        
        # 2. Per-category accuracy
        self._plot_category_accuracy(metrics, mode, plots_dir, timestamp)
        
        # 3. Confusion matrices
        self._plot_confusion_matrices(results, metrics, mode, plots_dir, timestamp)
        
        print(f"Plots saved to: {plots_dir}")
    
    def _plot_overall_accuracy(self, metrics: Dict, mode: str, plots_dir: Path, timestamp: str):
        """Plot overall accuracy comparison across models."""
        models = list(metrics.keys())
        accuracies = [metrics[m]["accuracy"] for m in models]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Accuracy bar plot
        bars = ax.bar(models, accuracies, color=sns.color_palette("husl", len(models)))
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title(f'Overall Accuracy by Model ({mode.replace("_", "-").title()})', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 1.1, 0.1)])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"overall_accuracy_{mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_accuracy(self, metrics: Dict, mode: str, plots_dir: Path, timestamp: str):
        """Plot per-category accuracy for each model."""
        models = list(metrics.keys())
        categories = self.config["dataset"]["categories"]
        category_names = [cat["name"] for cat in categories]
        
        # Prepare data
        data = []
        for model in models:
            for cat in category_names:
                cat_metrics = metrics[model]["category_accuracies"].get(cat, {})
                accuracy = cat_metrics.get("accuracy", 0)
                data.append({
                    "Model": model,
                    "Category": cat.replace("_", " ").title(),
                    "Accuracy": accuracy
                })
        
        # Create pivot table
        df = pd.DataFrame(data)
        pivot_data = df.pivot(index="Category", columns="Model", values="Accuracy")
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), max(6, len(categories) * 0.8)))
        sns.heatmap(pivot_data, annot=True, fmt='.2%', cmap='RdYlGn', vmin=0, vmax=1,
                   cbar_kws={'label': 'Accuracy'}, ax=ax, linewidths=0.5)
        ax.set_title(f'Per-Category Accuracy Heatmap ({mode.replace("_", "-").title()})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('GCS Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / f"category_accuracy_{mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(category_names))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            accuracies = [metrics[model]["category_accuracies"].get(cat, {}).get("accuracy", 0) 
                         for cat in category_names]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, accuracies, width, label=model, alpha=0.8)
        
        ax.set_xlabel('GCS Category', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Per-Category Accuracy Comparison ({mode.replace("_", "-").title()})', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([cat.replace("_", " ").title() for cat in category_names], rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 1.1, 0.1)])
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"category_accuracy_bars_{mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, results: List[Dict], metrics: Dict, mode: str, plots_dir: Path, timestamp: str):
        """Plot confusion matrices for each model."""
        models = list(metrics.keys())
        categories = {cat["name"]: cat["gcs_motor_score"] for cat in self.config["dataset"]["categories"]}
        score_to_cat = {v: k for k, v in categories.items()}
        scores = sorted(categories.values())
        
        for model in models:
            model_results = [r for r in results if r.get("model") == model and r.get("success", False)]
            if not model_results:
                continue
            
            # Build confusion matrix
            cm = np.zeros((len(scores), len(scores)), dtype=int)
            for result in model_results:
                expected = result.get("expected_score")
                predicted = result.get("predicted_score")
                if expected is not None and predicted is not None:
                    try:
                        expected_idx = scores.index(expected)
                        predicted_idx = scores.index(predicted)
                        cm[expected_idx, predicted_idx] += 1
                    except ValueError:
                        continue
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=[f'Score {s}' for s in scores],
                       yticklabels=[f'Score {s}' for s in scores],
                       cbar_kws={'label': 'Count'})
            ax.set_xlabel('Predicted Score', fontsize=12)
            ax.set_ylabel('Expected Score', fontsize=12)
            ax.set_title(f'Confusion Matrix - {model} ({mode.replace("_", "-").title()})', 
                        fontsize=14, fontweight='bold')
            
            # Add accuracy text
            total = cm.sum()
            correct = np.trace(cm)
            accuracy = correct / total if total > 0 else 0
            ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}', 
                   transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"confusion_matrix_{model}_{mode}_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_confidence_distribution(self, results: List[Dict], metrics: Dict, mode: str, plots_dir: Path, timestamp: str):
        """Plot confidence score distributions."""
        models = list(metrics.keys())
        
        fig, axes = plt.subplots(len(models), 1, figsize=(12, 4 * len(models)))
        if len(models) == 1:
            axes = [axes]
        
        for idx, model in enumerate(models):
            model_results = [r for r in results if r.get("model") == model and r.get("success", False)]
            confidences = [r.get("confidence") for r in model_results if r.get("confidence") is not None]
            
            if confidences:
                axes[idx].hist(confidences, bins=20, alpha=0.7, color=sns.color_palette("husl", len(models))[idx], edgecolor='black')
                axes[idx].axvline(np.mean(confidences), color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {np.mean(confidences):.2%}')
                axes[idx].set_xlabel('Confidence Score', fontsize=11)
                axes[idx].set_ylabel('Frequency', fontsize=11)
                axes[idx].set_title(f'Confidence Distribution - {model} ({mode.replace("_", "-").title()})', 
                                   fontsize=12, fontweight='bold')
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)
                axes[idx].set_xlim(0, 1)
            else:
                axes[idx].text(0.5, 0.5, 'No confidence data available', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'Confidence Distribution - {model} ({mode.replace("_", "-").title()})', 
                                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"confidence_distribution_{mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()


    def plot_comparison(self, zero_shot_results: List[Dict], few_shot_results: List[Dict]):
        """Compare zero-shot vs few-shot results for both 6-bin and 3-bin systems."""
        zero_metrics = self._calculate_metrics(zero_shot_results)
        few_metrics = self._calculate_metrics(few_shot_results)
        
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        models = list(zero_metrics.keys())
        if not models:
            print("No models found for comparison")
            return
        
        # 6-bin comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35
        
        zero_acc_6bin = [zero_metrics[m]["accuracy"] for m in models]
        few_acc_6bin = [few_metrics[m]["accuracy"] for m in models]
        
        bars1 = ax.bar(x - width/2, zero_acc_6bin, width, label='Zero-Shot', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, few_acc_6bin, width, label='Few-Shot', alpha=0.8, color='#2ecc71')
        
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('Zero-Shot vs Few-Shot Accuracy Comparison (6-bin)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 1.1, 0.1)])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"comparison_zero_vs_few_6bin_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"6-bin comparison plot saved to: {plots_dir / f'comparison_zero_vs_few_6bin_{timestamp}.png'}")
        
        # 3-bin comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35
        
        zero_acc_3bin = [zero_metrics[m]["accuracy_3bin"] for m in models]
        few_acc_3bin = [few_metrics[m]["accuracy_3bin"] for m in models]
        
        bars1 = ax.bar(x - width/2, zero_acc_3bin, width, label='Zero-Shot', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, few_acc_3bin, width, label='Few-Shot', alpha=0.8, color='#2ecc71')
        
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('Zero-Shot vs Few-Shot Accuracy Comparison (3-bin)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 1.1, 0.1)])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"comparison_zero_vs_few_3bin_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"3-bin comparison plot saved to: {plots_dir / f'comparison_zero_vs_few_3bin_{timestamp}.png'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate VLMs on Glasgow Coma Scale assessment")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero_shot", "few_shot", "both"],
        default="both",
        help="Prompting mode to use"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["6bin", "3bin", "both"],
        default="6bin",
        help="Scoring task to run (6bin standard, 3bin simplified, or both)"
    )
    
    args = parser.parse_args()
    
    evaluator = GCSEvaluator(config_path=args.config)
    
    results_zero = None
    results_few = None
    
    if args.mode in ["zero_shot", "both"]:
        print("\n" + "="*60)
        print("ZERO-SHOT EVALUATION")
        print("="*60)
        results_zero = []
        if args.task in ["6bin", "both"]:
            results_zero_6 = evaluator.evaluate(use_few_shot=False, task="6bin")
            evaluator.save_results(results_zero_6, use_few_shot=False, task="6bin")
            results_zero.extend(results_zero_6)
        if args.task in ["3bin", "both"]:
            results_zero_3 = evaluator.evaluate(use_few_shot=False, task="3bin")
            evaluator.save_results(results_zero_3, use_few_shot=False, task="3bin")
            results_zero.extend(results_zero_3)
    
    if args.mode in ["few_shot", "both"]:
        print("\n" + "="*60)
        print("FEW-SHOT EVALUATION")
        print("="*60)
        results_few = []
        if args.task in ["6bin", "both"]:
            results_few_6 = evaluator.evaluate(use_few_shot=True, task="6bin")
            evaluator.save_results(results_few_6, use_few_shot=True, task="6bin")
            results_few.extend(results_few_6)
        if args.task in ["3bin", "both"]:
            results_few_3 = evaluator.evaluate(use_few_shot=True, task="3bin")
            evaluator.save_results(results_few_3, use_few_shot=True, task="3bin")
            results_few.extend(results_few_3)
    
    # Generate comparison plot if both modes were run
    if args.mode == "both" and results_zero and results_few:
        print("\n" + "="*60)
        print("GENERATING COMPARISON PLOTS")
        print("="*60)
        evaluator.plot_comparison(results_zero, results_few)


if __name__ == "__main__":
    main()
            