#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Fast Checkpoint Evaluation Script using Deployed Models (vLLM/API)
# Evaluates multiple checkpoints on HotpotQA using already-deployed models

import os
import json
import argparse
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from taskutils.memory_eval.utils import (
    extract_answer,
    extract_boxed_answer,
    exact_match_score,
    sub_exact_match_score,
    f1_score,
    normalize_answer
)


@dataclass
class CheckpointInfo:
    """Information about a checkpoint"""
    exp_name: str
    checkpoint_name: str
    checkpoint_path: str
    global_step: int
    api_url: Optional[str] = None  # If using deployed model


@dataclass
class EvalMetrics:
    """Evaluation metrics for a checkpoint"""
    checkpoint_info: Dict[str, Any]
    total_samples: int
    exact_match: float
    sub_exact_match: float
    f1_score: float
    precision: float
    recall: float
    correct_count: int
    eval_time_seconds: float
    samples_per_second: float
    error_count: int = 0


@dataclass
class SampleResult:
    """Result for a single sample"""
    sample_id: str
    question: str
    ground_truth: str
    prediction: str
    extracted_answer: Optional[str]
    exact_match: bool
    sub_exact_match: bool
    f1: float
    precision: float
    recall: float


class DeployedCheckpointEvaluator:
    """Evaluates deployed checkpoints via API"""

    def __init__(
        self,
        outputs_dir: str = "/home/admin123/dl/MemAgent/outputs",
        test_data_path: str = None,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_samples: Optional[int] = None,
        save_predictions: bool = True,
        concurrency: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 128
    ):
        self.outputs_dir = Path(outputs_dir)
        self.test_data_path = test_data_path or str(
            Path(__file__).parent / "taskutils/memory_data/hotpotqa/hotpotqa_dev_20.parquet"
        )
        self.api_base = api_base
        self.api_key = api_key
        self.max_samples = max_samples
        self.save_predictions = save_predictions
        self.concurrency = concurrency
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def discover_checkpoints(self, exp_filter: Optional[str] = None) -> List[CheckpointInfo]:
        """Discover all checkpoints in outputs directory"""
        checkpoints = []

        # Search in memory_agent subdirectory
        memory_agent_dir = self.outputs_dir / "memory_agent"
        if memory_agent_dir.exists():
            for exp_dir in memory_agent_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                exp_name = exp_dir.name

                # Apply filter if specified
                if exp_filter and exp_filter not in exp_name:
                    continue

                # Find all global_step_* directories
                for ckpt_dir in exp_dir.iterdir():
                    if not ckpt_dir.is_dir() or not ckpt_dir.name.startswith("global_step_"):
                        continue

                    try:
                        global_step = int(ckpt_dir.name.split("_")[-1])
                    except ValueError:
                        continue

                    checkpoints.append(CheckpointInfo(
                        exp_name=exp_name,
                        checkpoint_name=ckpt_dir.name,
                        checkpoint_path=str(ckpt_dir),
                        global_step=global_step
                    ))

        # Sort by exp_name and global_step
        checkpoints.sort(key=lambda x: (x.exp_name, x.global_step))

        return checkpoints

    def load_test_data(self) -> pd.DataFrame:
        """Load test dataset and normalize field names"""
        print(f"Loading test data from: {self.test_data_path}")
        df = pd.read_parquet(self.test_data_path)

        # Normalize field names based on actual data format
        if 'prompt' in df.columns and isinstance(df['prompt'].iloc[0], list):
            # Standard format: extract from prompt and reward_model
            print("Detected standard format (prompt/reward_model)")
            df['input'] = df['prompt'].apply(lambda x: x[0]['content'] if isinstance(x, list) and len(x) > 0 else '')
            df['output'] = df['reward_model'].apply(
                lambda x: x['ground_truth'][0] if isinstance(x.get('ground_truth'), list) and len(x.get('ground_truth', [])) > 0
                else str(x.get('ground_truth', ''))
            )
        elif 'question' in df.columns and 'solution' in df.columns:
            # Simplified format: use question and solution
            print("Detected simplified format (question/solution)")
            df['input'] = df['question']
            df['output'] = df['solution']
        else:
            raise ValueError(f"Unknown data format. Columns: {df.columns.tolist()}")

        if self.max_samples:
            df = df.head(self.max_samples)

        print(f"Loaded {len(df)} samples")
        return df

    async def generate_prediction_async(
        self,
        session: aiohttp.ClientSession,
        context: str,
        question: str,
        model_name: str
    ) -> str:
        """Generate prediction for a single sample via API"""
        # Construct prompt (following the training format)
        # Limit context to 10000 characters
        prompt = f"""Given the following context, answer the question.
Format your answer as: \\boxed{{your answer}}

Context: {context[:10000]}

Question: {question}

Answer:"""

        try:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens
                },
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"API error {resp.status}: {error_text}")

                result = await resp.json()
                response = result['choices'][0]['message']['content']
                return response

        except Exception as e:
            print(f"Error generating prediction: {e}")
            return ""

    async def evaluate_checkpoint_async(
        self,
        checkpoint_info: CheckpointInfo,
        model_name: str
    ) -> tuple[EvalMetrics, List[SampleResult]]:
        """Evaluate a single checkpoint asynchronously"""
        print(f"\n{'='*80}")
        print(f"Evaluating: {checkpoint_info.exp_name} / {checkpoint_info.checkpoint_name}")
        print(f"Model: {model_name}")
        print(f"API: {self.api_base}")
        print(f"{'='*80}\n")

        start_time = time.time()

        # Load test data
        test_df = self.load_test_data()

        # Evaluate samples
        sample_results = []
        metrics = {
            'em': 0,
            'sub_em': 0,
            'f1': 0,
            'prec': 0,
            'recall': 0,
            'total': 0,
            'errors': 0
        }

        # Create aiohttp session
        timeout = aiohttp.ClientTimeout(total=3600)
        connector = aiohttp.TCPConnector(limit=self.concurrency)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Create tasks
            tasks = []
            for idx, row in test_df.iterrows():
                task = self._evaluate_sample(session, idx, row, model_name)
                tasks.append(task)

            # Run tasks with progress bar and collect results
            for coro in tqdm_asyncio(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Evaluating samples"
            ):
                sample_metrics, sample_result = await coro

                # Accumulate metrics
                metrics['em'] += sample_metrics['em']
                metrics['sub_em'] += sample_metrics['sub_em']
                metrics['f1'] += sample_metrics['f1']
                metrics['prec'] += sample_metrics['prec']
                metrics['recall'] += sample_metrics['recall']
                metrics['total'] += sample_metrics['total']
                metrics['errors'] += sample_metrics['errors']

                # Collect sample results
                if sample_result is not None:
                    sample_results.append(sample_result)

        # Compute average metrics
        total = metrics['total']
        eval_time = time.time() - start_time

        result = EvalMetrics(
            checkpoint_info=asdict(checkpoint_info),
            total_samples=total,
            exact_match=metrics['em'] / total if total > 0 else 0,
            sub_exact_match=metrics['sub_em'] / total if total > 0 else 0,
            f1_score=metrics['f1'] / total if total > 0 else 0,
            precision=metrics['prec'] / total if total > 0 else 0,
            recall=metrics['recall'] / total if total > 0 else 0,
            correct_count=int(metrics['em']),
            eval_time_seconds=eval_time,
            samples_per_second=total / eval_time if eval_time > 0 else 0,
            error_count=metrics['errors']
        )

        # Print summary
        print(f"\n{'='*80}")
        print(f"Results for {checkpoint_info.exp_name} / {checkpoint_info.checkpoint_name}:")
        print(f"  Exact Match:     {result.exact_match:.4f} ({result.correct_count}/{total})")
        print(f"  Sub-EM:          {result.sub_exact_match:.4f}")
        print(f"  F1 Score:        {result.f1_score:.4f}")
        print(f"  Precision:       {result.precision:.4f}")
        print(f"  Recall:          {result.recall:.4f}")
        print(f"  Eval Time:       {result.eval_time_seconds:.2f}s")
        print(f"  Speed:           {result.samples_per_second:.2f} samples/s")
        print(f"  Errors:          {result.error_count}")
        print(f"{'='*80}\n")

        return result, sample_results

    async def _evaluate_sample(
        self,
        session: aiohttp.ClientSession,
        idx: int,
        row: pd.Series,
        model_name: str
    ) -> tuple[dict, Optional[SampleResult]]:
        """Evaluate a single sample and return metrics and result"""
        try:
            # Generate prediction
            prediction = await self.generate_prediction_async(
                session, row['context'], row['input'], model_name
            )

            # Extract answer
            extracted = extract_boxed_answer(prediction)
            if extracted is None:
                extracted = extract_answer(prediction)

            ground_truth = row['output']

            # Compute metrics
            if extracted:
                em = exact_match_score(extracted, ground_truth)
                sub_em = sub_exact_match_score(extracted, ground_truth)
                f1, prec, rec = f1_score(extracted, ground_truth)
            else:
                em = sub_em = f1 = prec = rec = 0

            # Prepare metrics dict
            sample_metrics = {
                'em': em,
                'sub_em': sub_em,
                'f1': f1,
                'prec': prec,
                'recall': rec,
                'total': 1,
                'errors': 0
            }

            # Prepare sample result
            sample_result = None
            if self.save_predictions:
                sample_result = SampleResult(
                    sample_id=str(idx),
                    question=row['input'],
                    ground_truth=ground_truth,
                    prediction=prediction,
                    extracted_answer=extracted,
                    exact_match=bool(em),
                    sub_exact_match=bool(sub_em),
                    f1=f1,
                    precision=prec,
                    recall=rec
                )

            return sample_metrics, sample_result

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return {'em': 0, 'sub_em': 0, 'f1': 0, 'prec': 0, 'recall': 0, 'total': 0, 'errors': 1}, None

    def save_results(
        self,
        results: List[EvalMetrics],
        sample_results: Dict[str, List[SampleResult]],
        output_file: str
    ):
        """Save evaluation results to JSON"""
        output = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'test_data_path': self.test_data_path,
                'api_base': self.api_base,
                'max_samples': self.max_samples,
                'concurrency': self.concurrency,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'max_tokens': self.max_tokens
            },
            'summary': [],
            'detailed_results': {}
        }

        # Add summary
        for result in results:
            output['summary'].append(asdict(result))

        # Add detailed sample results if available
        if self.save_predictions:
            for ckpt_key, samples in sample_results.items():
                output['detailed_results'][ckpt_key] = [
                    asdict(sample) for sample in samples
                ]

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")

    def generate_comparison_table(self, results: List[EvalMetrics]) -> str:
        """Generate markdown comparison table"""
        lines = [
            "# Checkpoint Comparison Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Summary Table\n",
            "| Experiment | Checkpoint | Step | EM | Sub-EM | F1 | Precision | Recall | Speed (samples/s) | Errors |",
            "|------------|-----------|------|----|----|----|-----------| -------|-------------------|--------|"
        ]

        for result in results:
            ckpt = result.checkpoint_info
            lines.append(
                f"| {ckpt['exp_name']} | {ckpt['checkpoint_name']} | "
                f"{ckpt['global_step']} | "
                f"{result.exact_match:.4f} | "
                f"{result.sub_exact_match:.4f} | "
                f"{result.f1_score:.4f} | "
                f"{result.precision:.4f} | "
                f"{result.recall:.4f} | "
                f"{result.samples_per_second:.2f} | "
                f"{result.error_count} |"
            )

        # Add best performers
        if results:
            lines.extend([
                "\n## Best Performers\n",
                f"**Best EM:** {max(results, key=lambda x: x.exact_match).checkpoint_info['exp_name']} / "
                f"{max(results, key=lambda x: x.exact_match).checkpoint_info['checkpoint_name']} "
                f"({max(results, key=lambda x: x.exact_match).exact_match:.4f})",
                "",
                f"**Best F1:** {max(results, key=lambda x: x.f1_score).checkpoint_info['exp_name']} / "
                f"{max(results, key=lambda x: x.f1_score).checkpoint_info['checkpoint_name']} "
                f"({max(results, key=lambda x: x.f1_score).f1_score:.4f})",
                "",
                f"**Fastest:** {max(results, key=lambda x: x.samples_per_second).checkpoint_info['exp_name']} / "
                f"{max(results, key=lambda x: x.samples_per_second).checkpoint_info['checkpoint_name']} "
                f"({max(results, key=lambda x: x.samples_per_second).samples_per_second:.2f} samples/s)",
            ])

        return "\n".join(lines)


async def main_async(args):
    """Async main function"""
    # Initialize evaluator
    evaluator = DeployedCheckpointEvaluator(
        outputs_dir=args.outputs_dir,
        test_data_path=args.test_data,
        api_base=args.api_base,
        api_key=args.api_key,
        max_samples=args.max_samples,
        save_predictions=args.save_predictions,
        concurrency=args.concurrency,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # If checkpoint paths are provided directly, use them
    if args.checkpoint_paths:
        checkpoints = []
        for path in args.checkpoint_paths:
            path_obj = Path(path)
            ckpt_name = path_obj.name
            exp_name = path_obj.parent.name if path_obj.parent.name != "memory_agent" else "custom"

            try:
                global_step = int(ckpt_name.split("_")[-1]) if "global_step_" in ckpt_name else 0
            except ValueError:
                global_step = 0

            checkpoints.append(CheckpointInfo(
                exp_name=exp_name,
                checkpoint_name=ckpt_name,
                checkpoint_path=path,
                global_step=global_step
            ))
    else:
        # Discover checkpoints
        checkpoints = evaluator.discover_checkpoints(exp_filter=args.exp_filter)

        # Apply checkpoint filter
        if args.checkpoint_filter:
            checkpoints = [
                ckpt for ckpt in checkpoints
                if args.checkpoint_filter in ckpt.checkpoint_name
            ]

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"\nFound {len(checkpoints)} checkpoints to evaluate:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.exp_name} / {ckpt.checkpoint_name} (step {ckpt.global_step})")

    # Evaluate all checkpoints
    results = []
    sample_results_dict = {}

    for ckpt in checkpoints:
        try:
            # Use checkpoint path as model name (vLLM should be serving it)
            model_name = args.model_name or ckpt.checkpoint_path

            result, samples = await evaluator.evaluate_checkpoint_async(ckpt, model_name)
            results.append(result)

            ckpt_key = f"{ckpt.exp_name}_{ckpt.checkpoint_name}"
            sample_results_dict[ckpt_key] = samples
        except Exception as e:
            print(f"ERROR evaluating {ckpt.exp_name}/{ckpt.checkpoint_name}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("No successful evaluations!")
        return

    # Generate output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = args.output_json or f"eval_results_{timestamp}.json"
    output_report = args.output_report or f"eval_report_{timestamp}.md"

    # Save results
    evaluator.save_results(results, sample_results_dict, output_json)

    # Generate and save comparison report
    report = evaluator.generate_comparison_table(results)
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Comparison report saved to: {output_report}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Evaluated {len(results)} checkpoints")
    print(f"Results: {output_json}")
    print(f"Report: {output_report}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate deployed checkpoints on HotpotQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all checkpoints in outputs directory (requires vLLM serving)
  python batch_evaluate_deployed.py --api_base http://localhost:8000/v1

  # Evaluate specific experiment with filter
  python batch_evaluate_deployed.py --exp_filter 0.5B_kv --max_samples 100

  # Evaluate specific checkpoints
  python batch_evaluate_deployed.py --checkpoint_paths \\
      /path/to/outputs/memory_agent/0.5B_kv/global_step_1000 \\
      /path/to/outputs/memory_agent/0.5B_kv/global_step_2000

  # Use higher concurrency for faster evaluation
  python batch_evaluate_deployed.py --concurrency 50 --max_samples 200
        """
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="/home/admin123/dl/MemAgent/outputs",
        help="Directory containing checkpoint outputs"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test dataset (parquet file)"
    )
    parser.add_argument(
        "--exp_filter",
        type=str,
        default=None,
        help="Filter experiments by name (e.g., '0.5B', '7B_kv')"
    )
    parser.add_argument(
        "--checkpoint_filter",
        type=str,
        default=None,
        help="Filter checkpoints by pattern (e.g., 'global_step_1000')"
    )
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of checkpoint paths to evaluate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to evaluate"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="API base URL (vLLM or OpenAI-compatible endpoint)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for authentication"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name to use (if different from checkpoint path)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent API requests"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated)"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Output markdown report path (default: auto-generated)"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Save detailed predictions for each sample"
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
