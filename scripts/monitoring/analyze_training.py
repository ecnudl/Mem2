#!/usr/bin/env python3
"""
Post-training analysis script for MemAgent.
Generates comprehensive training reports with detailed metrics and visualizations.

Usage:
    # Analyze from wandb
    python analyze_training.py --source wandb --project verl-memagent --run lora_4gpu_balanced_20k_n8

    # Analyze from log file
    python analyze_training.py --source file --log-file /path/to/training.log

    # Analyze from experiment directory (auto-detect)
    python analyze_training.py --source auto --exp-dir /home/admin123/dl/MemAgent/outputs/lora_4gpu_balanced_20k_n8
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import sys

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy import signal


class TrainingAnalyzer:
    """Analyze completed training runs."""

    def __init__(self, output_dir: str = "./analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []

    def parse_console_line(self, line: str) -> Optional[Dict]:
        """Parse a console log line to extract metrics."""
        if "step:" not in line:
            return None

        metrics = {}

        # Extract step
        step_match = re.search(r'step:(\d+)', line)
        if not step_match:
            return None
        metrics['step'] = int(step_match.group(1))

        # Extract all numerical metrics
        metric_pattern = r'([a-zA-Z_/]+):([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
        for match in re.finditer(metric_pattern, line):
            key, value = match.groups()
            if key != 'step':
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue

        return metrics

    def load_from_file(self, log_file: Path):
        """Load metrics from log file."""
        print(f"Loading metrics from: {log_file}")

        if not log_file.exists():
            print(f"Error: Log file not found: {log_file}")
            sys.exit(1)

        with open(log_file, 'r') as f:
            for line in f:
                metrics = self.parse_console_line(line.strip())
                if metrics:
                    self.metrics_history.append(metrics)

        print(f"Loaded {len(self.metrics_history)} metric records")

    def load_from_wandb(self, project: str, run_name: str):
        """Load metrics from wandb."""
        try:
            import wandb
            api = wandb.Api()

            print(f"Loading from wandb - project: {project}, run: {run_name}")

            runs = api.runs(project, filters={"display_name": run_name})
            if not runs:
                print(f"No run found with name: {run_name}")
                sys.exit(1)

            run = runs[0]
            print(f"Found run: {run.name} (id: {run.id})")

            history = run.history()

            for _, row in history.iterrows():
                metrics = {'step': int(row.get('_step', 0))}
                for col in history.columns:
                    if col not in ['_step', '_timestamp', '_runtime']:
                        try:
                            metrics[col] = float(row[col])
                        except (ValueError, TypeError):
                            continue

                self.metrics_history.append(metrics)

            print(f"Loaded {len(self.metrics_history)} metric records")

        except ImportError:
            print("wandb is not installed. Install with: pip install wandb")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading from wandb: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def get_metric_series(self, key: str) -> tuple:
        """Extract a time series for a specific metric."""
        steps = []
        values = []

        for record in self.metrics_history:
            if key in record:
                steps.append(record['step'])
                values.append(record[key])

        return np.array(steps), np.array(values)

    def smooth_curve(self, values: np.ndarray, window_size: int = 10) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(values) < window_size:
            return values

        return signal.savgol_filter(values, min(window_size, len(values) // 2 * 2 + 1), 3)

    def analyze_trend(self, values: np.ndarray) -> Dict:
        """Analyze the trend of a metric."""
        if len(values) < 2:
            return {
                'trend': 'insufficient_data',
                'slope': 0,
                'improvement': 0
            }

        # Linear fit
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]

        # Calculate improvement
        start_mean = np.mean(values[:max(1, len(values)//10)])
        end_mean = np.mean(values[-max(1, len(values)//10):])
        improvement = end_mean - start_mean

        if abs(slope) < 1e-6:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'

        return {
            'trend': trend,
            'slope': slope,
            'improvement': improvement,
            'start_value': start_mean,
            'end_value': end_mean
        }

    def generate_report(self):
        """Generate comprehensive analysis report."""
        if not self.metrics_history:
            print("No metrics to analyze")
            return

        print("\n" + "="*80)
        print("TRAINING ANALYSIS REPORT")
        print("="*80)

        # Identify available metrics
        all_keys = set()
        for record in self.metrics_history:
            all_keys.update(record.keys())
        all_keys.discard('step')

        # Separate train and val metrics
        train_keys = [k for k in all_keys if not k.startswith('val/')]
        val_keys = [k for k in all_keys if k.startswith('val/')]

        report = {
            'total_steps': len(self.metrics_history),
            'metrics': {}
        }

        print(f"\nTotal training steps: {len(self.metrics_history)}")
        print(f"\nAvailable training metrics: {', '.join(train_keys)}")
        print(f"Available validation metrics: {', '.join(val_keys)}")

        # Analyze key metrics
        key_metrics = ['actor_loss', 'critic_loss', 'reward_mean', 'entropy', 'kl']

        print("\n" + "-"*80)
        print("KEY METRICS ANALYSIS")
        print("-"*80)

        for metric in key_metrics:
            # Try both with and without train/ prefix
            steps, values = self.get_metric_series(metric)
            if len(values) == 0:
                steps, values = self.get_metric_series(f'train/{metric}')

            if len(values) == 0:
                continue

            analysis = self.analyze_trend(values)
            report['metrics'][metric] = analysis

            print(f"\n{metric.upper()}:")
            print(f"  Trend: {analysis['trend']}")
            print(f"  Start value: {analysis['start_value']:.4f}")
            print(f"  End value: {analysis['end_value']:.4f}")
            print(f"  Total change: {analysis['improvement']:.4f}")
            print(f"  Slope: {analysis['slope']:.6f}")

        # Analyze validation metrics
        if val_keys:
            print("\n" + "-"*80)
            print("VALIDATION METRICS")
            print("-"*80)

            for metric in val_keys:
                steps, values = self.get_metric_series(metric)
                if len(values) == 0:
                    continue

                analysis = self.analyze_trend(values)
                report['metrics'][metric] = analysis

                print(f"\n{metric}:")
                print(f"  Trend: {analysis['trend']}")
                print(f"  Best value: {np.max(values):.4f}")
                print(f"  Final value: {values[-1]:.4f}")

        # Training effectiveness assessment
        print("\n" + "-"*80)
        print("TRAINING EFFECTIVENESS ASSESSMENT")
        print("-"*80)

        effectiveness_score = 0
        checks = []

        # Check 1: Loss decreasing
        if 'actor_loss' in report['metrics']:
            if report['metrics']['actor_loss']['trend'] == 'decreasing':
                effectiveness_score += 1
                checks.append("✓ Actor loss is decreasing")
            else:
                checks.append("✗ Actor loss is not decreasing")

        # Check 2: Reward increasing
        if 'reward_mean' in report['metrics']:
            if report['metrics']['reward_mean']['trend'] == 'increasing':
                effectiveness_score += 1
                checks.append("✓ Reward is increasing")
            else:
                checks.append("✗ Reward is not increasing")

        # Check 3: Critic loss stable or decreasing
        if 'critic_loss' in report['metrics']:
            trend = report['metrics']['critic_loss']['trend']
            if trend in ['decreasing', 'stable']:
                effectiveness_score += 1
                checks.append("✓ Critic loss is stable/decreasing")
            else:
                checks.append("✗ Critic loss is increasing")

        # Check 4: No entropy collapse
        if 'entropy' in report['metrics']:
            _, entropy_values = self.get_metric_series('entropy')
            if len(entropy_values) > 0 and np.mean(entropy_values[-10:]) > 0.01:
                effectiveness_score += 1
                checks.append("✓ No entropy collapse")
            else:
                checks.append("✗ Entropy collapsed")

        for check in checks:
            print(f"  {check}")

        print(f"\nOverall effectiveness score: {effectiveness_score}/{len(checks)}")

        if effectiveness_score >= len(checks) * 0.75:
            print("✓ Training appears to be EFFECTIVE")
        elif effectiveness_score >= len(checks) * 0.5:
            print("⚠ Training shows MIXED results")
        else:
            print("✗ Training may be INEFFECTIVE")

        # Save report to JSON
        report_file = self.output_dir / 'training_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")

        print("\n" + "="*80 + "\n")

    def plot_comprehensive_analysis(self):
        """Generate comprehensive plots."""
        if not self.metrics_history:
            print("No metrics to plot")
            return

        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Actor Loss with smoothing
        ax1 = fig.add_subplot(gs[0, 0])
        steps, values = self.get_metric_series('actor_loss')
        if len(values) == 0:
            steps, values = self.get_metric_series('train/actor_loss')
        if len(values) > 0:
            ax1.plot(steps, values, alpha=0.3, color='blue', label='Raw')
            if len(values) > 10:
                smoothed = self.smooth_curve(values)
                ax1.plot(steps, smoothed, linewidth=2, color='blue', label='Smoothed')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Actor Loss')
            ax1.set_title('Actor Loss over Training')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Critic Loss with smoothing
        ax2 = fig.add_subplot(gs[0, 1])
        steps, values = self.get_metric_series('critic_loss')
        if len(values) == 0:
            steps, values = self.get_metric_series('train/critic_loss')
        if len(values) > 0:
            ax2.plot(steps, values, alpha=0.3, color='red', label='Raw')
            if len(values) > 10:
                smoothed = self.smooth_curve(values)
                ax2.plot(steps, smoothed, linewidth=2, color='red', label='Smoothed')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Critic Loss')
            ax2.set_title('Critic Loss over Training')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Reward Mean with smoothing
        ax3 = fig.add_subplot(gs[0, 2])
        steps, values = self.get_metric_series('reward_mean')
        if len(values) == 0:
            steps, values = self.get_metric_series('train/reward_mean')
        if len(values) > 0:
            ax3.plot(steps, values, alpha=0.3, color='green', label='Raw')
            if len(values) > 10:
                smoothed = self.smooth_curve(values)
                ax3.plot(steps, smoothed, linewidth=2, color='green', label='Smoothed')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Reward Mean')
            ax3.set_title('Mean Reward over Training')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Reward distribution over time
        ax4 = fig.add_subplot(gs[1, 0])
        steps_mean, reward_mean = self.get_metric_series('reward_mean')
        steps_std, reward_std = self.get_metric_series('reward_std')
        if len(reward_mean) > 0:
            if len(reward_std) > 0 and len(steps_mean) == len(steps_std):
                ax4.plot(steps_mean, reward_mean, color='green', linewidth=2, label='Mean')
                ax4.fill_between(steps_mean,
                                reward_mean - reward_std,
                                reward_mean + reward_std,
                                alpha=0.3, color='green', label='±1 std')
            else:
                ax4.plot(steps_mean, reward_mean, color='green', linewidth=2)
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Reward')
            ax4.set_title('Reward Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Plot 5: KL Divergence
        ax5 = fig.add_subplot(gs[1, 1])
        steps, values = self.get_metric_series('kl')
        if len(values) == 0:
            steps, values = self.get_metric_series('train/kl')
        if len(values) > 0:
            ax5.plot(steps, values, color='purple', linewidth=2)
            ax5.set_xlabel('Step')
            ax5.set_ylabel('KL Divergence')
            ax5.set_title('KL Divergence over Training')
            ax5.grid(True, alpha=0.3)

        # Plot 6: Entropy
        ax6 = fig.add_subplot(gs[1, 2])
        steps, values = self.get_metric_series('entropy')
        if len(values) == 0:
            steps, values = self.get_metric_series('train/entropy')
        if len(values) > 0:
            ax6.plot(steps, values, color='brown', linewidth=2)
            ax6.set_xlabel('Step')
            ax6.set_ylabel('Entropy')
            ax6.set_title('Policy Entropy over Training')
            ax6.grid(True, alpha=0.3)

        # Plot 7: Validation metrics
        ax7 = fig.add_subplot(gs[2, 0])
        steps, values = self.get_metric_series('val/reward_mean')
        if len(values) > 0:
            ax7.plot(steps, values, color='orange', linewidth=2, marker='o', markersize=5)
            ax7.set_xlabel('Step')
            ax7.set_ylabel('Val Reward Mean')
            ax7.set_title('Validation Reward')
            ax7.grid(True, alpha=0.3)

        # Plot 8: Validation accuracy
        ax8 = fig.add_subplot(gs[2, 1])
        steps, values = self.get_metric_series('val/accuracy')
        if len(values) > 0:
            ax8.plot(steps, values, color='teal', linewidth=2, marker='s', markersize=5)
            ax8.set_xlabel('Step')
            ax8.set_ylabel('Val Accuracy')
            ax8.set_title('Validation Accuracy')
            ax8.set_ylim(0, 1.0)
            ax8.grid(True, alpha=0.3)

        # Plot 9: Loss vs Reward correlation
        ax9 = fig.add_subplot(gs[2, 2])
        _, loss_values = self.get_metric_series('actor_loss')
        _, reward_values = self.get_metric_series('reward_mean')
        if len(loss_values) > 0 and len(reward_values) > 0:
            min_len = min(len(loss_values), len(reward_values))
            ax9.scatter(loss_values[:min_len], reward_values[:min_len],
                       alpha=0.5, c=range(min_len), cmap='viridis')
            ax9.set_xlabel('Actor Loss')
            ax9.set_ylabel('Reward Mean')
            ax9.set_title('Loss vs Reward Correlation')
            ax9.grid(True, alpha=0.3)

        fig.suptitle('Comprehensive Training Analysis', fontsize=20, fontweight='bold')

        # Save plot
        output_file = self.output_dir / 'comprehensive_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Comprehensive analysis plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze completed MemAgent training")
    parser.add_argument('--source', type=str, default='auto',
                       choices=['file', 'wandb', 'auto'],
                       help='Data source for analysis')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Path to log file (for file source)')
    parser.add_argument('--project', type=str, default='verl-memagent',
                       help='Wandb project name (for wandb source)')
    parser.add_argument('--run', type=str, default=None,
                       help='Wandb run name (for wandb source)')
    parser.add_argument('--exp-dir', type=str, default=None,
                       help='Experiment directory (for auto source)')
    parser.add_argument('--output-dir', type=str, default='./analysis_results',
                       help='Directory to save analysis results')

    args = parser.parse_args()

    analyzer = TrainingAnalyzer(output_dir=args.output_dir)

    if args.source == 'wandb':
        if not args.run:
            print("Error: --run is required for wandb source")
            sys.exit(1)
        analyzer.load_from_wandb(args.project, args.run)

    elif args.source == 'file':
        if not args.log_file:
            print("Error: --log-file is required for file source")
            sys.exit(1)
        log_file = Path(args.log_file)
        analyzer.load_from_file(log_file)

    elif args.source == 'auto':
        if not args.exp_dir:
            print("Error: --exp-dir is required for auto source")
            sys.exit(1)

        exp_dir = Path(args.exp_dir)
        if not exp_dir.exists():
            print(f"Experiment directory does not exist: {exp_dir}")
            sys.exit(1)

        # Try to find log file
        log_files = list(exp_dir.glob("*.log"))
        if not log_files:
            log_files = list(exp_dir.glob("**/*.log"))

        if log_files:
            log_file = log_files[0]
            print(f"Found log file: {log_file}")
            analyzer.load_from_file(log_file)
        else:
            print(f"No log files found in {exp_dir}")
            sys.exit(1)

    # Generate analysis
    analyzer.generate_report()
    analyzer.plot_comprehensive_analysis()

    print(f"\nAnalysis complete! Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
