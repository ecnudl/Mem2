#!/usr/bin/env python3
"""
Real-time training monitoring script for MemAgent.
Monitors loss and reward metrics during training and plots curves.

Usage:
    # Monitor wandb (recommended)
    python monitor_training.py --mode wandb --project verl-memagent --run lora_4gpu_balanced_20k_n8

    # Monitor from log file
    python monitor_training.py --mode file --log-file /path/to/training.log

    # Auto-detect from experiment directory
    python monitor_training.py --mode auto --exp-dir /home/admin123/dl/MemAgent/outputs/lora_4gpu_balanced_20k_n8
"""

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import numpy as np


class TrainingMonitor:
    """Monitor training metrics and plot curves."""

    def __init__(self, save_dir: str = "./monitoring_plots", update_interval: int = 10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.update_interval = update_interval

        # Storage for metrics
        self.steps = []
        self.train_metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'reward_mean': [],
            'reward_std': [],
            'entropy': [],
            'kl': [],
            'advantages': [],
            'clipfrac': [],
        }
        self.val_metrics = {
            'reward_mean': [],
            'accuracy': [],
        }
        self.val_steps = []

        self.last_update_time = 0

    def parse_console_line(self, line: str) -> Optional[Dict]:
        """Parse a console log line to extract metrics."""
        # Example format: "step:123 - actor/pg_loss:0.456 - critic/score/mean:0.312"
        if "step:" not in line:
            return None

        metrics = {}

        # Extract step
        step_match = re.search(r'step:(\d+)', line)
        if not step_match:
            return None
        metrics['step'] = int(step_match.group(1))

        # Extract all numerical metrics (handle paths with /)
        metric_pattern = r'([a-zA-Z_/]+):([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
        for match in re.finditer(metric_pattern, line):
            key, value = match.groups()
            if key != 'step':
                try:
                    metrics[key] = float(value)
                    # Also create simplified aliases for common metrics
                    if key == 'actor/pg_loss':
                        metrics['actor_loss'] = float(value)
                    elif key == 'critic/returns/mean':
                        # Use negative returns as proxy for critic loss
                        # (lower returns = worse predictions = higher loss)
                        metrics['critic_loss'] = -float(value)
                    elif key == 'critic/rewards/mean' or key == 'critic/score/mean':
                        # Both are reward metrics (score and reward are the same)
                        metrics['reward_mean'] = float(value)
                    elif key == 'critic/rewards/max':
                        metrics['reward_max'] = float(value)
                    elif key == 'critic/rewards/min':
                        metrics['reward_min'] = float(value)
                    elif key == 'actor/entropy_loss':
                        metrics['entropy'] = float(value)
                    elif key == 'actor/ppo_kl':
                        metrics['kl'] = float(value)
                except ValueError:
                    continue

        return metrics

    def add_metrics(self, metrics: Dict):
        """Add metrics to history."""
        if 'step' not in metrics:
            return

        step = metrics['step']

        # Determine if this is validation or training
        is_val = any(k.startswith('val/') for k in metrics.keys())

        if is_val:
            if step not in self.val_steps:
                self.val_steps.append(step)
            for key in self.val_metrics:
                val_key = f'val/{key}' if not key.startswith('val/') else key
                if val_key in metrics:
                    self.val_metrics[key].append(metrics[val_key])
        else:
            if step not in self.steps:
                self.steps.append(step)
            for key in self.train_metrics:
                train_key = f'train/{key}' if not key.startswith('train/') else key
                # Also check without prefix
                if train_key in metrics:
                    self.train_metrics[key].append(metrics[train_key])
                elif key in metrics:
                    self.train_metrics[key].append(metrics[key])

    def should_update_plot(self) -> bool:
        """Check if enough time has passed to update the plot."""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

    def plot_metrics(self, force: bool = False):
        """Plot all metrics."""
        if not force and not self.should_update_plot():
            return

        if len(self.steps) == 0:
            print("No training data to plot yet...")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MemAgent Training Monitoring', fontsize=16, fontweight='bold')

        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        if self.train_metrics['actor_loss']:
            ax1.plot(self.steps[:len(self.train_metrics['actor_loss'])],
                    self.train_metrics['actor_loss'],
                    label='Actor Loss', color='blue', linewidth=2)
        if self.train_metrics['critic_loss']:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(self.steps[:len(self.train_metrics['critic_loss'])],
                         self.train_metrics['critic_loss'],
                         label='Critic Loss', color='red', linewidth=2, linestyle='--')
            ax1_twin.set_ylabel('Critic Loss', color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
            ax1_twin.legend(loc='upper right')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Actor Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title('Training Loss')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Reward curves
        ax2 = axes[0, 1]
        if self.train_metrics['reward_mean']:
            steps_reward = self.steps[:len(self.train_metrics['reward_mean'])]
            rewards = self.train_metrics['reward_mean']
            ax2.plot(steps_reward, rewards, label='Train Reward (mean)',
                    color='green', linewidth=2)

            # Add standard deviation as shaded area if available
            if self.train_metrics['reward_std']:
                reward_std = self.train_metrics['reward_std'][:len(rewards)]
                ax2.fill_between(steps_reward,
                               np.array(rewards) - np.array(reward_std),
                               np.array(rewards) + np.array(reward_std),
                               alpha=0.2, color='green')

        if self.val_metrics['reward_mean'] and self.val_steps:
            ax2.plot(self.val_steps[:len(self.val_metrics['reward_mean'])],
                    self.val_metrics['reward_mean'],
                    label='Val Reward (mean)', color='orange',
                    linewidth=2, marker='o', markersize=4)

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.set_title('Reward Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: KL divergence and entropy
        ax3 = axes[1, 0]
        if self.train_metrics['kl']:
            ax3.plot(self.steps[:len(self.train_metrics['kl'])],
                    self.train_metrics['kl'],
                    label='KL Divergence', color='purple', linewidth=2)
        if self.train_metrics['entropy']:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(self.steps[:len(self.train_metrics['entropy'])],
                         self.train_metrics['entropy'],
                         label='Entropy', color='brown', linewidth=2, linestyle='--')
            ax3_twin.set_ylabel('Entropy', color='brown')
            ax3_twin.tick_params(axis='y', labelcolor='brown')
            ax3_twin.legend(loc='upper right')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('KL Divergence', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        ax3.set_title('KL Divergence & Entropy')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Validation accuracy
        ax4 = axes[1, 1]
        if self.val_metrics['accuracy'] and self.val_steps:
            ax4.plot(self.val_steps[:len(self.val_metrics['accuracy'])],
                    self.val_metrics['accuracy'],
                    label='Val Accuracy', color='teal',
                    linewidth=2, marker='s', markersize=5)
            ax4.set_ylim(0, 1.0)
        elif self.val_metrics['reward_mean'] and self.val_steps:
            # If no accuracy, show val reward again
            ax4.plot(self.val_steps[:len(self.val_metrics['reward_mean'])],
                    self.val_metrics['reward_mean'],
                    label='Val Reward', color='orange',
                    linewidth=2, marker='o', markersize=5)

        ax4.set_xlabel('Step')
        ax4.set_ylabel('Accuracy / Reward')
        ax4.set_title('Validation Metrics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_file = self.save_dir / 'training_curves.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Plot updated: {output_file}")
        print(f"  Steps: {len(self.steps)}, Last step: {self.steps[-1] if self.steps else 0}")
        if self.train_metrics['reward_mean']:
            print(f"  Latest train reward: {self.train_metrics['reward_mean'][-1]:.4f}")
        if self.val_metrics['reward_mean']:
            print(f"  Latest val reward: {self.val_metrics['reward_mean'][-1]:.4f}")

    def monitor_file(self, log_file: Path, follow: bool = True):
        """Monitor metrics from a log file."""
        print(f"Monitoring log file: {log_file}")

        if not log_file.exists():
            print(f"Waiting for log file to be created...")
            while not log_file.exists():
                time.sleep(1)

        with open(log_file, 'r') as f:
            # Read existing content
            for line in f:
                metrics = self.parse_console_line(line.strip())
                if metrics:
                    self.add_metrics(metrics)

            self.plot_metrics(force=True)

            if not follow:
                return

            # Follow new content
            print("Following log file for updates (Ctrl+C to stop)...")
            while True:
                line = f.readline()
                if line:
                    metrics = self.parse_console_line(line.strip())
                    if metrics:
                        self.add_metrics(metrics)
                        self.plot_metrics()
                else:
                    time.sleep(0.5)

    def monitor_wandb(self, project: str, run_name: str):
        """Monitor metrics from wandb."""
        try:
            import wandb
            api = wandb.Api()

            print(f"Connecting to wandb project: {project}, run: {run_name}")

            # Find the run
            runs = api.runs(project, filters={"display_name": run_name})
            if not runs:
                print(f"No run found with name: {run_name}")
                return

            run = runs[0]
            print(f"Monitoring run: {run.name} (id: {run.id})")
            print("Press Ctrl+C to stop...")

            while True:
                # Fetch history
                history = run.history()

                for _, row in history.iterrows():
                    metrics = {'step': row.get('_step', 0)}
                    for col in history.columns:
                        if col not in ['_step', '_timestamp', '_runtime']:
                            metrics[col] = row[col]

                    self.add_metrics(metrics)

                self.plot_metrics()
                time.sleep(self.update_interval)

        except ImportError:
            print("wandb is not installed. Install with: pip install wandb")
            sys.exit(1)
        except Exception as e:
            print(f"Error monitoring wandb: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Monitor MemAgent training in real-time")
    parser.add_argument('--mode', type=str, default='file',
                       choices=['file', 'wandb', 'auto'],
                       help='Monitoring mode')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Path to log file (for file mode)')
    parser.add_argument('--project', type=str, default='verl-memagent',
                       help='Wandb project name (for wandb mode)')
    parser.add_argument('--run', type=str, default=None,
                       help='Wandb run name (for wandb mode)')
    parser.add_argument('--exp-dir', type=str, default=None,
                       help='Experiment directory (for auto mode)')
    parser.add_argument('--save-dir', type=str, default='./monitoring_plots',
                       help='Directory to save plots')
    parser.add_argument('--update-interval', type=int, default=10,
                       help='Plot update interval in seconds')
    parser.add_argument('--no-follow', action='store_true',
                       help='Do not follow log file (process once and exit)')

    args = parser.parse_args()

    monitor = TrainingMonitor(save_dir=args.save_dir, update_interval=args.update_interval)

    try:
        if args.mode == 'wandb':
            if not args.run:
                print("Error: --run is required for wandb mode")
                sys.exit(1)
            monitor.monitor_wandb(args.project, args.run)

        elif args.mode == 'file':
            if not args.log_file:
                print("Error: --log-file is required for file mode")
                sys.exit(1)
            log_file = Path(args.log_file)
            monitor.monitor_file(log_file, follow=not args.no_follow)

        elif args.mode == 'auto':
            if not args.exp_dir:
                print("Error: --exp-dir is required for auto mode")
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
                monitor.monitor_file(log_file, follow=not args.no_follow)
            else:
                print(f"No log files found in {exp_dir}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        monitor.plot_metrics(force=True)
        print(f"Final plot saved to: {monitor.save_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
