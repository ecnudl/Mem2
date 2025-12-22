#!/usr/bin/env python3
"""
Test script to validate monitoring functionality.
Creates sample log data and tests the monitoring tools.
"""

import sys
import time
from pathlib import Path
import tempfile

def generate_sample_log():
    """Generate sample training log for testing."""
    log_lines = []

    # Simulate training progress
    for step in range(0, 100, 5):
        # Simulate decreasing loss and increasing reward
        actor_loss = 0.5 - step * 0.003 + (step % 3) * 0.01
        critic_loss = 1.2 - step * 0.005 + (step % 5) * 0.02
        reward_mean = 0.3 + step * 0.005 + (step % 4) * 0.01
        reward_std = 0.1 + (step % 2) * 0.02
        entropy = 0.5 - step * 0.002
        kl = 0.01 + (step % 3) * 0.005

        log_line = (
            f"step:{step} - actor_loss:{actor_loss:.3f} - "
            f"critic_loss:{critic_loss:.3f} - reward_mean:{reward_mean:.3f} - "
            f"reward_std:{reward_std:.3f} - entropy:{entropy:.3f} - kl:{kl:.3f}"
        )
        log_lines.append(log_line)

        # Add validation metrics every 20 steps
        if step % 20 == 0 and step > 0:
            val_reward = 0.4 + step * 0.006
            val_accuracy = 0.5 + step * 0.004
            val_line = (
                f"step:{step} - val/reward_mean:{val_reward:.3f} - "
                f"val/accuracy:{val_accuracy:.3f}"
            )
            log_lines.append(val_line)

    return "\n".join(log_lines)


def test_monitor_parsing():
    """Test if monitor can parse log lines correctly."""
    print("Testing monitor parsing...")

    sys.path.insert(0, str(Path(__file__).parent))
    from monitor_training import TrainingMonitor

    monitor = TrainingMonitor()

    # Test parsing
    test_line = "step:100 - actor_loss:0.456 - critic_loss:1.234 - reward_mean:0.789"
    metrics = monitor.parse_console_line(test_line)

    assert metrics is not None, "Failed to parse line"
    assert metrics['step'] == 100, "Step not parsed correctly"
    assert abs(metrics['actor_loss'] - 0.456) < 1e-6, "Actor loss not parsed correctly"
    assert abs(metrics['reward_mean'] - 0.789) < 1e-6, "Reward not parsed correctly"

    print("✓ Parsing test passed")
    return True


def test_monitor_plot():
    """Test if monitor can generate plots."""
    print("\nTesting plot generation...")

    sys.path.insert(0, str(Path(__file__).parent))
    from monitor_training import TrainingMonitor

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = TrainingMonitor(save_dir=tmpdir)

        # Add some sample data
        for step in range(10):
            metrics = {
                'step': step * 10,
                'actor_loss': 0.5 - step * 0.03,
                'critic_loss': 1.2 - step * 0.05,
                'reward_mean': 0.3 + step * 0.05,
                'reward_std': 0.1,
                'entropy': 0.5 - step * 0.02,
                'kl': 0.01
            }
            monitor.add_metrics(metrics)

        # Try to generate plot
        monitor.plot_metrics(force=True)

        # Check if plot was created
        plot_file = Path(tmpdir) / 'training_curves.png'
        assert plot_file.exists(), "Plot file was not created"

        print(f"✓ Plot generation test passed")
        print(f"  Sample plot created at: {plot_file}")
        return True


def test_analyzer():
    """Test if analyzer can process data."""
    print("\nTesting analyzer...")

    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_training import TrainingAnalyzer

    # Create temporary log file
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / 'test.log'
        log_content = generate_sample_log()
        log_file.write_text(log_content)

        analyzer = TrainingAnalyzer(output_dir=tmpdir)
        analyzer.load_from_file(log_file)

        assert len(analyzer.metrics_history) > 0, "No metrics loaded"

        # Test trend analysis
        steps, values = analyzer.get_metric_series('reward_mean')
        assert len(values) > 0, "No reward values found"

        analysis = analyzer.analyze_trend(values)
        assert analysis['trend'] == 'increasing', f"Expected increasing trend, got {analysis['trend']}"

        # Test report generation
        analyzer.generate_report()

        # Check if report was created
        report_file = Path(tmpdir) / 'training_report.json'
        assert report_file.exists(), "Report file was not created"

        print("✓ Analyzer test passed")
        print(f"  Sample report created at: {report_file}")
        return True


def main():
    print("="*60)
    print("MemAgent Monitoring Tools - Test Suite")
    print("="*60)

    try:
        # Check dependencies
        print("\nChecking dependencies...")
        import matplotlib
        import numpy
        print("✓ All dependencies available")

        # Run tests
        test_monitor_parsing()
        test_monitor_plot()
        test_analyzer()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        print("\nYou can now use the monitoring tools:")
        print("  - Real-time monitoring: ./quick_monitor.sh")
        print("  - Post-training analysis: ./quick_analyze.sh")
        print("\nFor more details, see README.md")

    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("  pip install matplotlib numpy scipy")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
