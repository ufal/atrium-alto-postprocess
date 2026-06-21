import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "alto_stats_create.py"


def test_alto_stats_cli_help():
    """Ensure the statistics generator script compiles and parses help arguments."""
    result = subprocess.run([sys.executable, str(SCRIPT_PATH), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()


def test_alto_stats_missing_args():
    """Ensure the script handles missing directory arguments safely."""
    result = subprocess.run([sys.executable, str(SCRIPT_PATH)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "error" in result.stderr.lower() or "required" in result.stderr.lower()
