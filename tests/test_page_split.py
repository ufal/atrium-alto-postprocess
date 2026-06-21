import subprocess
import sys
from pathlib import Path

# Target the page_split.py script in the repository root
SCRIPT_PATH = Path(__file__).parent.parent / "page_split.py"


def test_page_split_cli_help():
    """Smoke test to ensure the script compiles and parses arguments."""
    result = subprocess.run([sys.executable, str(SCRIPT_PATH), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()


def test_page_split_cli_missing_args():
    """Ensure the script fails gracefully when required arguments are omitted."""
    result = subprocess.run([sys.executable, str(SCRIPT_PATH)], capture_output=True, text=True)
    # The script should exit with an error code due to missing required positional/flag args
    assert result.returncode != 0
