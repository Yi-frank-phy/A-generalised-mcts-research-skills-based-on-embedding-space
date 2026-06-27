import subprocess
import sys


def test_smoke_workflow_script_completes():
    completed = subprocess.run(
        [sys.executable, "scripts/smoke_workflow.py"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "DTE smoke workflow ok" in completed.stdout
