import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from contrabin import __version__
from contrabin.cli import app

runner = CliRunner()


def test_cli_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert __version__ in res.output


def test_cli_make_synthetic(tmp_path: Path):
    out = tmp_path / "syn.jsonl"
    res = runner.invoke(app, ["make-synthetic", "-o", str(out), "-n", "4"])
    assert res.exit_code == 0, res.output
    assert out.exists()
    assert sum(1 for _ in out.open()) == 4


def test_cli_write_default_config(tmp_path: Path):
    out = tmp_path / "c.yaml"
    res = runner.invoke(app, ["write-default-config", "-o", str(out)])
    assert res.exit_code == 0
    assert out.exists()


def test_cli_entrypoint_help():
    # Use the installed entry point, smoke-test that it doesn't crash.
    r = subprocess.run(
        [sys.executable, "-m", "contrabin.cli", "--help"], capture_output=True, text=True
    )
    assert r.returncode == 0
