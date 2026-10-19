import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def run_cli(method, payload):
    return subprocess.run(
        [sys.executable, "-m", "epiplexity", method, "-"],
        input=json.dumps(payload), text=True, capture_output=True, cwd=ROOT,
    )


def test_prequential_cli_reads_stdin_and_emits_json():
    completed = run_cli("prequential", {
        "tokens": [0, 2, 5], "losses": [4, 2, 1], "loss_unit": "bits",
        "metadata": {"seed": 3407, "dataset_digest": "example"},
    })
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["total_bits"] == 10.5
    assert result["metadata"]["seed"] == 3407
    assert completed.stderr == ""


def test_requential_cli_reads_a_file(tmp_path):
    log = tmp_path / "training.json"
    log.write_text(json.dumps({"kl_values": [0.5], "token_counts": [4],
                               "final_loss": 1, "evaluation_tokens": 8,
                               "loss_unit": "bits"}))
    completed = subprocess.run(
        [sys.executable, "-m", "epiplexity", "requential", str(log)],
        text=True, capture_output=True, cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["total_bits"] == 10


def test_invalid_cli_input_has_nonzero_exit_and_no_partial_result():
    for payload in [{"tokens": [0], "losses": [1]}, [], {"bogus": 2}]:
        completed = run_cli("prequential", payload)
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert "error:" in completed.stderr


def test_core_import_does_not_load_ml_frameworks():
    completed = subprocess.run(
        [sys.executable, "-c", "import epiplexity, sys; "
         "assert not {'torch', 'transformers', 'scipy'} & sys.modules.keys(); "
         "assert epiplexity.PrequentialEstimator"],
        text=True, capture_output=True, cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr
