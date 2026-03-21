"""リポジトリ構成と設定ファイルの妥当性（GPU・学習ライブラリ不要）。"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_params_yaml_exists_and_loads():
    path = ROOT / "training" / "params.yaml"
    assert path.is_file(), f"Missing {path}"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    for key in (
        "hf_model_repo",
        "hf_lora_repo",
        "lora_r",
        "max_steps",
        "output_dir",
    ):
        assert key in data, f"Missing key: {key}"


def test_dataset_jsonl_exists():
    path = ROOT / "data" / "dataset.jsonl"
    assert path.is_file(), f"Missing {path}"
    text = path.read_text(encoding="utf-8").strip()
    assert text, "dataset.jsonl is empty"


def test_finetune_script_syntax():
    """構文チェック（unsloth 未インストールでも通る）。"""
    path = ROOT / "training" / "finetune_script.py"
    src = path.read_text(encoding="utf-8")
    compile(src, str(path), "exec")
