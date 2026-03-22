"""
LoRA fine-tuning entrypoint for Google Colab and local runs.

No interactive prompts: set HF_TOKEN in the environment and configure Hub repo ids via
training/params.yaml and/or environment variables.

Colab Secrets (🔑): HF_TOKEN (required). HF_LORA_REPO and HF_MODEL_REPO override
params.yaml when set (recommended so you need not edit the repo on GitHub).

Optional training data: TRAINING_DATASET_PATH (local path to JSONL),
TRAINING_DATASET_URL (https URL to download), or Colab secrets of the same names.
Default file: training/params.yaml → dataset_jsonl (usually data/dataset.jsonl).

Dataset format: params.yaml dataset_format or TRAINING_DATASET_FORMAT (env / Colab secret).
Values: alpaca | messages | text | prompt_completion | auto (infer from first row keys).
Locally: use .env (see .env.example) or export the same variable names.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU が使えません。Unsloth の学習には GPU が必須です。\n"
        "Google Colab: メニュー「ランタイム」→「ランタイムのタイプを変更」→"
        "「ハードウェア アクセラレータ」で GPU（T4 など）を選び「保存」してから、"
        "このセル／スクリプトを再実行してください。"
    )

# Unsloth は trl / transformers より先に import する（最適化のため）
from unsloth import FastLanguageModel

import yaml
from datasets import load_dataset
from huggingface_hub import login
from trl import SFTTrainer
from transformers import AutoTokenizer, TrainingArguments


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    path = _repo_root() / ".env"
    if path.is_file():
        load_dotenv(path)


def _load_params() -> dict:
    params_path = _repo_root() / "training" / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _colab_userdata(name: str) -> str:
    """Colab シークレット（環境変数に未反映のときのフォールバック）。"""
    try:
        from google.colab import userdata
    except ImportError:
        return ""
    try:
        raw = userdata.get(name)
    except Exception:
        return ""
    if raw is None:
        return ""
    return str(raw).strip()


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        token = _colab_userdata("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. In Google Colab, add HF_TOKEN in Secrets (🔑) and grant "
            "this notebook access when prompted. Run section 2 before training, or set "
            "os.environ['HF_TOKEN']. Locally: use .env (see .env.example) or export HF_TOKEN."
        )
    return token


def _coerce_float(value, *, name: str) -> float:
    """YAML / 環境によって str になる数値を float へ（bitsandbytes の lr TypeError 対策）。"""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got bool: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise TypeError(f"{name}: expected number, got {type(value).__name__}: {value!r}")


def _resolve_hf_model_repo(params: dict) -> str:
    env_repo = os.environ.get("HF_MODEL_REPO", "").strip()
    if env_repo:
        return env_repo
    colab_repo = _colab_userdata("HF_MODEL_REPO")
    if colab_repo:
        os.environ["HF_MODEL_REPO"] = colab_repo
        return colab_repo
    repo = (params.get("hf_model_repo") or "").strip()
    if not repo:
        raise RuntimeError(
            "Set hf_model_repo in training/params.yaml, or set HF_MODEL_REPO in the environment."
        )
    return repo


def _resolve_hf_lora_repo(params: dict) -> str:
    env_repo = os.environ.get("HF_LORA_REPO", "").strip()
    if env_repo:
        return env_repo
    colab_repo = _colab_userdata("HF_LORA_REPO")
    if colab_repo:
        os.environ["HF_LORA_REPO"] = colab_repo
        return colab_repo
    repo = (params.get("hf_lora_repo") or "").strip()
    if not repo or "YOUR_USERNAME" in repo:
        raise RuntimeError(
            "LoRA の Hub 先（モデル ID）が未設定です。\n"
            "・Google Colab: 左の 🔑 シークレットに **HF_LORA_REPO** を追加し、値に "
            "`あなたのユーザー名/空のモデルリポジトリ名` を入れる。実行時に「このノートブックに許可」"
            "を選ぶ。\n"
            "・または `training/params.yaml` の **hf_lora_repo** を同じ形式の実 ID に書き換える。\n"
            "・ローカル: `.env` に HF_LORA_REPO=... か、環境変数で export する。"
        )
    return repo


def _download_training_dataset_url(url: str, repo_root: Path) -> Path:
    if not url.lower().startswith(("http://", "https://")):
        raise RuntimeError(
            "TRAINING_DATASET_URL must start with http:// or https:// "
            f"(got: {url[:80]!r})"
        )
    cache_dir = repo_root / "training" / ".dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    out = cache_dir / f"from_url_{digest}.jsonl"
    if not out.is_file() or out.stat().st_size == 0:
        print(f"Downloading training dataset: {url}\n  -> {out}")
        urllib.request.urlretrieve(url, out)
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"Download failed or empty file: {out}")
    return out.resolve()


def _resolve_training_dataset_path(params: dict, repo_root: Path) -> Path:
    """TRAINING_DATASET_PATH / TRAINING_DATASET_URL / params dataset_jsonl の順で解決。"""
    raw_path = os.environ.get("TRAINING_DATASET_PATH", "").strip()
    if not raw_path:
        raw_path = _colab_userdata("TRAINING_DATASET_PATH")
    if raw_path:
        p = Path(raw_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"TRAINING_DATASET_PATH は存在するファイルを指してください: {p}"
            )
        return p.resolve()

    raw_url = os.environ.get("TRAINING_DATASET_URL", "").strip()
    if not raw_url:
        raw_url = _colab_userdata("TRAINING_DATASET_URL")
    if raw_url:
        return _download_training_dataset_url(raw_url, repo_root)

    rel = str(params.get("dataset_jsonl") or "data/dataset.jsonl").strip()
    q = Path(rel)
    dataset_path = q if q.is_absolute() else (repo_root / q)
    return dataset_path.resolve()


def _infer_dataset_format_from_row(keys: set[str]) -> str:
    """Infer JSONL record format from column names (first row)."""
    if "messages" in keys:
        return "messages"
    if "instruction" in keys and "output" in keys:
        return "alpaca"
    if "prompt" in keys and "completion" in keys:
        return "prompt_completion"
    if "text" in keys:
        return "text"
    raise RuntimeError(
        "dataset_format=auto ですが、形式を推定できませんでした。"
        f" 先頭行のキー: {sorted(keys)}。"
        " alpaca（instruction/input/output）、messages、prompt+completion、text のいずれかにしてください。"
    )


def _normalize_dataset_format(name: str) -> str:
    s = (name or "").strip().lower().replace("-", "_")
    aliases = {
        "alpaca": "alpaca",
        "messages": "messages",
        "chat": "messages",
        "text": "text",
        "text_only": "text",
        "prompt_completion": "prompt_completion",
        "promptcompletion": "prompt_completion",
        "auto": "auto",
    }
    if s not in aliases:
        raise RuntimeError(
            "dataset_format は alpaca | messages | text | prompt_completion | auto のいずれかにしてください。"
            f"（指定値: {name!r}）"
        )
    return aliases[s]


def _resolve_dataset_format(params: dict, dataset) -> str:
    """TRAINING_DATASET_FORMAT / Colab userdata / params dataset_format の順。"""
    raw = os.environ.get("TRAINING_DATASET_FORMAT", "").strip()
    if not raw:
        raw = _colab_userdata("TRAINING_DATASET_FORMAT")
    if not raw:
        raw = str(params.get("dataset_format") or "alpaca").strip()
    fmt = _normalize_dataset_format(raw)
    if fmt == "auto":
        keys = set(dataset[0].keys())
        fmt = _infer_dataset_format_from_row(keys)
    return fmt


def _prepare_sft_text_dataset(
    dataset,
    *,
    dataset_format: str,
    hf_model_repo: str,
    hf_token: str,
    eos_token: str,
):
    """Build a `text` column for TRL SFTTrainer from Alpaca / messages / text / prompt-completion JSONL."""

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    def _alpaca_batch(examples):
        instructions = examples["instruction"]
        inputs_ = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output_text in zip(instructions, inputs_, outputs):
            texts.append(alpaca_prompt.format(instruction, input_text, output_text) + eos_token)
        return {"text": texts}

    def _prompt_completion_batch(examples):
        texts = []
        for p, c in zip(examples["prompt"], examples["completion"]):
            texts.append(f"{p}{c}{eos_token}")
        return {"text": texts}

    def _text_only_batch(examples):
        texts = []
        for t in examples["text"]:
            if t is None:
                t = ""
            s = str(t)
            if not s.endswith(eos_token):
                s = s + eos_token
            texts.append(s)
        return {"text": texts}

    if dataset_format == "alpaca":
        missing = {"instruction", "input", "output"} - set(dataset.column_names)
        if missing:
            raise RuntimeError(
                f"Alpaca 型には列 instruction, input, output が必要です。足りない列: {sorted(missing)}"
            )
        rc = [c for c in ("instruction", "input", "output") if c in dataset.column_names]
        return dataset.map(_alpaca_batch, batched=True, remove_columns=rc)

    if dataset_format == "prompt_completion":
        missing = {"prompt", "completion"} - set(dataset.column_names)
        if missing:
            raise RuntimeError(
                f"Prompt–Completion 型には列 prompt, completion が必要です。足りない列: {sorted(missing)}"
            )
        rc = [c for c in ("prompt", "completion") if c in dataset.column_names]
        return dataset.map(_prompt_completion_batch, batched=True, remove_columns=rc)

    if dataset_format == "text":
        if "text" not in dataset.column_names:
            raise RuntimeError("Text-only 型には列 text が必要です。")
        rc = [c for c in dataset.column_names if c != "text"]
        return dataset.map(_text_only_batch, batched=True, remove_columns=rc)

    if dataset_format == "messages":
        if "messages" not in dataset.column_names:
            raise RuntimeError("Messages 型には列 messages が必要です。")

        tok = AutoTokenizer.from_pretrained(hf_model_repo, token=hf_token)
        if getattr(tok, "chat_template", None) is None:
            raise RuntimeError(
                f"ベースモデル {hf_model_repo!r} の tokenizer に chat_template がありません。"
                " Messages 型の学習にはチャットテンプレート付きの Instruct モデルを選ぶか、"
                " Alpaca 型へ変換した JSONL を使ってください。"
            )

        def _messages_batch(examples):
            texts = []
            for msgs in examples["messages"]:
                if not msgs:
                    raise RuntimeError("messages が空の行があります。")
                s = tok.apply_chat_template(
                    list(msgs),
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(s + eos_token)
            return {"text": texts}

        rc = [c for c in dataset.column_names if c != "text"]
        return dataset.map(_messages_batch, batched=True, remove_columns=rc)

    raise RuntimeError(f"Unknown dataset_format: {dataset_format!r}")


def main() -> None:
    _maybe_load_dotenv()
    hf_token = _require_hf_token()
    login(token=hf_token, add_to_git_credential=False)

    params = _load_params()
    hf_model_repo = _resolve_hf_model_repo(params)
    hf_lora_repo = _resolve_hf_lora_repo(params)

    # YAML やエディタの都合で数値が str になることがある（bitsandbytes が lr で TypeError）
    lora_r = int(params["lora_r"])
    lora_alpha = int(params["lora_alpha"])
    lora_dropout = float(params["lora_dropout"])
    max_seq_length = int(params["max_seq_length"])
    per_device_train_batch_size = int(params["per_device_train_batch_size"])
    gradient_accumulation_steps = int(params["gradient_accumulation_steps"])
    warmup_steps = int(params["warmup_steps"])
    max_steps = int(params["max_steps"])
    learning_rate = _coerce_float(params["learning_rate"], name="learning_rate")
    fp16 = bool(params["fp16"])
    logging_steps = int(params["logging_steps"])
    output_dir = str(params["output_dir"])
    optim = str(params["optim"])
    seed = int(params["seed"])

    repo_root = _repo_root()
    dataset_path = _resolve_training_dataset_path(params, repo_root)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("2. Data Preparation")
    print(f"   (dataset: {dataset_path})")
    dataset = load_dataset(
        "json",
        data_files={"train": str(dataset_path)},
        split="train",
    )

    EOS_TOKEN = "<|end_of_text|>"
    dataset_format = _resolve_dataset_format(params, dataset)
    print(f"   (dataset_format: {dataset_format})")
    dataset = _prepare_sft_text_dataset(
        dataset,
        dataset_format=dataset_format,
        hf_model_repo=hf_model_repo,
        hf_token=hf_token,
        eos_token=EOS_TOKEN,
    )
    dataset = dataset.select_columns(["text"])
    print("Data preparation complete.")

    print("3. Model Loading")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_model_repo,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        # Unsloth Zoo 2026.x は "current_device" を受け付けず True / False / "unsloth" のみ
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=max_seq_length,
    )
    print("Model loading and LoRA configuration complete.")

    print("5. Training")
    train_output = repo_root / "training" / output_dir
    train_output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        logging_steps=logging_steps,
        output_dir=str(train_output),
        optim=optim,
        seed=seed,
    )
    # Transformers / TRL の内部で str に化けるケースへの保険（bitsandbytes が lr 比較で落ちる）
    object.__setattr__(training_args, "learning_rate", float(training_args.learning_rate))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    trainer.train()
    print("Training complete.")

    merged_dir = repo_root / "training" / "lora_adapter_merged"
    print("6. Saving LoRA adapter locally...")
    model.save_pretrained_merged(str(merged_dir), tokenizer)
    print(f"LoRA adapter saved to {merged_dir}")

    print(f"7. Uploading LoRA adapter to Hugging Face Hub: {hf_lora_repo}")
    model.push_to_hub(hf_lora_repo, token=hf_token)
    tokenizer.push_to_hub(hf_lora_repo, token=hf_token)
    print("LoRA adapter uploaded to Hugging Face Hub.")

    lora_url = f"https://huggingface.co/{hf_lora_repo}"
    base_url = f"https://huggingface.co/{hf_model_repo}"

    print()
    print("#" * 64)
    print("# 学習完了 — Hugging Face の「モデル ID」（repo_id）コピー用")
    print("#" * 64)
    print()
    print("┌─ ① 今回アップロードした LoRA のモデル ID（推論では主にこちら）")
    print("│")
    print(f"│     {hf_lora_repo}")
    print("│")
    print(f"│   → Gradio のアダプタ選択 / DEFAULT_LORA_ADAPTER_REPO に上記をそのまま貼る")
    print(f"│   → Hub: {lora_url}")
    print("│")
    print("└─ （形式は「ユーザー名または組織名 / リポジトリ名」＝ Hub のモデル ID）")
    print()
    print("┌─ ② ベースモデル ID（学習に使った下支え。推論の BASE と一致させる）")
    print("│")
    print(f"│     {hf_model_repo}")
    print("│")
    print(f"│   → inference/app.py の BASE_MODEL_REPO と同じであること")
    print(f"│   → Hub: {base_url}")
    print("└" + "─" * 58)
    print()
    print("#" * 64)


if __name__ == "__main__":
    main()
