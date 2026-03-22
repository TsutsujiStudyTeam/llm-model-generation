"""
LoRA fine-tuning entrypoint for Google Colab and local runs.

No interactive prompts: configure training/params.yaml and set HF_TOKEN in the environment.
In Colab, register HF_TOKEN under Secrets (🔑). Optionally override the adapter repo with
HF_LORA_REPO.
"""

from __future__ import annotations

import os
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
from transformers import TrainingArguments


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


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. In Google Colab, add HF_TOKEN in Secrets (🔑). "
            "Locally: create a .env file in the repo root with HF_TOKEN=... (see .env.example), "
            "or export HF_TOKEN in the shell."
        )
    return token


def _resolve_hf_lora_repo(params: dict) -> str:
    env_repo = os.environ.get("HF_LORA_REPO", "").strip()
    if env_repo:
        return env_repo
    repo = (params.get("hf_lora_repo") or "").strip()
    if not repo or "YOUR_HF_USERNAME" in repo:
        raise RuntimeError(
            "Set hf_lora_repo in training/params.yaml to your Hugging Face repo id "
            '(e.g. "username/my-adapter"), or set HF_LORA_REPO in the environment.'
        )
    return repo


def main() -> None:
    _maybe_load_dotenv()
    hf_token = _require_hf_token()
    login(token=hf_token, add_to_git_credential=False)

    params = _load_params()
    hf_model_repo = params["hf_model_repo"]
    hf_lora_repo = _resolve_hf_lora_repo(params)

    lora_r = params["lora_r"]
    lora_alpha = params["lora_alpha"]
    lora_dropout = params["lora_dropout"]
    max_seq_length = params["max_seq_length"]
    per_device_train_batch_size = params["per_device_train_batch_size"]
    gradient_accumulation_steps = params["gradient_accumulation_steps"]
    warmup_steps = params["warmup_steps"]
    max_steps = params["max_steps"]
    learning_rate = params["learning_rate"]
    fp16 = params["fp16"]
    logging_steps = params["logging_steps"]
    output_dir = params["output_dir"]
    optim = params["optim"]
    seed = params["seed"]

    repo_root = _repo_root()
    dataset_path = repo_root / "data" / "dataset.jsonl"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("2. Data Preparation")
    dataset = load_dataset(
        "json",
        data_files={"train": str(dataset_path)},
        split="train",
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = "<|end_of_text|>"

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs_ = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output_text in zip(instructions, inputs_, outputs):
            text = alpaca_prompt.format(instruction, input_text, output_text) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
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
        use_gradient_checkpointing="current_device",
        random_state=seed,
        max_seq_length=max_seq_length,
    )
    print("Model loading and LoRA configuration complete.")

    print("5. Training")
    train_output = repo_root / "training" / output_dir
    train_output.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
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
        ),
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
