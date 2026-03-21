# ✨ LLM Fine-tuning & Inference Starter Kit ✨

🚀 Welcome to the LLM Fine-tuning & Inference Starter Kit! This project provides a comprehensive, "completely free" and "engineer-friendly" setup for fine-tuning Llama 3.2 (3B) based LLMs and deploying them for inference.

---

## 🌟 1. プロジェクト概要

このプロジェクトは、Llama 系（Unsloth 4bit）モデルのファインチューニング（学習）から、学習済み LoRA アダプタの Hugging Face Hub へのデプロイ、そして複数のモデルを切り替えて試せる推論環境までをカバーします。Google Colab と Hugging Face Spaces を活用し、コストをかけずに LLM の微調整から公開までを体験できることを目指します。

**操作を最小にする方針**: 学習ノートブックは対話入力（`input` / `notebook_login` 等）を使いません。初回のみ Colab のシークレットに `HF_TOKEN` を登録し、`training/params.yaml` を編集したうえで **「すべてのセルを実行」** すれば学習が完走します。

---

## ▶ 実行手順（早見表）

よく使うのは次のパターンです。**はじめてなら「A」だけ**で問題ありません。細かい説明はこの表のあとの各章にあります。

### A. 学習する（Google Colab・本プロジェクトのメイン）

事前に手元（または GitHub 上）で `params.yaml` を直し、**GitHub に反映**してから Colab を開いてください。

| # | やること |
|---|----------|
| 1 | [Hugging Face](https://huggingface.co/join) にログインし、[Access Tokens](https://huggingface.co/settings/tokens) で **Write** 権限のトークンを作成する。 |
| 2 | Hugging Face 上で **空のモデル用リポジトリ**を新規作成する（例: `あなたのユーザー名/好きなリポジトリ名`）。 |
| 3 | `training/params.yaml` の **`hf_lora_repo`** を、手順 2 の `ユーザー名/リポジトリ名` に書き換えて **コミット・プッシュ**する。 |
| 4 | （**フォークして**使う場合のみ）`training/finetune.ipynb` 内の **`REPO_OWNER`** と **`REPO_NAME`** を自分の GitHub に合わせて編集し、**プッシュ**する。 |
| 5 | この README の **「5. LLM ファインチューニング」** セクションにある **Open in Colab** バッジから `finetune.ipynb` を開く。 |
| 6 | （**初回のみ**）Colab 左メニュー **🔑 シークレット** に、名前 **`HF_TOKEN`**、値に手順 1 のトークンを登録する。GitHub が **プライベート**なら **`GITHUB_TOKEN`**（`repo` スコープ）も追加する。 |
| 7 | メニュー **ランタイム → ランタイムのタイプを変更** で **GPU（T4 など）** を選ぶ。 |
| 8 | メニュー **ランタイム → すべてのセルを実行** を選び、最後まで完了するのを待つ。 |

**結果**: 指定した Hub リポジトリに LoRA がアップロードされます。ノートブック末尾のセルは、Colab 上での簡易推論テスト（任意）です。

### B. 設定ファイルだけ検証する（ローカル・GPU 不要）

**必ず仮想環境を作り、その中でだけ** `pip` してください（システムの Python へ直接インストールしない）。

**Windows (PowerShell)**

```powershell
cd このリポジトリをクローンしたフォルダ
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest tests/ -v
```

**macOS / Linux (bash)**

```bash
cd このリポジトリをクローンしたフォルダ
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/ -v
```

### C. 推論アプリを動かす（任意）

| 場所 | 手順の場所 |
|------|------------|
| **Hugging Face Spaces**（ブラウザ・無料 CPU 想定） | 下の **「6. LLM 推論環境」**（GitHub に push → Space 作成 → `HF_TOKEN` を Secret に設定）。 |
| **自分の PC**（GPU・大きな依存が必要） | 下の **「7. ローカルでの推論アプリ実行」**。 |

---

## 🎯 2. 主要機能

-   **学習環境**:
    *   Google Colab（無料版 T4 GPU）を利用。
    *   VS Code の Google Colab 拡張機能による編集・実行（任意）。
    *   `Unsloth` による高速かつメモリ効率の良い学習。
    *   **非対話**: `HF_TOKEN`（環境変数 / Colab Secrets）と `params.yaml` のみで学習・Hub アップロード。
    *   学習スクリプト: `training/finetune_script.py`（ノートブックから 1 コマンドで実行）。
-   **推論環境**:
    *   Hugging Face Spaces（無料版 CPU Basic）を利用。
    *   ブラウザから操作可能な Gradio チャット UI。
    *   **動的モデル切り替え**: プルダウンから Hub 上の LoRA アダプタを選択してロード。
-   **管理・運用**:
    *   コードは GitHub、成果物は Hugging Face Hub。
    *   秘匿情報は `.env`（ローカル）または各サービスの Secret（Colab / Spaces）で管理し、コードに直書きしない。

---

## 📂 3. フォルダ構成

```text
.
├── documents/             # ドキュメント（要件・設計）
├── training/
│   ├── finetune.ipynb     # Colab 用メインノートブック（Run all）
│   ├── finetune_script.py # 学習エントリ（対話なし）
│   └── params.yaml        # 学習パラメータ・Hub 上の LoRA リポジトリ名
├── inference/             # Hugging Face Spaces 用（Gradio）
├── data/
│   └── dataset.jsonl
├── tests/                   # レイアウト・設定の簡易テスト（GPU 不要）
├── requirements-dev.txt     # テスト用依存（ローカル venv 専用）
├── .env                     # ローカル用（Git 対象外）
└── README.md
```

---

## 🛠️ 4. 環境セットアップガイド

### 1. Hugging Face アカウントとトークン

✅ [Hugging Face](https://huggingface.co/join) でアカウントを作成します。

🔑 **Access Token（Write）**  
[Settings → Access Tokens](https://huggingface.co/settings/tokens) で **Write** 権限のトークンを作成します。学習完了後の `push_to_hub` に必要です。

**Colab での使い方（推奨）**: ノートブック実行前に、Colab 左メニュー **🔑 シークレット** に名前 `HF_TOKEN`、値に上記トークンを保存します。セル内での入力は不要です。

**ローカルで学習スクリプトを試す場合**（GPU が必要）: シェルで `HF_TOKEN` をエクスポートしてから `python training/finetune_script.py` を実行します。

### 2. Google Colab Pro（任意）

無料枠で T4 が付かない・セッションが短い場合は [Colab Pro](https://colab.research.google.com/pricing) の利用を検討してください。

### 3. GitHub とリポジトリの clone（Colab 内）

- **公開リポジトリ**の場合: ノートブックが **トークンなし**で `git clone` します（追加操作なし）。
- **プライベートリポジトリ**の場合: Colab のシークレットに **`GITHUB_TOKEN`**（`repo` スコープ）を登録します。

---

## 🧠 5. LLM ファインチューニング（Colab・最小操作）

> **早見表の「A. 学習する」との対応**: README 冒頭の **「実行手順（早見表）」** に全体のチェックリストがあります。ここでは同じ流れを、補足付きで説明します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TsutsujiStudyTeam/llm-model-generation/blob/main/training/finetune.ipynb)

### 事前準備（ローカルまたは GitHub 上で）

1. `training/params.yaml` の **`hf_lora_repo`** を、自分の Hugging Face 上の**空のモデルリポジトリ**（例: `username/my-lora-adapter`）に変更してコミット・プッシュします。
2. （フォークして使う場合）`training/finetune.ipynb` 内の **`REPO_OWNER`** / **`REPO_NAME`** を、自分の GitHub のユーザー名・リポジトリ名に合わせて編集してから push します。
3. 学習データを `data/dataset.jsonl` に用意します（形式は `documents/アプリケーション詳細設計.md` を参照）。

### Colab での操作（ほぼ「実行」だけ）

1. 上記バッジ、または VS Code の Colab 拡張で `training/finetune.ipynb` を開く。
2. （初回のみ）シークレット **`HF_TOKEN`** を登録する。
3. **ランタイム → ランタイムのタイプを変更** で **GPU（T4 等）** を選択。
4. **ランタイム → すべてのセルを実行**。

ノートブックは依存関係のインストール → リポジトリの clone → `training/finetune_script.py` の実行までを自動で行います。`finetune_script.py` は `HF_TOKEN` と `params.yaml` を読み、**対話入力は行いません**。アップロード先を一時的に変えたい場合は、Colab の「シークレット」ではなく、ランタイム上で `HF_LORA_REPO` を設定する方法もあります（例: 学習セルの直前に `%env HF_LORA_REPO=username/other-repo`）。

### 学習後の確認

- Hugging Face 上の `hf_lora_repo` にファイルがアップロードされていることを確認します。
- ノートブック末尾の（任意）セルで、Colab 上に保存された `training/lora_adapter_merged` を読み込み簡易推論できます。

---

## 💬 6. LLM 推論環境（Hugging Face Spaces）

> **早見表の「C」**: Hugging Face 上で推論 UI を動かす手順です。

### 1. GitHub へプッシュ

```bash
git add .
git commit -m "feat: update inference and training config"
git push origin main
```

### 2. Space の作成（GitHub 連携）

1. [Hugging Face Spaces](https://huggingface.co/spaces/new) で **New Space**。
2. **SDK**: Gradio、**Hardware**: CPU Basic（無料枠）。
3. **Link to a GitHub repository** で本リポジトリを選択（または手動で `inference/` を配置）。
4. Space の **Settings → Secrets** に `HF_TOKEN` を設定（プライベートアダプタの読み込み等に使用）。

Space は `inference/requirements.txt` をインストールし、`inference/app.py` を起動します（リポジトリルートに `app.py` を置く構成の場合は、Space の README に従いパスを調整してください）。

### 3. アプリの利用

プルダウンで LoRA を選び、チャットで試します。初回はモデル・アダプタのダウンロードに時間がかかることがあります。

---

## 🖥️ 7. ローカルでの推論アプリ実行（オプション）

> **早見表の「C」**: 自分の PC で Gradio を起動する場合。GPU と大きな依存が必要です。

**仮想環境を必ず使用**してください（システムの Python へ直接 `pip install` しないこと）。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r inference/requirements.txt
$env:HF_TOKEN="YOUR_TOKEN"
python inference/app.py
```

---

## 🧪 8. 開発者向け: ローカルでのテスト（GPU 不要）

> **早見表の「B」**: 学習はせず、設定ファイルとフォルダ構成だけ検証する手順です。

リポジトリ構成と `params.yaml` の妥当性を **pytest** で確認できます。**仮想環境上で**開発用依存だけを入れてください。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## ⚠️ 9. トラブルシューティング

-   **`HF_TOKEN is not set`**: Colab のシークレット名が正確に `HF_TOKEN` か確認する。ローカルでは環境変数をエクスポートする。
-   **`hf_lora_repo` / `YOUR_HF_USERNAME` エラー**: `training/params.yaml` を自分の Hub リポジトリ名に更新するか、`HF_LORA_REPO` を設定する。
-   **GitHub clone 失敗（プライベート）**: `GITHUB_TOKEN` を Colab シークレットに追加する。
-   **Hugging Face 認証・権限**: トークンに **Write** があり、モデル用リポジトリが作成済みか確認する。
-   **Colab で GPU が付かない**: ランタイムを GPU に変更する。無料枠では付かない時間帯がある。
-   **Spaces のビルド失敗**: Logs を確認。`inference/requirements.txt` と `app.py` のパス・エントリポイントを確認する。

---

## 🤝 10. 貢献

バグ報告・機能要望・プルリクエストを歓迎します。

---

## 📄 11. ライセンス

MIT License。詳細は `LICENSE` を参照してください。
