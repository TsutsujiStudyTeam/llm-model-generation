# ✨ LLM 学習&推論 体験キット ✨

## 🌟 1. プロジェクト概要

このプロジェクトは、Llama 系（Unsloth 4bit）モデルのファインチューニング（学習）から、学習済み LoRA アダプタの Hugging Face Hub へのデプロイ、そして複数のモデルを切り替えて試せる推論環境を作りました。
Google Colab と Hugging Face Spaces を活用し、コストをかけずに LLM の学習から推論までを体験できることを目指します。

**特徴：操作は最小限**: 学習ノートブックは対話入力（`input` / `notebook_login` 等）を使いません。初回のみ Colab のシークレットに `HF_TOKEN` を登録し、`training/params.yaml` を編集したうえで **「すべてのセルを実行」** すれば学習が完走します。

---

## ▶ 実行手順

### A. LLMに学習させる

事前に手元（または GitHub 上）で `params.yaml` の `hf_lora_repo` を直し、**GitHub に反映**してから Colab を開いてください。

| # | やること |
|---|----------|
| 1 | [Hugging Face](https://huggingface.co/join) にログインし、[Access Tokens](https://huggingface.co/settings/tokens) で **Write** 権限のトークンを作成する。 |
| 2 | Hugging Face 上で **空のモデル用リポジトリ**を新規作成する（例: `あなたのユーザー名/好きなリポジトリ名`）。 |
| 3 | `training/params.yaml` の **`hf_lora_repo`** を、手順 2 の `ユーザー名/リポジトリ名` に書き換えて **コミット・プッシュ**する。 |
| 4 | ブラウザからGoogle Colabにアクセスし、GitHUBからこのプロジェクトをロードする。 |
| 5 | （**初回のみ**）Colab 左メニュー **🔑 シークレット** に、名前 **`HF_TOKEN`**、値に手順 1 のトークンを登録する。 |
| 6 | メニュー **ランタイム → ランタイムのタイプを変更** で **GPU（T4 など）** を選ぶ。 |
| 7 | メニュー **ランタイム → すべてのセルを実行** を選び、最後まで完了するのを待つ。 |

**結果**: 指定した Hub リポジトリに LoRA学習したモデル がアップロードされます。ノートブック末尾のセルは、Colab 上での簡易推論テストできるものです。

### B. 学習させたモデルで推論する

| 場所 | 手順の場所 |
|------|------------|
| **Hugging Face Spaces** | 下の **「6. LLM 推論環境」**（GitHub に push → Space 作成 → `HF_TOKEN` を Secret に設定）。 |
| **自分の PC**（GPU・大きな依存が必要） | 下の **「7. ローカルでの推論アプリ実行」**。 |

- TODO：手順を準備

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
├── .env.example             # HF_TOKEN 記載例（.env にコピーして利用）
├── .env                     # ローカル用（Git 対象外・自分で作成）
└── README.md
```

---

## 🛠️ 4. 環境セットアップガイド

### Hugging Face アカウントとトークン

✅ [Hugging Face](https://huggingface.co/join) でアカウントを作成します。

🔑 **Access Token（Write）**  
[Settings → Access Tokens](https://huggingface.co/settings/tokens) で **Write** 権限のトークンを作成します。学習完了後の `push_to_hub` に必要です。

---

## 💬 5. クラウド推論環境（Hugging Face Spaces）での推論実行

### 1. GitHub へプッシュ

### 2. Space の作成（GitHub 連携）

1. [Hugging Face Spaces](https://huggingface.co/spaces/new) で **New Space**。
2. **SDK**: Gradio、**Hardware**: CPU Basic（無料枠）。
3. **Link to a GitHub repository** で本リポジトリを選択（または手動で `inference/` を配置）。
4. Space の **Settings → Secrets** に `HF_TOKEN` を設定（プライベートアダプタの読み込み等に使用）。

Space は `inference/requirements.txt` をインストールし、`inference/app.py` を起動します（リポジトリルートに `app.py` を置く構成の場合は、Space の README に従いパスを調整してください）。

### 3. アプリの利用

プルダウンで LoRAモデル を選び、チャットで試します。初回はモデル・アダプタのダウンロードに時間がかかることがあります。

---

## 🖥️ 6. ローカルでの推論実行

**仮想環境を必ず使用**してください（システムの Python へ直接 `pip install` しないこと）。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r inference/requirements.txt
python inference/app.py
```

---

## 🤝 7. 貢献

バグ報告・機能要望・プルリクエストを歓迎します。

---

## 📄 8. ライセンス

MIT License。詳細は `LICENSE` を参照してください。
