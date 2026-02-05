# ✨ LLM Fine-tuning & Inference Starter Kit ✨

🚀 Welcome to the LLM Fine-tuning & Inference Starter Kit! This project provides a comprehensive, "completely free" and "engineer-friendly" setup for fine-tuning Llama 3.2 (3B) based LLMs and deploying them for inference.

---

## 🌟 1. プロジェクト概要

このプロジェクトは、Llama-3.2-3B-Instructモデルのファインチューニング（学習）から、学習済みLoRAアダプタのHugging Face Hubへのデプロイ、そして複数のモデルを切り替えて試せる推論環境までをカバーします。Google ColabとHugging Face Spacesを活用し、コストをかけずに最新のLLM開発を体験できることを目指しています。

---

## 🎯 2. 主要機能

-   **学習環境**:
    *   Google Colab (無料版 T4 GPU) を利用。
    *   VS CodeのGoogle Colab拡張機能による快適な開発体験。
    *   `Unsloth` ライブラリによる高速かつメモリ効率の良い学習。
    *   学習済みLoRAアダプタのHugging Face Hubへの直接アップロード。
-   **推論環境**:
    *   Hugging Face Spaces (無料版 CPU Basic) を利用。
    *   ブラウザから操作可能なGradioチャットUIを提供。
    *   **動的モデル切り替え**: プルダウンメニューからHugging Face Hub上の異なるLoRAアダプタを選択し、即座にロードして推論を実行可能。
-   **管理・運用**:
    *   コード管理はGitHub、成果物管理はHugging Face Hub。
    *   秘匿情報の安全な管理（`.env` またはサービス側のSecret管理）。

---

## 📂 3. フォルダ構成

プロジェクトの主要なフォルダ構成は以下の通りです。

```text
.
├── documents/             # 📄 ドキュメント類 (要件定義、設計など)
│   ├── 要件定義.md
│   ├── システムアーキテクチャ.md
│   ├── アプリケーション詳細設計.md
│   └── 運用設計.md
├── training/              # 🧠 学習用関連ファイル
│   ├── finetune.ipynb    # メインの学習ノートブック
│   └── params.yaml       # 学習パラメータ設定
├── inference/             # 💬 推論用関連ファイル (Hugging Face Spaces同期用)
│   ├── app.py            # 推論UIメインコード (Gradio)
│   └── requirements.txt  # 推論環境依存ライブラリ
├── data/                  # 📊 学習用データセット
│   └── dataset.jsonl     # 練習用サンプルデータ (JSONL形式)
├── .env                   # 🔑 ローカル環境変数 (Git管理対象外)
└── README.md              # 📖 プロジェクト概要と手順 (このファイル)
```

---

## 🛠️ 4. 環境セットアップガイド

このプロジェクトを開始する前に、以下の準備が必要です。

### 1. リポジトリのクローン

まず、このリポジトリをローカル環境にクローンします。

```bash
git clone https://github.com/your-username/llm-model-generation.git
cd llm-model-generation
```

---

### 2. Hugging Faceアカウントとアクセストークン

<p align="center">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD200?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Logo">
</p>

✅ **Hugging Faceアカウントの作成**:
まだアカウントをお持ちでない場合は、[Hugging Face](https://huggingface.co/join) でサインアップしてください。

🔑 **アクセストークンの発行**:
ファインチューニングしたLoRAアダプタをHugging Face Hubにアップロードしたり、プライベートなモデルをロードするために必要です。
1.  Hugging Faceのウェブサイトにログインします。
2.  右上のプロフィールアイコンをクリックし、「Settings」を選択します。
3.  左側のメニューで「Access Tokens」を選択します。
4.  新しいトークンを作成し、「Write」権限を与えてください。このトークンは安全に保管してください。

💡 **ヒント**: このトークンは後でColabで認証する際や、Hugging Face Spacesで環境変数として設定する際に使用します。

---

### 3. `training/params.yaml` の更新

学習済みLoRAアダプタのアップロード先を設定します。

1.  `training/params.yaml` ファイルを開きます。
2.  `hf_lora_repo` の値を、ご自身のHugging Faceユーザー名（または組織名）とリポジトリ名に更新します。

    ```yaml
    # training/params.yaml の一部

    # Hugging Face LoRA Adapter Repository (USER MUST UPDATE THIS)
    # 例: "YOUR_HF_USERNAME/llama-3.2-lora-adapter"
    hf_lora_repo: "YOUR_HF_USERNAME/my-llama-3.2-lora-adapter" # 💡 ここをご自身の情報に書き換えてください！
    ```

    ⚠️ **重要**: `YOUR_HF_USERNAME` の部分は、ご自身のHugging Faceのユーザー名（または、アダプタをアップロードしたい組織のID）に置き換えてください。

---

## 🧠 5. LLMファインチューニングガイド (Google Colab)

ここでは、VS CodeのColab拡張機能を使ってファインチューニングを実行する手順を説明します。

<p align="center">
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab Logo">
  <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" alt="VS Code Logo">
</p>

### 1. VS CodeでGoogle Colab拡張機能を使用

1.  VS Codeを開き、左側のアクティビティバーから「Remote Explorer」アイコン（雲のアイコン）をクリックします。
2.  「Colab」セクションを選択し、Googleアカウントでログインします。
3.  新しいColabノートブックのセッションを開始します。
4.  VS Codeのファイルエクスプローラーで、プロジェクト内の `training/finetune.ipynb` を開きます。この時、Colabランタイムに接続していることを確認してください。

    💡 **ヒント**: これにより、ローカルのVS Codeでノートブックを編集し、Colabの無料GPUリソースで実行できるようになります。ファイルをColabに手動でアップロードする必要はありません。

### 2. `finetune.ipynb` の実行

`finetune.ipynb` ノートブックは、以下のステップで構成されています。各セルを上から順に実行してください。

#### ステップ 1: セットアップとHugging Face認証
-   **目的**: 必要なライブラリのインストールとHugging Faceアクセストークンによる認証を行います。
-   **アクション**: 最初の数セルを実行します。Hugging Faceのログインプロンプトが表示されたら、ステップ4.2で取得したアクセストークンを入力してください。

    <p align="center">
      <img src="https://via.placeholder.com/600x200?text=Diagram:+Hugging+Face+Login+Prompt" alt="Hugging Face Login Prompt Diagram">
      <br>
      <em>図1: Hugging Faceログインプロンプトのイメージ</em>
    </p>

#### ステップ 2: データ準備
-   **目的**: `data/dataset.jsonl` からデータセットをロードし、学習に適した形式に変換します。
-   **アクション**: 関連セルを実行します。

#### ステップ 3 & 4: モデルとトークナイザーのロード
-   **目的**: ベースとなるLlamaモデルと対応するトークナイザーをロードし、LoRAファインチューニングのための設定を行います。
-   **アクション**: 関連セルを実行します。Unslothが自動的に4bit量子化されたモデルをロードし、LoRAアダプタを準備します。

    <p align="center">
      <img src="https://via.placeholder.com/600x250?text=Diagram:+Model+Loading+Process" alt="Model Loading Process Diagram">
      <br>
      <em>図2: モデルロードとLoRA設定のイメージ</em>
    </p>

#### ステップ 5: トレーニング
-   **目的**: 設定されたハイパーパラメータに基づいてモデルのファインチューニングを開始します。
-   **アクション**: トレーニングセルを実行します。これには時間がかかります（ColabのT4 GPUで数分から数十分）。

    <p align="center">
      <img src="https://via.placeholder.com/600x200?text=Diagram:+Training+Progress+Metrics" alt="Training Progress Metrics Diagram">
      <br>
      <em>図3: トレーニング進行状況の出力イメージ</em>
    </p>

#### ステップ 6: LoRAアダプタの保存
-   **目的**: ファインチューニングが完了した後、学習済みのLoRAアダプタをローカルに保存します。
-   **アクション**: 関連セルを実行します。

#### ステップ 7: Hugging Face Hubへのアップロード
-   **目的**: 保存したLoRAアダプタをHugging Face Hubにプッシュします。
-   **アクション**: 関連セルを実行します。これにより、ステップ4.3で `params.yaml` に設定したリポジトリにアダプタがアップロードされます。

    <p align="center">
      <img src="https://via.placeholder.com/600x150?text=Diagram:+Hugging+Face+Hub+Upload+Confirmation" alt="Hugging Face Hub Upload Confirmation Diagram">
      <br>
      <em>図4: Hugging Face Hubへのアップロード成功イメージ</em>
    </p>

#### ステップ 8: 推論の確認 (Colab内)
-   **目的**: ファインチューニングされたモデルが意図通りに動作するかをColab内で簡単に確認します。
-   **アクション**: 最後のセルを実行し、サンプルプロンプトに対するモデルの応答を確認します。

---

## 💬 6. LLM推論環境ガイド (Hugging Face Spaces)

ファインチューニングしたモデルをGradioアプリとしてHugging Face Spacesにデプロイし、ブラウザから操作できるようにします。

<p align="center">
  <img src="https://img.shields.io/badge/Hugging%20Face%20Spaces-000000?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Spaces Logo">
</p>

### 1. GitHubへのプッシュ

Hugging Face Spacesは、GitHubリポジトリと直接連携してデプロイできます。
プロジェクトの変更（特に `inference/` フォルダの内容）をGitHubリポジトリにプッシュします。

```bash
git add .
git commit -m "feat: complete initial project setup and code"
git push origin main # あるいはご自身のブランチ
```

### 2. Hugging Face Spacesの作成とGitHub連携

1.  [Hugging Face Spaces](https://huggingface.co/spaces/new) にアクセスし、ログインします。
2.  「New Space」ボタンをクリックします。
3.  以下の情報を入力します。
    *   **Space name**: 推論アプリの名前（例: `my-llm-inference-demo`）
    *   **Owner**: ご自身のユーザー名または組織を選択
    *   **Visibility**: `Public` または `Private` (必要に応じて)
    *   **SDK**: `Gradio` を選択
    *   **Space hardware**: `CPU Basic` を選択 (無料枠)
    *   **Link to a GitHub repository**: **このオプションを有効にし、ステップ6.1でプッシュしたご自身のGitHubリポジトリを選択します。**

    <p align="center">
      <img src="https://via.placeholder.com/600x350?text=Diagram:+Hugging+Face+Spaces+Creation" alt="Hugging Face Spaces Creation Diagram">
      <br>
      <em>図5: Hugging Face Spaces作成画面のイメージ</em>
    </p>

4.  「Create Space」をクリックします。
5.  Hugging Face Spacesが自動的にGitHubリポジトリからコードをクローンし、`inference/requirements.txt` に基づいて依存ライブラリをインストールし、`inference/app.py` を実行してアプリケーションを起動します。

    💡 **ヒント**: デプロイのログはSpaceの「Logs」タブで確認できます。エラーが発生した場合はここでデバッグしてください。

### 3. 推論アプリの使用

デプロイが成功すると、Hugging Face SpacesのページにGradioのチャットUIが表示されます。

1.  **LoRAアダプタの選択**: UI上部のプルダウンメニューから、Hugging Face HubにアップロードしたLoRAアダプタを選択します。これにより、選択したアダプタが動的にロードされます。
    *   初回ロード時は、モデルとアダプタのダウンロードに時間がかかる場合があります。
2.  **チャットの開始**: テキストボックスに質問を入力し、「Send」ボタンをクリックして、ファインチューニングされたモデルとの会話を開始します。

    <p align="center">
      <img src="https://via.placeholder.com/600x400?text=Diagram:+Gradio+Chat+UI+with+Dropdown" alt="Gradio Chat UI with Dropdown Diagram">
      <br>
      <em>図6: GradioチャットUIとLoRA選択ドロップダウンのイメージ</em>
    </p>

---

## 🖥️ 7. ローカルでの推論アプリ実行 (オプション)

Hugging Face Spacesにデプロイする前に、ローカルで推論アプリをテストしたい場合は以下の手順を実行します。

1.  **仮想環境の作成とアクティベート** (推奨)

    ```bash
    python -m venv venv
    ./venv/Scripts/activate # Windows
    source venv/bin/activate # macOS/Linux
    ```

2.  **依存ライブラリのインストール**

    ```bash
    pip install -r inference/requirements.txt
    ```

3.  **環境変数の設定** (必要な場合)
    プライベートなHugging FaceリポジトリからLoRAアダプタをロードする場合は、Hugging Faceアクセストークンを環境変数 `HF_TOKEN` として設定します。

    ```bash
    # Windows PowerShell
    $env:HF_TOKEN="YOUR_HF_READ_TOKEN"

    # macOS/Linux Bash
    export HF_TOKEN="YOUR_HF_READ_TOKEN"
    ```

4.  **Gradioアプリの実行**

    ```bash
    python inference/app.py
    ```
    アプリケーションが起動すると、ローカルホストのURL（通常は `http://127.0.0.1:7860`）が表示されます。ブラウザでそのURLにアクセスしてください。

---

## ⚠️ 8. トラブルシューティング

-   **Hugging Face認証エラー**: アクセストークンが正しいか、十分な権限（Write権限）があるかを確認してください。
-   **ColabでのGPU不足**: Colab無料版ではT4 GPUが割り当てられない場合があります。ランタイムタイプを「GPU」に設定しているか確認し、数回試してみてください。
-   **Hugging Face Spacesでのデプロイ失敗**:
    *   Spaceの「Logs」タブを確認し、エラーメッセージを特定してください。
    *   `inference/requirements.txt` に必要なライブラリがすべて含まれているか確認してください。
    *   `inference/app.py` に構文エラーがないか確認してください。
    *   `BASE_MODEL_REPO` や `DEFAULT_LORA_ADAPTER_REPO` が正しいリポジトリ名を指しているか確認してください。
-   **LoRAアダプタのロード失敗**:
    *   Hugging Face Hubにアダプタが正しくアップロードされているか確認してください。
    *   `inference/app.py` 内の `HF_TOKEN` 環境変数が正しく設定されているか（プライベートリポジトリの場合）確認してください。

---

## 🤝 9. 貢献

このプロジェクトへの貢献を歓迎します！バグレポート、機能リクエスト、プルリクエストなど、お気軽にお寄せください。

---

## 📄 10. ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細については `LICENSE` ファイルを参照してください。
`README.md` には `LICENSE` ファイルが存在する旨を記載するが、今回は `LICENSE` ファイル自体は作成しない。