# ✨ LLM Fine-tuning & Inference Starter Kit ✨

🚀 Welcome to the LLM Fine-tuning & Inference Starter Kit! This project provides a comprehensive, "completely free" and "engineer-friendly" setup for fine-tuning Llama 3.2 (3B) based LLMs and deploying them for inference.

---

## 🌟 1. プロジェクト概要

このプロジェクトは、Llama-3.2-3B-Instructモデルのファインチューニング（学習）から、学習済みLoRAアダプタのHugging Face Hubへのデプロイ、そして複数のモデルを切り替えて試せる推論環境までをカバーします。Google ColabとHugging Face Spacesを活用し、コストをかけずに最新のLLM開発を体験できることを目指します。

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
│   ├── 運用設計.md
│   └── 設計.md
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

このプロジェクトは、VS CodeのColab拡張機能を通じてGitHubから直接クローンされます。**手動でリポジトリをクローンする必要はありません。**

### 2. Hugging FaceアカウントとPersonal Access Token (PAT)

<p align="center">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD200?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Logo">
</p>

✅ **Hugging Faceアカウントの作成**:
まだアカウントをお持ちでない場合は、[Hugging Face](https://huggingface.co/join) でサインアップしてください。

🔑 **Personal Access Token (PAT) の発行**:
ファインチューニングしたLoRAアダプタをHugging Face Hubにアップロードしたり、プライベートなモデルをロードするために必要です。
1.  Hugging Faceのウェブサイトにログインします。
2.  右上のプロフィールアイコンをクリックし、「Settings」を選択します。
3.  左側のメニューで「Access Tokens」を選択します。
4.  新しいトークンを作成し、「Write」権限を与えてください（リポジトリへのアップロードに必要です）。このトークンは安全に保管し、**後でColabで入力を求められた際に使用します**。

💡 **ヒント**: ColabでのGitHubプライベートリポジトリのクローンやHugging Face Hubへのモデルアップロード時に、このPATの入力を求められます。

---

### 3. Google Colab Proの使用 (推奨)

Colab無料版でもT4 GPUが提供されることがありますが、より安定した環境と長時間の実行が必要な場合は、[Google Colab Pro](https://colab.research.google.com/pricing) の使用を検討してください。

---

## 🧠 5. LLMファインチューニングガイド (VS Code & Google Colab)

ここでは、VS CodeのColab拡張機能を使ってファインチューニングを実行する手順を説明します。

<p align="center">
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab Logo">
  <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" alt="VS Code Logo">
</p>

### 1. VS CodeでColab環境をセットアップ

1.  **VS Codeを開き、`training/finetune.ipynb` ファイルを開きます。**
2.  VS Codeの右上に表示される「サーバー選択」のようなアイコン（通常は「Colab」または地球儀のアイコン）をクリックし、「**Connect to Google Colab**」を選択します。
3.  ブラウザが起動し、Googleアカウントの認証を求められます。ご自身のGoogleアカウントで認証を完了してください。
4.  認証後、VS CodeのColab接続画面に戻り、**「GPU」タブを選択し、「T4 GPU」など利用可能なGPUランタイムを選択します。** これにより、ColabのGPUインスタンスが割り当てられます。

    💡 **ヒント**: この手順により、ローカルのVS CodeからColabのGPUリソースに接続し、ノートブックを直接実行・編集できるようになります。

### 2. `finetune.ipynb` の実行

`finetune.ipynb` ノートブックは、以下のステップで構成されています。**各セルを上から順に実行してください。**

#### ステップ 1: セットアップとGitHubリポジトリのクローン
-   **目的**: 必要なライブラリのインストールと、GitHubプライベートリポジトリのクローンを行います。
-   **アクション**: 最初の数セルを実行します。
    *   **GitHub Personal Access Token (PAT) の入力**: `Enter your GitHub Personal Access Token:` というプロンプトが表示されたら、ステップ4.2で発行したPATを正確に入力してください。これにより、リポジトリがColab環境にクローンされ、ノートブックがクローンされたディレクトリに移動します。

#### ステップ 2: パラメータ設定とHugging Faceリポジトリ名の入力
-   **目的**: 学習パラメータをロードし、Hugging Face HubにアップロードするLoRAアダプタのリポジトリ名を設定します。
-   **アクション**: 関連セルを実行します。
    *   **Hugging Face LoRA Repository Name の入力**: `Enter your Hugging Face LoRA Repository Name:` というプロンプトが表示されたら、ご自身のHugging Faceユーザー名（または組織名）とリポジトリ名（例: `"YOUR_HF_USERNAME/your-lora-adapter"`）を入力してください。入力がない場合は、`params.yaml`に設定されているデフォルト値が使用されます。

#### ステップ 3: データ準備
-   **目的**: `data/dataset.jsonl` からデータセットをロードし、学習に適した形式に変換します。
-   **アクション**: 関連セルを実行します。

#### ステップ 4 & 5: モデルとトークナイザーのロード
-   **目的**: ベースとなるLlamaモデルと対応するトークナイザーをロードし、LoRAファインチューニングのための設定を行います。
-   **アクション**: 関連セルを実行します。Unslothが自動的に4bit量子化されたモデルをロードし、LoRAアダプタを準備します。

#### ステップ 6: トレーニング
-   **目的**: 設定されたハイパーパラメータに基づいてモデルのファインチューニングを開始します。
-   **アクション**: トレーニングセルを実行します。これには時間がかかります（ColabのT4 GPUで数分から数十分）。

#### ステップ 7: LoRAアダプタの保存
-   **目的**: ファインチューニングが完了した後、学習済みのLoRAアダプタをローカルに保存します。
-   **アクション**: 関連セルを実行します。

#### ステップ 8: Hugging Face Hubへのアップロード
-   **目的**: 保存したLoRAアダプタをHugging Face Hubにプッシュします。
-   **アクション**: 関連セルを実行します。
    *   **Hugging Face認証**: `notebook_login()`のプロンプトが表示されたら、ステップ4.2で取得したHugging Faceアクセストークン（PATとは別です。これはHugging Face Hubへのログイン用トークンです）を入力してください。
    *   これにより、ステップ2で入力したリポジトリ名にアダプタがアップロードされます。

#### ステップ 9: 推論の確認 (Colab内)
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

4.  「Create Space」をクリックします。
5.  Hugging Face Spacesが自動的にGitHubリポジトリからコードをクローンし、`inference/requirements.txt` に基づいて依存ライブラリをインストールし、`inference/app.py` を実行してアプリケーションを起動します。

    💡 **ヒント**: デプロイのログはSpaceの「Logs」タブで確認できます。エラーが発生した場合はここでデバッグしてください。

### 3. 推論アプリの使用

デプロイが成功すると、Hugging Face SpacesのページにGradioのチャットUIが表示されます。

1.  **LoRAアダプタの選択**: UI上部のプルダウンメニューから、Hugging Face HubにアップロードしたLoRAアダプタを選択します。これにより、選択したアダプタが動的にロードされます。
    *   初回ロード時は、モデルとアダプタのダウンロードに時間がかかる場合があります。
2.  **チャットの開始**: テキストボックスに質問を入力し、「Send」ボタンをクリックして、ファインチューニングされたモデルとの会話を開始します。

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

-   **GitHubクローンエラー**:
    *   GitHub Personal Access Token (PAT) が正しいか、有効期限が切れていないか、`repo`スコープが付与されているかを確認してください。
    *   Colabランタイムを再起動してみてください。
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
