# テキスト処理パイプライン: FineWeb サンプル

このプロジェクトは、[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) データセットのサンプルテキストを処理するPythonパイプラインです。主な処理は、HojiCharによるテキストのクリーニング（フィルタリング）、事前学習済みTransformerモデルによる教育コンテンツとしての品質スコアリング、そしてGemini APIを利用した指示（テキスト）の難易度評価、品質評価、カテゴリ分類です。

## 主な機能

- **多段階処理**: データの読み込み、フィルタリング、スコアリング、Gemini APIによる評価を段階的に実行します。
- **テキストフィルタリング**:
    - `HojiChar`ライブラリを利用し、テキストの正規化や不要な文字の除去を行います。
    - CPUのマルチコアを活用した並列処理により、大量のデータを高速に処理します。
- **AIによるテキストスコアリング**:
    - Hugging Faceの`hotchpotch/fineweb-2-edu-japanese-classifier`モデルを使用し、テキストが教育的な内容かを判定・スコアリングします。
    - `transformers`と`torch`を基盤とし、利用可能な場合はGPU（CUDA）を自動で検知・使用します（GPUがない場合はCPUにフォールバック）。
- **Gemini APIによる評価**:
    - 各テキスト（指示）に対して、Gemini API（`gemini-1.5-flash-latest`モデル）を使用して以下の評価を行います。
        - **難易度評価**: 指示の意図、解決に必要な知識、および難易度（very easy から very hard）。
        - **品質評価**: 指示の明確さ、具体性、一貫性に基づいた品質（very poor から excellent）。
        - **カテゴリ分類**: 指示内容に最も適した主要タスクタグと、関連するその他のタスクタグ。
    - これらの評価は構造化出力（JSON）を利用して取得されます。
- **堅牢な設計**:
    - 処理の進捗やエラーをコンソールとログファイル（`pipeline.log`）に出力します。
    - ファイルI/OやAPI呼び出し時のエラーを適切にハンドリングし、予期せぬ中断を防ぎます。

## パイプラインの処理フロー

メインスクリプト `main.py` は、以下の順序で処理を調整します。

1.  **データ読み込み**
    - 入力: `data/fineweb_100_samples.json` (注: READMEの古いバージョンでは `docs/` ディレクトリでしたが、コード内では `data/` が使われています。こちらに合わせました。)
    - ファイルからテキストデータを抽出します。
    ↓
2.  **フィルタリング (CPU)**
    - HojiCharを使い、テキストをクリーニングします。
    - 中間出力: `filtered_texts.jsonl`
    ↓
3.  **スコアリング (GPU/CPU)**
    - フィルタリング済みのテキストをローカルAIモデルで教育的品質を評価します。
    ↓
4.  **Gemini評価 (API)**
    - スコアリング済みの各テキストに対し、Gemini APIを呼び出して以下を取得します。
        - 難易度評価 (`difficulty_rating`)
        - 品質評価 (`quality_rating`)
        - カテゴリ分類 (`classification`)
    ↓
5.  **最終出力**
    - 出力: `final_processed_texts.jsonl`
    - 元のテキスト、教育的判定、スコア、およびGeminiによる各評価結果をJSONL形式で保存します。

## 依存関係

パイプラインの各ステップを実行するには、`src/text-filter-pipeline/requirements.txt` ファイルに記載されたライブラリが必要です。主なものとして以下が含まれます。

- `hojichar`: テキストフィルタリング用
- `torch`, `transformers`: ローカルAIモデルによるスコアリング用
- `google-generativeai`: Gemini APIアクセス用
- `psutil`: システムユーティリティ（テキストプロセッサ内で利用）

インストールは以下のコマンドで行います。
```bash
pip install -r src/text-filter-pipeline/requirements.txt
```

**注記**:
- PyTorch (`torch`) は、特にGPU（CUDA）サポート付きの場合、ファイルサイズが非常に大きくなることがあります。CPU専用でインストールする場合は、[公式PyTorchインストールガイド](https://pytorch.org/get-started/locally/) を参照してください。
- Gemini APIを利用するステップでは、`GEMINI_API_KEY` 環境変数の設定が必要です。このキーは Google AI Studio などで取得できます。APIキーが設定されていない場合、Gemini評価ステップはエラーメッセージを出力してスキップされます（または結果にエラーが記録されます）。

## 実行方法

1.  **リポジトリのクローンと移動**:
    ```bash
    # git clone <repository_url>
    # cd <repository_name>
    ```
    (上記は一般的なコマンドです。実際のプロジェクトに合わせてください。)
    このREADMEがある `src/text-filter-pipeline` ディレクトリを基準に説明します。

2.  **依存関係のインストール**:
    プロジェクトのルートディレクトリ（`text-filter-pipeline`の親など）または`src/text-filter-pipeline`内で実行します。
    ```bash
    pip install -r src/text-filter-pipeline/requirements.txt
    ```

3.  **データファイルの準備**: スクリプトは `src/text-filter-pipeline/data/fineweb_100_samples.json` を使用します。このファイルが存在しない場合は、提供されている `src/text-filter-pipeline/サンプルで100件取得する.py` スクリプト（別途 `datasets` と `pandas` ライブラリが必要な場合があります）を実行するか、手動で配置してください。
    ```bash
    # (もしサンプル生成スクリプトを実行する場合)
    # python src/text-filter-pipeline/サンプルで100件取得する.py
    ```
    このサンプル生成スクリプトは、データファイルを `src/text-filter-pipeline/data/` ディレクトリに保存するように調整されているか確認してください。

4.  **GEMINI_API_KEYの設定**: Gemini APIを利用するために、APIキーを環境変数として設定します。
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
    (Windowsの場合は `set GEMINI_API_KEY=YOUR_API_KEY_HERE` または `setx GEMINI_API_KEY "YOUR_API_KEY_HERE"`)

5.  **メインパイプラインの実行**: `src/text-filter-pipeline` ディレクトリにいる場合:
    ```bash
    python main.py
    ```
    プロジェクトのルートディレクトリから実行する場合:
    ```bash
    python src/text-filter-pipeline/main.py
    ```
    スクリプトは各ステップの進捗をコンソールに出力します。スコアリングやGemini評価に必要なライブラリやAPIキーが不足している場合、該当ステップはスキップされるかエラーがログに出力されます。

## 生成されるファイル

このプロジェクトでは、実行するスクリプトに応じて以下のファイルが `src/text-filter-pipeline/` ディレクトリ（またはスクリプト実行場所からの相対パスで `main.py` が生成する場所）に生成されます。

- **`data/fineweb_100_samples.json`**:
  - パイプラインの**入力データ**です。`main.py` はこのパスのファイルを期待します。
  - `サンプルで100件取得する.py` を実行することで `src/text-filter-pipeline/data/` 配下に生成されることを想定。

- **`pipeline.log`**:
  - `main.py` の実行ログ。デバッグや処理の追跡に役立ちます。デフォルトでは `main.py` と同じディレクトリに生成されます。

- **`filtered_texts.jsonl`**:
  - `main.py` によって生成される**中間ファイル**。HojiCharでクリーニングされたテキストデータが含まれます（JSONL形式）。デフォルトでは `main.py` と同じディレクトリに生成されます。

- **`final_processed_texts.jsonl`**:
  - `main.py` によって生成される**最終出力ファイル**。各テキストに対して、ローカルモデルによるスコアリング結果 (`is_educational`, `score`) と、Gemini APIによる評価結果 (`difficulty_rating`, `quality_rating`, `classification`) が含まれます（JSONL形式）。デフォルトでは `main.py` と同じディレクトリに生成されます。
  - 例: `{"text": "...", "is_educational": true, "score": 0.95, "difficulty_rating": {"intent": "...", "knowledge": "...", "difficulty": "easy"}, "quality_rating": {"explanation": "...", "quality": "good"}, "classification": {"primary_tag": "Information seeking", "other_tags": []}}`
```
