# テキスト処理パイプライン: FineWeb サンプル

このプロジェクトは、FineWeb データセットのテキストサンプルを処理するパイプラインを実装します。データの読み込み、HojiChar を使用したフィルタリング、事前学習済みトランスフォーマーモデルを使用した教育コンテンツのスコアリングが含まれます。

## パイプライン概要

メインスクリプト `main.py` は、以下のステップを調整します。

1.  **データ読み込み**: テキストは `docs/fineweb_100_samples.json` から読み込まれます。このファイルには100件のサンプルが含まれており、各サンプルには "text" フィールドがあります。
    *   データファイル `docs/fineweb_100_samples.json` は事前に提供されています。また、`python サンプルで100件取得する.py` を実行することで（再）生成することもできます（`datasets` ライブラリが必要です）。
2.  **テキストフィルタリング (CPU マルチコア)**: 読み込まれたテキストは HojiChar (`hojichar==0.9.0`) を使用して処理されます。このステップではテキストが正規化されます。
    *   出力: `filtered_texts.jsonl` (各行はJSONオブジェクト: `{"text": "filtered_text_here"}`)
3.  **テキストスコアリング (GPU/CPU)**: フィルタリングされたテキストは、`hotchpotch/fineweb-2-edu-japanese-classifier` モデルを使用して教育的品質がスコアリングされます。`gpu_scorer.py` スクリプトが分類ロジックをカプセル化しています。
    *   このステップでは `transformers` と `torch` を使用します。利用可能で `torch` がCUDAサポート付きでインストールされていればGPUを使用しようとします。それ以外の場合はCPUにフォールバックします。
    *   出力: `scored_texts.jsonl` (各行はJSONオブジェクト: `{"text": "original_filtered_text", "is_educational": true/false, "score": float_score}`)

## 依存関係

完全なパイプラインを実行するには、以下のPythonライブラリが必要です。

*   `hojichar==0.9.0`: テキストフィルタリング用。
    ```bash
    pip install hojichar==0.9.0
    ```
*   `datasets`: `サンプルで100件取得する.py` でデータを取得するために使用され、また `gpu_scorer.py` でも使用されます（テスト用に `__main__` ブロックを実行する場合）。
    ```bash
    pip install datasets
    ```
*   `pandas`: `gpu_scorer.py` （テスト用に `__main__` ブロックを実行する場合）および `サンプルで100件取得する.py` で使用されます。
    ```bash
    pip install pandas
    ```
*   スコアリングステップ（ステップ3）には、以下が必要です。
    *   `torch`: コア機械学習ライブラリ。
    *   `transformers`: 事前学習済みモデルの読み込みと使用のため。
    *   `scikit-learn`: `gpu_scorer.py` でテストブロック内のメトリクス計算に使用されます。
    *   `sentencepiece`: トランスフォーマーモデルでよく使用されるトークナイザー。
    *   `accelerate`: PyTorch用ユーティリティ。
    *   `protobuf`: 一部のトランスフォーマーモデルの依存関係。
    ```bash
    pip install torch transformers scikit-learn sentencepiece accelerate protobuf
    ```
    **`torch` のインストールに関する注意**: PyTorch、特にGPU使用のためのCUDAサポート付きのものは非常に大きくなる可能性があります。ディスク容量の問題でインストールに問題が発生した場合や、CPU専用バージョン（スコアリングが遅くなります）が必要な場合は、[公式PyTorchインストールガイド](https://pytorch.org/get-started/locally/) を参照して、代替インストールコマンド（例: CPU専用バージョンの場合は `--index-url https://download.pytorch.org/whl/cpu` を指定）を確認してください。

## パイプラインの実行

1.  **依存関係がインストールされていることを確認**: 上記の「依存関係」セクションを参照してください。フィルタリング部分では、`main.py` には `hojichar` のみが必要です。データ生成には `datasets` と `pandas`。スコアリングには `torch`、`transformers` などが必要です。
2.  **データファイルが存在することを確認**: スクリプトは `docs/fineweb_100_samples.json` を想定しています。見つからない場合は、以下を実行して生成できます。
    ```bash
    python サンプルで100件取得する.py
    ```
3.  **メインスクリプトの実行**:
    ```bash
    python main.py
    ```
    スクリプトは各ステップの進捗メッセージを出力します。`torch` または `transformers` がインストールされていない場合、スコアリングステップはメッセージと共にスキップされます。

## 出力ファイル

*   `filtered_texts.jsonl`: HojiChar処理後のテキストが含まれます。
*   `scored_texts.jsonl`: 元のフィルタリングされたテキストと、それらの教育スコア、および教育的基準を満たすかどうかを示すブールフラグが含まれます。（このファイルは、スコアリングステップが正常に実行された場合にのみ生成されます）。

## 元のタスク説明（以前のREADME構造より）

最初の要求には、3段階のパイプラインが含まれていました。

1.  `docs/fineweb_100_samples.json` からデータを読み込む（`text` フィールドのみ使用）。
2.  CPUマルチコア処理でHojiCharを使用してテキストをフィルタリングし、保存する。
3.  `hotchpotch/fineweb-2-edu-japanese-classifier` (GPU) を使用して、フィルタリングされたテキストに品質スコアを割り当てる。

この機能は現在 `main.py` と `gpu_scorer.py` に実装されています。
