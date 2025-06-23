```
# テキスト処理パイプラインの概要

このPythonスクリプトは、JSON形式のテキストデータを入力とし、**クリーニング（フィルタリング）**と**AIモデルによるスコアリング**を段階的に実行するデータ処理パイプラインです。

## 主な機能と特徴

- **データソース**: `JSON`配列形式のファイルからテキストデータを読み込みます。
- **テキストフィルタリング**:
    - **`HojiChar`ライブラリ**を利用して、テキストの正規化や不要な文字の除去を行います。
    - CPUの全コアを活用した**並列処理**により、大量のデータを高速にクリーニングします。
- **テキストスコアリング**:
    - **`transformers`と`torch`**を基盤とした分類モデル（`hotchpotch/fineweb-2-edu-japanese-classifier`）を使用します。
    - テキストが**「教育的な内容か」**を判定し、スコアを付けます。
    - **GPU (CUDA)を自動で検知・利用**し、高速な推論を実現します（GPUがなければCPUにフォールバック）。
    - 依存ライブラリ（`torch`など）がインストールされていない場合、このステップは安全にスキップされます。
- **堅牢な設計**:
    - **ロギング機能**により、処理の進捗やエラーがコンソールとログファイル（`pipeline.log`）の両方に出力されます。
    - ファイルI/OエラーやGPUメモリ不足などの実行時エラーを適切に捕捉し、プログラムが意図せず停止するのを防ぎます。

## 処理フロー

このパイプラインは、以下の順序でデータを処理します。

1.  **入力**: `docs/fineweb_100_samples.json`
    - JSONファイルからテキストを抽出します。
    ↓
2.  **フィルタリング (HojiChar)**
    - 抽出されたテキストをクリーニングします。
    - 中間ファイルとして `filtered_texts.jsonl` を出力します。
    ↓
3.  **スコアリング (AIモデル)**
    - クリーニング後のテキスト（`filtered_texts.jsonl`から読み込み）をAIモデルで評価します。
    ↓
4.  **最終出力**: `scored_texts.jsonl`
    - 元のテキスト、教育的かどうかの判定（`is_educational`）、スコア（`score`）を含む結果を保存します。

## 出力ファイル

このスクリプトを実行すると、以下のファイルが生成されます。

- `pipeline.log`: スクリプトの実行ログ。デバッグや処理の追跡に役立ちます。
- `filtered_texts.jsonl`: HojiCharによってクリーニングされたテキストデータ。1行1JSONのJSONL形式です。
- `scored_texts.jsonl`: スコアリング結果を含む最終的なデータ。こちらもJSONL形式です。
```


import json
import os
import hojichar
import logging
from hojichar.filters.document_filters import JSONLoader, DocumentNormalizer, JSONDumper
from hojichar.core.parallel import Parallel # This was likely correct

def load_data(file_path):
    """Loads data from a JSON file and extracts the 'text' field."""
    texts = []
    # Ensure the input is treated as a JSON array of objects,
    # where each object might contain a "text" field.
    # The sample fineweb_100_samples.json is a JSON array.
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f) # Loads the entire JSON array
        for item in data:
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            # If the items are just strings in a list, adapt accordingly.
            # Based on problem description, it's List[Dict[str, any]]
    return texts

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("pipeline.log")
# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def main():
    logger.info("Starting the text processing pipeline.")
    # Step 1: Load data
    data_file = "docs/fineweb_100_samples.json"
    if not os.path.exists(data_file):
        logger.error(f"Data file not found at {data_file}")
        # Suggestion to run the generation script might be removed if data is always pre-supplied
        # logger.info("Please run 'python サンプルで100件取得する.py' to generate the data.")
        return

    logger.info(f"Loading data from {data_file}...")
    texts = load_data(data_file)
    if not texts:
        logger.error("No texts found in the data file.")
        return
    logger.info(f"Loaded {len(texts)} texts.")
    if texts:
        logger.debug(f"First text sample: '{texts[0][:100]}...'")

    # Step 2: Filter text using HojiChar
    logger.info("Filtering texts using HojiChar...")

    # HojiChar's Parallel processor expects an iterable of hojichar.Document objects.
    # We need to wrap our raw text strings into Document objects.
    # The DocumentNormalizer will then operate on doc.text.
    # We will then extract the processed text from the document.

    # Define the HojiChar pipeline for strings
    # DocumentNormalizer normalizes the text within a Document.
    # We don't need JSONLoader/Dumper here as we are dealing with a list of strings in memory.
    cleaner = hojichar.Compose([
        DocumentNormalizer(),
        # Add other text-based filters if needed, e.g., hojichar.filters.text_cleaning.CleanBadWords()
    ])

    input_documents = [hojichar.Document(text) for text in texts]

    num_cores = os.cpu_count() or 1 # Default to 1 core if os.cpu_count() is None
    logger.info(f"Using {num_cores} CPU cores for HojiChar processing.")

    filtered_texts_content = []
    with Parallel(cleaner, num_jobs=num_cores) as pfilter:
        # pfilter.imap_apply returns an iterator of processed Document objects
        for i, doc in enumerate(pfilter.imap_apply(input_documents)):
            filtered_texts_content.append(doc.text)
            if (i + 1) % 20 == 0: # Log progress every 20 texts
                logger.info(f"HojiChar processed {i+1}/{len(texts)} texts...")

    logger.info(f"HojiChar filtering complete. Processed {len(filtered_texts_content)} texts.")

    # Save filtered texts to a JSONL file
    filtered_output_file = "filtered_texts.jsonl"
    try:
        with open(filtered_output_file, 'w', encoding='utf-8') as f:
            for text in filtered_texts_content:
                f.write(json.dumps({"text": text}) + "\n")
        logger.info(f"Filtered texts saved to {filtered_output_file}")
    except IOError as e:
        logger.error(f"Could not write filtered texts to {filtered_output_file}: {e}")
        return # Stop if we can't save filtered texts

    if filtered_texts_content:
        logger.debug(f"First filtered text sample: '{filtered_texts_content[0][:100]}...'")

    # Step 3: Score filtered texts using GPU scorer
    logger.info("Preparing to score filtered texts...")

    Fineweb2EduJapaneseScoreClassifier = None # Define in outer scope
    try:
        from gpu_scorer import Fineweb2EduJapaneseScoreClassifier
        logger.info("Successfully imported Fineweb2EduJapaneseScoreClassifier from gpu_scorer.py.")
    except ImportError as e:
        logger.warning(f"Could not import Fineweb2EduJapaneseScoreClassifier from gpu_scorer.py: {e}")
        logger.warning("Please ensure transformers and torch are installed if you intend to run the scoring step.")
        # Fineweb2EduJapaneseScoreClassifier remains None

    if Fineweb2EduJapaneseScoreClassifier:
        # Load filtered texts from the JSONL file
        texts_to_score = []
        if os.path.exists(filtered_output_file):
            logger.info(f"Loading filtered texts from {filtered_output_file} for scoring...")
            try:
                with open(filtered_output_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            texts_to_score.append(json.loads(line)['text'])
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping malformed line {line_num} in {filtered_output_file}: {line.strip()}")
                        except KeyError:
                            logger.warning(f"Skipping line {line_num} in {filtered_output_file} due to missing 'text' key: {line.strip()}")
            except IOError as e:
                logger.error(f"Could not read filtered texts from {filtered_output_file}: {e}")
                texts_to_score = [] # Ensure it's empty if loading fails

        if not texts_to_score:
            logger.warning(f"No texts loaded from {filtered_output_file} to score. Skipping scoring step.")
        else:
            logger.info(f"Loaded {len(texts_to_score)} texts for scoring from {filtered_output_file}.")

            model_name = "hotchpotch/fineweb-2-edu-japanese-classifier"
            # Attempt to use CUDA, fall back to CPU if not available or specified otherwise.
            # The classifier class itself has fallback logic.
            device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
            # Allow user to force CPU for testing/compatibility
            # device_to_use = "cpu" # Uncomment to force CPU

            logger.info(f"Initializing classifier with model '{model_name}' on device: {device_to_use}...")

            # Default batch size and num_workers, adjust based on device
            batch_size = 32 if device_to_use == "cuda" else 8 # Smaller batch for CPU
            num_workers = 4 if device_to_use == "cuda" else 1  # Fewer workers for CPU

            logger.info(f"Using batch_size: {batch_size}, num_workers: {num_workers}")

            try:
                # Determine dtype based on device and availability
                if device_to_use == "cuda" and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                    dtype_to_use = torch.bfloat16
                    logger.info("Using bfloat16 for GPU.")
                elif device_to_use == "cuda":
                    dtype_to_use = torch.float16 # Fallback to float16 if bfloat16 not supported
                    logger.info("bfloat16 not supported on this GPU, using float16.")
                else: # CPU
                    dtype_to_use = torch.float32
                    logger.info("Using float32 for CPU.")

                classifier = Fineweb2EduJapaneseScoreClassifier(
                    model_path=model_name,
                    device=device_to_use,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    show_progress=True, # This will use tqdm for progress
                    dtype=dtype_to_use
                )
                logger.info("Classifier initialized successfully.")

                logger.info("Scoring texts...")
                scored_results = classifier.predict(texts_to_score) # List of (is_edu, score)
                logger.info(f"Scoring complete. Received {len(scored_results)} results.")

                scored_output_file = "scored_texts.jsonl"
                try:
                    with open(scored_output_file, 'w', encoding='utf-8') as f:
                        for i, text in enumerate(texts_to_score):
                            if i < len(scored_results): # Ensure we don't go out of bounds
                                is_educational, score = scored_results[i]
                                f.write(json.dumps({"text": text, "is_educational": is_educational, "score": score}) + "\n")
                            else:
                                logger.warning(f"Mismatch between number of texts to score and results at index {i}. Skipping writing this result.")
                    logger.info(f"Scored texts saved to {scored_output_file}")
                    if scored_results:
                        logger.debug(f"First scored text sample: Score={scored_results[0][1]:.2f}, Educational={scored_results[0][0]}, Text='{texts_to_score[0][:100]}...'")
                except IOError as e:
                    logger.error(f"Could not write scored texts to {scored_output_file}: {e}")


            except NameError as ne: # Catch if torch or transformers aren't installed (should be caught by Fineweb2EduJapaneseScoreClassifier import)
                 if 'torch' in str(ne).lower() or 'transformers' in str(ne).lower():
                    logger.error(f"Skipping scoring: Required library not found ({ne}). Please install torch and transformers.")
                 else:
                    logger.error(f"An unexpected NameError occurred during scoring: {ne}")
                    raise ne # Re-raise if it's another NameError
            except RuntimeError as re: # Catch common PyTorch runtime errors e.g. CUDA OOM
                logger.error(f"RuntimeError during scoring: {re}. This might be due to insufficient GPU memory. Try reducing batch_size.")
                logger.error("Skipping scoring step due to RuntimeError.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during scoring: {e}")
                logger.error("Skipping scoring step.")
    else:
        logger.info("Skipping scoring step as classifier could not be imported or initialized (likely missing dependencies like torch/transformers).")

    logger.info("Text processing pipeline finished.")

if __name__ == "__main__":
    # Import torch here for device check before main() is called, if needed for device_to_use logic outside main
    # For now, torch is checked within main() or when Fineweb2EduJapaneseScoreClassifier is imported/used.
    try:
        import torch # To check for CUDA availability early
    except ImportError:
        pass # Handled inside main
    main()
