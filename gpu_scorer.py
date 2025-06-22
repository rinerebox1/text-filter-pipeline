# GPU 対応、5段階スコア評価 (0〜4)
import unicodedata
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class JapaneseTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Unicode正規化
        normalized_text = unicodedata.normalize("NFKC", text)

        # トークン化
        encoding = self.tokenizer(
            normalized_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class Fineweb2EduJapaneseScoreClassifier:
    def __init__(
        self,
        model_path: str,
        threshold: float = 2.5,
        batch_size: int = 512,
        num_workers: int = 15,
        device: str = "cuda",
        show_progress: bool = True,
        max_length: int = 512,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.threshold = threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dtype = dtype
        self.show_progress = show_progress

        # Ensure device is appropriate
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            self.device = "cpu"
        elif device == "cuda" and torch.cuda.is_available():
             # Check if bfloat16 is supported on the CUDA device
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                print("bfloat16 is not supported on this CUDA device. Falling back to float32.")
                self.dtype = torch.float32
            self.device = "cuda"
        else:
            self.device = "cpu"
            if dtype == torch.bfloat16: # bfloat16 is primarily for GPUs/TPUs
                print("bfloat16 is selected with CPU. Consider float32 for CPU for better compatibility/performance.")
                # self.dtype = torch.float32 # Optionally force float32 on CPU

        self.max_length = max_length

        # トークナイザーの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Tokenizer pad_token not set. Using eos_token: {self.tokenizer.eos_token} as pad_token.")

        self.model = self._init_model(model_path)


    def _init_model(self, model_path: str):
        print(f"Initializing model on device: {self.device} with dtype: {self.dtype}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, # Use the passed model_path
            num_labels=1,
            torch_dtype=self.dtype,
        ).to(self.device)
        model.eval()
        return model

    def predict(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """
        Predict educational content scores for input texts
        Returns list of tuples (is_educational: bool, score: float)
        """
        # カスタムデータセットの作成
        dataset = JapaneseTextDataset(texts, self.tokenizer, self.max_length)

        # データローダーの設定
        # pin_memory should only be True if on CUDA
        pin_memory_setting = True if self.device == "cuda" else False
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=pin_memory_setting,
            # Make dataloader resilient to empty texts list
            drop_last=False # Important if len(texts) < batch_size
        )

        if not texts: # Handle empty input list
            return []

        model = self.model
        predictions = []
        # Ensure tqdm description is clear
        iterator = (
            tqdm(dataloader, desc=f"Predicting scores ({self.device})") if self.show_progress else dataloader
        )

        with torch.no_grad():
            for batch in iterator:
                batch_ids = batch["input_ids"].to(self.device)
                batch_mask = batch["attention_mask"].to(self.device)

                outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
                # Ensure logits are correctly handled even for single item batches
                logits = outputs.logits.squeeze(-1) # Squeeze the last dimension (num_labels=1)

                # Convert to float and move to CPU, then to numpy
                # Using .item() for single value tensors, .tolist() for multi-value
                if logits.numel() == 1:
                    processed_logits = [logits.item()]
                else:
                    processed_logits = logits.float().cpu().tolist()

                predictions.extend(processed_logits)

        if self.device == "cuda":
            torch.cuda.empty_cache()

        scores = np.array(predictions)
        # Ensure threshold comparison is robust
        return [(float(s) >= float(self.threshold), float(s)) for s in scores]


if __name__ == "__main__":
    import time
    # from typing import List, Tuple # Already imported at the top
    import datasets
    # import numpy as np # Already imported at the top
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    def compute_detailed_metrics(pred_scores, true_scores, threshold: float = 2.5):
        scores = np.array(pred_scores)

        # Multi-class classification (0-4)
        multi_preds = np.clip(np.round(scores), 0, 4).astype(int)

        # Binary classification
        binary_preds = (scores >= threshold).astype(int)

        # Generate random labels for demonstration (replace with actual labels)
        # Note: In real usage, you would pass actual labels instead
        # np.random.seed(42)  # For reproducible example
        multi_labels = np.array(true_scores) # Ensure true_scores are numpy array
        binary_labels = (multi_labels >= threshold).astype(int)

        # Multi-class metrics
        multi_report = classification_report(
            multi_labels, multi_preds, labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0
        )
        multi_cm = confusion_matrix(multi_labels, multi_preds, labels=[0, 1, 2, 3, 4])

        # Binary metrics
        binary_report = classification_report(
            binary_labels,
            binary_preds,
            labels=[0, 1],
            target_names=["それ以外", "教育的"],
            output_dict=True,
            zero_division=0
        )
        binary_cm = confusion_matrix(binary_labels, binary_preds, labels=[0,1])

        # Print multi-class results
        print("Multi-class Classification Report:")
        print("-" * 80)
        df_multi = pd.DataFrame(multi_report).transpose()
        print(df_multi.round(4))

        print("\nMulti-class Confusion Matrix:")
        print("-" * 80)
        print(
            pd.DataFrame(
                multi_cm,
                index=[f"Actual {i}" for i in range(5)],
                columns=[f"Pred {i}" for i in range(5)],
            )
        )

        # Print binary results
        print("\nBinary Classification Report (それ以外/教育的):")
        print("-" * 80)
        df_binary = pd.DataFrame(binary_report).transpose()
        print(df_binary.round(4))

        print("\nBinary Confusion Matrix:")
        print("-" * 80)
        print(
            pd.DataFrame(
                binary_cm,
                index=["Actual それ以外", "Actual 教育的"],
                columns=["Pred それ以外", "Pred 教育的"],
            )
        )

        # Print key binary metrics
        print("\nKey Binary Metrics:")
        print("-" * 80)
        # Ensure '教育的' key exists or handle missing key gracefully
        precision = binary_report.get("教育的", {}).get("precision", 0)
        recall = binary_report.get("教育的", {}).get("recall", 0)
        f1_score = binary_report.get("教育的", {}).get("f1-score", 0)
        accuracy = binary_report.get("accuracy", 0)

        metrics_df = pd.DataFrame(
            {
                "Metric": ["Precision", "Recall", "F1-score", "Accuracy"],
                "Value": [precision, recall, f1_score, accuracy],
            }
        )
        print(metrics_df.round(4))

        return {
            "multi_report": multi_report,
            "multi_confusion_matrix": multi_cm,
            "binary_report": binary_report,
            "binary_confusion_matrix": binary_cm,
        }

    def print_score_matrix(scores: List[float], num_bins: int = 10):
        """Print the distribution of scores in a text-based matrix"""
        if not scores: # Handle empty scores list
            print("No scores to display.")
            return

        clipped_scores = np.clip(scores, 0, 4)
        hist, bins = np.histogram(clipped_scores, bins=num_bins, range=(0, 4))

        # Handle cases where max_count could be 0 to avoid division by zero
        max_count = max(hist) if len(hist) > 0 and max(hist) > 0 else 1


        print("\nScore Distribution (★ = approximately X samples):") # Clarify star meaning later
        print("-" * 50)
        # Determine scaling factor for stars, ensuring it's at least 1
        # star_scale = max(1, max_count / 20) # Ensure stars are drawn even for small counts

        for i in range(len(hist)):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            count = hist[i]
            # stars = "★" * int(count / star_scale)
            stars = "★" * int(count / (max_count / 20)) if max_count > 0 else "" # More robust star calculation
            print(f"{bin_start:4.1f}-{bin_end:4.1f}: {stars} ({count:4d})")
        print("-" * 50)

    # Initialize classifier
    # Note: In a real application, model_path might come from config or args
    model_name = "hotchpotch/fineweb-2-edu-japanese-classifier"

    # Adjust num_workers based on system capabilities for testing
    # Forcing CPU for initial testing due to potential CUDA memory issues in sandboxed env.
    # User can change this to "cuda" if a GPU is available and configured.
    test_device = "cpu" # Or "cuda" if available
    print(f"Attempting to use device: {test_device}")

    classifier = Fineweb2EduJapaneseScoreClassifier(
        model_name,
        show_progress=True,
        num_workers=1, # Start with 1 worker for stability, can be increased
        device=test_device,
        # Using float32 for CPU for broader compatibility, bfloat16 is better on compatible GPUs
        dtype=torch.float32 if test_device == "cpu" else torch.bfloat16
    )

    # Load test dataset
    target = "test" # This is the 'test' split of the HF dataset
    print(f"Loading '{target}' split of 'hotchpotch/fineweb-2-edu-japanese-scores' dataset...")

    try:
        ds = datasets.load_dataset("hotchpotch/fineweb-2-edu-japanese-scores", split=target, trust_remote_code=True)
        test_texts = ds["text"]
        true_scores_for_eval = ds["score"] # These are the reference scores for evaluation
        total_chars = sum(len(text) for text in test_texts)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Using dummy data for demonstration.")
        test_texts = ["これは教育的な内容です。", "これはあまり教育的ではないかもしれません。", "猫がマットの上に座っていました。"] * 10
        true_scores_for_eval = [3.5, 1.5, 0.5] * 10 # Dummy scores for structure
        total_chars = sum(len(text) for text in test_texts)


    # Get predictions with timing
    print(f"Starting prediction for {len(test_texts)} texts...")
    start_time = time.time()
    predictions_output = classifier.predict(test_texts) # Returns list of (is_edu, score)
    elapsed_time = time.time() - start_time

    # Extract raw predicted scores for metrics calculation and distribution display
    predicted_raw_scores = [score for _, score in predictions_output]

    # Clip scores for certain displays/calculations if necessary, but use raw for metrics
    # clipped_pred_scores = np.clip(predicted_raw_scores, 0, 4)

    is_edu_count = sum(1 for is_edu, _ in predictions_output if is_edu)


    # Print statistics
    print("\nPrediction Results:")
    print(f"Total samples: {len(predictions_output)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    if len(predictions_output) > 0:
        print(f"Average time per text: {(elapsed_time * 1000 / len(predictions_output)):.2f} ms")
        print(f"Processing speed: {total_chars / elapsed_time:.1f} characters/second" if elapsed_time > 0 else "N/A")
        print(
            f"Educational content ratio (based on threshold {classifier.threshold}): {is_edu_count / len(predictions_output) * 100:.1f}% ({is_edu_count}/{len(predictions_output)})"
        )
    else:
        print("No predictions were made.")

    # Print score distribution using predicted raw scores
    print_score_matrix(predicted_raw_scores)

    # Compute detailed metrics using predicted raw scores and true scores from dataset
    if true_scores_for_eval:
        print("\nCalculating detailed metrics against true scores from dataset...")
        detailed_metrics = compute_detailed_metrics(predicted_raw_scores, true_scores_for_eval, classifier.threshold)
    else:
        print("\nSkipping detailed metrics as true scores are not available.")


    # Print sample predictions
    print("\nSample Predictions (from test run):")

    # Sort by score
    # Ensure samples are correctly formed for sorting
    samples_to_display = list(zip(test_texts, predictions_output)) # [(text, (is_edu, score)), ...]
    samples_to_display.sort(key=lambda x: x[1][1], reverse=True) # Sort by score

    print("\nHigh-scoring Examples (raw scores):")
    for text, (is_edu, score) in samples_to_display[:3]:
        print(
            f"Score {score:.2f} (Clipped: {min(max(score, 0), 4):.2f}, Educational: {is_edu}): {text[:200]}...\n"
        )

    print("\nLow-scoring Examples (raw scores):")
    for text, (is_edu, score) in samples_to_display[-3:]:
        print(
            f"Score {score:.2f} (Clipped: {min(max(score, 0), 4):.2f}, Educational: {is_edu}): {text[:200]}...\n"
        )

    print("GPU scorer script execution finished.")
