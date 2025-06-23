import json
import os
import logging
import importlib

# Attempt to import torch for device checks and dtype.
try:
    import torch
except ImportError:
    torch = None # Will be checked before use

# Attempt to import Fineweb2EduJapaneseScoreClassifier
# This allows the module to be imported even if gpu_scorer or its deps are missing,
# and the error can be handled gracefully by the calling code.
try:
    from gpu_scorer import Fineweb2EduJapaneseScoreClassifier
except ImportError as e:
    Fineweb2EduJapaneseScoreClassifier = None
    _Fineweb2EduJapaneseScoreClassifier_import_error = e

logger = logging.getLogger(__name__)

def score_texts(
    texts_to_score: list[str],
    model_name: str = "hotchpotch/fineweb-2-edu-japanese-classifier",
    device: str | None = None, # Auto-detect if None
    batch_size: int | None = None, # Auto-set based on device if None
    num_workers: int | None = None, # Auto-set based on device if None
    dtype: str | None = None # Auto-set based on device if None, e.g. "bfloat16", "float16", "float32"
) -> list[tuple[bool, float]] | None:
    """
    Scores a list of text strings using Fineweb2EduJapaneseScoreClassifier.

    Args:
        texts_to_score: A list of cleaned text strings.
        model_name: The name or path of the classifier model.
        device: The device to use ('cuda', 'cpu'). Auto-detects if None.
        batch_size: Batch size for scoring. Auto-sets if None.
        num_workers: Number of workers for data loading. Auto-sets if None.
        dtype: The torch dtype to use ('bfloat16', 'float16', 'float32'). Auto-sets if None.

    Returns:
        A list of tuples, where each tuple is (is_educational: bool, score: float),
        or None if scoring could not be performed (e.g., classifier not available).
    """
    if Fineweb2EduJapaneseScoreClassifier is None:
        logger.error(f"Fineweb2EduJapaneseScoreClassifier could not be imported: {_Fineweb2EduJapaneseScoreClassifier_import_error}")
        logger.error("Please ensure 'gpu_scorer.py' is available and its dependencies (transformers, torch) are installed.")
        return None

    if not texts_to_score:
        logger.info("No texts provided to score_texts function.")
        return []

    logger.info(f"Preparing to score {len(texts_to_score)} texts.")

    if torch is None:
        logger.error("PyTorch is not installed, which is required for the scorer. Please install PyTorch.")
        return None

    # Determine device
    if device is None:
        effective_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        effective_device = device
    logger.info(f"Using device: {effective_device}")

    # Determine batch size and num_workers based on device if not specified
    if batch_size is None:
        effective_batch_size = 32 if effective_device == "cuda" else 8
    else:
        effective_batch_size = batch_size

    if num_workers is None:
        effective_num_workers = 4 if effective_device == "cuda" else 1
    else:
        effective_num_workers = num_workers
    logger.info(f"Using batch_size: {effective_batch_size}, num_workers: {effective_num_workers}")

    # Determine dtype
    torch_dtype = None
    if dtype is None: # Auto-detection
        if effective_device == "cuda":
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
                logger.info("Using bfloat16 for GPU (auto-detected).")
            else:
                torch_dtype = torch.float16 # Fallback to float16 if bfloat16 not supported
                logger.info("bfloat16 not supported on this GPU, using float16 for GPU (auto-detected).")
        else: # CPU
            torch_dtype = torch.float32
            logger.info("Using float32 for CPU (auto-detected).")
    else: # User specified
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            logger.warning(f"Unsupported dtype '{dtype}' specified. Defaulting to auto-detection.")
            # Re-run auto-detection logic for specified dtype
            if effective_device == "cuda":
                if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
            else: # CPU
                torch_dtype = torch.float32
        logger.info(f"Using specified or auto-defaulted dtype: {torch_dtype}")


    try:
        classifier = Fineweb2EduJapaneseScoreClassifier(
            model_path=model_name,
            device=effective_device,
            batch_size=effective_batch_size,
            num_workers=effective_num_workers,
            show_progress=True, # Enable progress bar
            dtype=torch_dtype
        )
        logger.info(f"Classifier '{model_name}' initialized successfully on {effective_device}.")

        logger.info(f"Starting scoring for {len(texts_to_score)} texts...")
        scored_results = classifier.predict(texts_to_score) # List of (is_edu, score)
        logger.info(f"Scoring complete. Received {len(scored_results)} results.")
        return scored_results

    except NameError as ne:
         if 'torch' in str(ne).lower() or 'transformers' in str(ne).lower():
            logger.error(f"Skipping scoring: Required library not found ({ne}). Please install torch and transformers.")
         else:
            logger.error(f"An unexpected NameError occurred during scoring setup: {ne}")
         return None
    except RuntimeError as re:
        logger.error(f"RuntimeError during scoring: {re}. This might be due to insufficient GPU memory or other runtime issues. Try reducing batch_size or checking CUDA setup.")
        logger.error("Skipping scoring due to RuntimeError.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during scoring: {e}")
        logger.error("Skipping scoring due to unexpected error.")
        return None

if __name__ == '__main__':
    # Example usage:
    # This part is for testing the module directly.
    # It requires gpu_scorer.py and its dependencies (torch, transformers) to be installed.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Mock gpu_scorer.py if it's not available for a simple test run of the structure
    # For a real test, you'd need it.
    if Fineweb2EduJapaneseScoreClassifier is None:
        logger.warning("Fineweb2EduJapaneseScoreClassifier not available. Mocking for test.")

        class MockClassifier:
            def __init__(self, model_path, device, batch_size, num_workers, show_progress, dtype):
                logger.info(f"MockClassifier initialized with: model_path={model_path}, device={device}, batch_size={batch_size}, num_workers={num_workers}, show_progress={show_progress}, dtype={dtype}")

            def predict(self, texts: list[str]) -> list[tuple[bool, float]]:
                logger.info(f"MockClassifier predicting for {len(texts)} texts.")
                # Return some dummy data based on text length or content
                results = []
                for i, text in enumerate(texts):
                    is_edu = "educational" in text.lower() or len(text) > 50
                    score = float(len(text) % 100) / 100.0
                    results.append((is_edu, score + (0.1 if is_edu else 0.0)))
                    if i < 2: # Log first few mock predictions
                        logger.info(f"Mock prediction for '{text[:30]}...': is_edu={is_edu}, score={results[-1][1]:.2f}")
                return results

        # Replace the actual class with the mock for the test
        Fineweb2EduJapaneseScoreClassifier = MockClassifier
        _Fineweb2EduJapaneseScoreClassifier_import_error = None # Clear error for mock

    sample_texts_for_scoring = [
        "これは教育的な内容を含む可能性のある長いテキストです。多くの情報を提供します。",
        "短いテキスト。",
        "Another example of an educational document with specific keywords.",
        "これはテストです。"
    ]

    logger.info(f"Texts to score: {sample_texts_for_scoring}")

    # Test with auto-detection for device, batch_size, etc.
    results = score_texts(sample_texts_for_scoring)

    if results:
        logger.info(f"Scored results (auto-params): {results}")
        for text, (is_edu, score) in zip(sample_texts_for_scoring, results):
            logger.info(f"Text: '{text[:30]}...' -> Is Educational: {is_edu}, Score: {score:.4f}")
        assert len(results) == len(sample_texts_for_scoring)
    else:
        logger.error("Scoring returned no results or failed.")

    # Example: Test forcing CPU and specific batch size (if classifier is available)
    # if Fineweb2EduJapaneseScoreClassifier is not None and not isinstance(Fineweb2EduJapaneseScoreClassifier, MockClassifier): # only if real one
    #     logger.info("Testing with forced CPU and specific batch size...")
    #     results_cpu = score_texts(
    #         sample_texts_for_scoring,
    #         device="cpu",
    #         batch_size=2,
    #         num_workers=1,
    #         dtype="float32"
    #     )
    #     if results_cpu:
    #         logger.info(f"Scored results (CPU, batch_size=2): {results_cpu}")
    #     else:
    #         logger.error("Scoring on CPU returned no results or failed.")

    logger.info("Test run of text_processor.py complete.")
