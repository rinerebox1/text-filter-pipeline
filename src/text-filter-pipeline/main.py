import json
import os
import logging
# import hojichar # No longer directly used in main for Compose
# from hojichar.filters.document_filters import JSONLoader, DocumentNormalizer, JSONDumper # Moved to cleaner
# from hojichar.core.parallel import Parallel # Moved to cleaner

# Import the new utility functions
from utils.text_cleaner import clean_texts
from utils.text_processor import score_texts

# Attempt to import torch for initial device check for logging, if desired.
# Actual torch dependency is handled within text_processor.
try:
    import torch
except ImportError:
    torch = None

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

    # Step 2: Filter text using the utility function from text_cleaner
    logger.info("Filtering texts using text_cleaner.clean_texts...")
    num_cores_for_cleaning = os.cpu_count() or 1
    # Note: clean_texts will log its own progress internally if its logger is configured.
    # We pass the main logger's level for consistency if desired, or let it use its own.
    # For now, clean_texts has its own logger.
    filtered_texts_content = clean_texts(texts, num_cores=num_cores_for_cleaning)

    if not filtered_texts_content:
        logger.warning("Text cleaning returned no content. Proceeding might lead to errors or empty results.")
        # Decide if to return or continue based on requirements. For now, continue.
    else:
        logger.info(f"Text cleaning complete. Processed {len(filtered_texts_content)} texts via utility.")

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

    # Step 3: Score filtered texts using the utility function from text_processor
    logger.info("Preparing to score filtered texts using text_processor.score_texts...")

    # Load filtered texts from the JSONL file to pass to the scorer utility
    # The utility itself doesn't handle file I/O for input texts to keep it focused.
    texts_to_score_from_file = []
    if os.path.exists(filtered_output_file):
        logger.info(f"Loading filtered texts from {filtered_output_file} for scoring...")
        try:
            with open(filtered_output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        texts_to_score_from_file.append(json.loads(line)['text'])
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line {line_num} in {filtered_output_file}: {line.strip()}")
                    except KeyError:
                        logger.warning(f"Skipping line {line_num} in {filtered_output_file} due to missing 'text' key: {line.strip()}")
        except IOError as e:
            logger.error(f"Could not read filtered texts from {filtered_output_file}: {e}")
            # texts_to_score_from_file remains empty

    if not texts_to_score_from_file:
        logger.warning(f"No texts loaded from {filtered_output_file} to score. Skipping scoring step.")
    else:
        logger.info(f"Loaded {len(texts_to_score_from_file)} texts for scoring from {filtered_output_file}.")

        # Parameters for the scorer can be defined here or loaded from config
        # The text_processor.score_texts function will handle auto-detection if some are None.
        # For example, device, batch_size, num_workers, dtype can be None for auto-config.
        scored_results = score_texts(
            texts_to_score_from_file
            # model_name="hotchpotch/fineweb-2-edu-japanese-classifier", # Default in function
            # device=None, # Auto-detect
            # batch_size=None, # Auto-set
            # num_workers=None, # Auto-set
            # dtype=None # Auto-set
        )

        if scored_results is not None:
            logger.info(f"Scoring complete via utility. Received {len(scored_results)} results.")
            scored_output_file = "scored_texts.jsonl"
            try:
                with open(scored_output_file, 'w', encoding='utf-8') as f:
                    for i, text_content in enumerate(texts_to_score_from_file): # Use the original text for output
                        if i < len(scored_results):
                            is_educational, score_value = scored_results[i]
                            f.write(json.dumps({"text": text_content, "is_educational": is_educational, "score": score_value}) + "\n")
                        else:
                            logger.warning(f"Mismatch between number of texts to score and results at index {i}. Skipping writing this result.")
                logger.info(f"Scored texts saved to {scored_output_file}")
                if scored_results:
                    logger.debug(f"First scored text sample (from main): Score={scored_results[0][1]:.2f}, Educational={scored_results[0][0]}, Text='{texts_to_score_from_file[0][:100]}...'")
            except IOError as e:
                logger.error(f"Could not write scored texts to {scored_output_file}: {e}")
        else:
            logger.info("Scoring step was skipped or failed (e.g., classifier not available or error during processing). Check logs from text_processor.")

    logger.info("Text processing pipeline finished.")

if __name__ == "__main__":
    # Initial torch import for CUDA check is still fine here, or can be removed
    # as text_processor also handles torch import.
    # If torch is imported here, it's available for the device check in main's scope if any.
    # For this refactoring, it's less critical in main.py directly.
    main()
