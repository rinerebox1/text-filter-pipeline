import json
import os
import logging
# import hojichar # No longer directly used in main for Compose
# from hojichar.filters.document_filters import JSONLoader, DocumentNormalizer, JSONDumper # Moved to cleaner
# from hojichar.core.parallel import Parallel # Moved to cleaner

# Import the new utility functions
from utils.text_cleaner import clean_texts
from utils.text_processor import score_texts
from utils.gemini_utils import (
    get_difficulty_rating,
    get_quality_rating,
    get_instruction_classification
)

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
    data_file = "data/fineweb_100_samples.json"
    if not os.path.exists(data_file):
        logger.error(f"Data file not found at {data_file}")
        return

    logger.info(f"Loading data from {data_file}...")
    raw_texts_from_load = load_data(data_file) # Renamed to avoid confusion
    if not raw_texts_from_load:
        logger.error("No texts found in the data file.")
        return
    logger.info(f"Loaded {len(raw_texts_from_load)} texts.")
    if raw_texts_from_load:
        logger.debug(f"First text sample: '{raw_texts_from_load[0][:100]}...'")

    # Step 2: Filter text using the utility function from text_cleaner
    logger.info("Filtering texts using text_cleaner.clean_texts...")
    num_cores_for_cleaning = os.cpu_count() or 1
    filtered_texts_content = clean_texts(raw_texts_from_load, num_cores=num_cores_for_cleaning)

    if not filtered_texts_content:
        logger.warning("Text cleaning returned no content. Proceeding might lead to errors or empty results.")
    else:
        logger.info(f"Text cleaning complete. Processed {len(filtered_texts_content)} texts via utility.")

    # Save filtered texts to a JSONL file
    # This step remains, as scoring might still want to read from a file, or it's good for caching.
    filtered_output_file = "filtered_texts.jsonl"
    try:
        with open(filtered_output_file, 'w', encoding='utf-8') as f:
            for text_content_item in filtered_texts_content: # Iterate through list of strings
                f.write(json.dumps({"text": text_content_item}) + "\n")
        logger.info(f"Filtered texts saved to {filtered_output_file}")
    except IOError as e:
        logger.error(f"Could not write filtered texts to {filtered_output_file}: {e}")
        return

    if not filtered_texts_content: # If cleaning resulted in no texts, stop or handle.
        logger.warning("No texts available after cleaning. Skipping subsequent steps.")
        logger.info("Text processing pipeline finished.")
        return

    logger.debug(f"First filtered text sample: '{filtered_texts_content[0][:100]}...'")

    # Step 3: Score filtered texts
    # The `score_texts` utility takes a list of strings.
    # We can use `filtered_texts_content` directly if it's available in memory.
    # Or, if preferred, load from `filtered_output_file` as originally done.
    # Using `filtered_texts_content` directly avoids re-reading if the list is not excessively large.
    logger.info("Preparing to score filtered texts...")

    # texts_to_process will hold the text content for scoring and subsequent Gemini steps
    texts_to_process = filtered_texts_content # Use the in-memory list of cleaned texts

    scored_results = score_texts(texts_to_process)

    if scored_results is None:
        logger.error("Scoring step failed or returned no results. Cannot proceed with Gemini evaluations.")
        logger.info("Text processing pipeline finished.")
        return

    logger.info(f"Scoring complete. Received {len(scored_results)} results.")

    # Step 4: Perform Gemini evaluations (Difficulty, Quality, Classification)
    # and combine with scored results.

    # The GEMINI_API_KEY should be set in the environment for gemini_utils to work.
    # Check if it's set and log a warning if not, as gemini_utils also does.
    if not os.getenv("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY environment variable is not set. Gemini API calls will likely fail.")

    logger.info("Performing Gemini evaluations for each text...")

    final_results_list = []
    # texts_to_process contains the actual text strings.
    # scored_results contains tuples of (is_educational, score_value).
    # Ensure they are of the same length.
    if len(texts_to_process) != len(scored_results):
        logger.error(f"Mismatch in length between texts_to_process ({len(texts_to_process)}) and scored_results ({len(scored_results)}). Aborting Gemini processing.")
        # Optionally, decide how to handle this: skip Gemini, or stop. For now, stopping further processing.
        logger.info("Text processing pipeline finished due to data mismatch.")
        return

    for i, text_content in enumerate(texts_to_process):
        logger.info(f"Processing text {i+1}/{len(texts_to_process)} with Gemini...")
        # It's important that `text_content` here is the actual instruction/text, not just metadata.
        # Assuming `texts_to_process` holds these strings.

        difficulty_info = get_difficulty_rating(text_content)
        quality_info = get_quality_rating(text_content)
        classification_info = get_instruction_classification(text_content)

        is_educational, score_value = scored_results[i]

        # Construct the final dictionary for this text item
        # Use placeholder or error values if Gemini calls fail (return None)
        result_item = {
            "text": text_content,
            "is_educational": is_educational,
            "score": score_value,
            "difficulty_rating": difficulty_info if difficulty_info else {"error": "Failed to get difficulty"},
            "quality_rating": quality_info if quality_info else {"error": "Failed to get quality"},
            "classification": classification_info if classification_info else {"error": "Failed to get classification"}
        }
        final_results_list.append(result_item)

        if (i + 1) % 10 == 0: # Log progress every 10 texts
             logger.info(f"Gemini processing: {i+1}/{len(texts_to_process)} texts processed.")

    logger.info("Gemini evaluations complete.")

    # Step 5: Save final combined results
    final_output_file = "final_processed_texts.jsonl" # New output file name
    try:
        with open(final_output_file, 'w', encoding='utf-8') as f:
            for result_item in final_results_list:
                f.write(json.dumps(result_item) + "\n")
        logger.info(f"Final processed texts (including scores and Gemini evaluations) saved to {final_output_file}")
        if final_results_list:
            # Log details of the first processed item as a sample
            first_item_log = {k: v for k, v in final_results_list[0].items() if k != "text"}
            first_item_log["text_preview"] = final_results_list[0]["text"][:100] + "..."
            logger.debug(f"First final processed text sample (from main): {json.dumps(first_item_log)}")

    except IOError as e:
        logger.error(f"Could not write final processed texts to {final_output_file}: {e}")

    logger.info("Text processing pipeline finished.")

if __name__ == "__main__":
    main()
