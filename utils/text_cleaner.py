import os
import logging
import hojichar
from hojichar.filters.document_filters import DocumentNormalizer
from hojichar.core.parallel import Parallel

# It's good practice for utility modules to have their own logger
# or allow a logger to be passed in. For simplicity, using a local logger.
logger = logging.getLogger(__name__)

def clean_texts(texts: list[str], num_cores: int | None = None) -> list[str]:
    """
    Cleans a list of text strings using HojiChar.

    Args:
        texts: A list of raw text strings.
        num_cores: The number of CPU cores to use for parallel processing.
                   Defaults to os.cpu_count().

    Returns:
        A list of cleaned text strings.
    """
    if not texts:
        logger.info("No texts provided to clean_texts function.")
        return []

    logger.info(f"Starting text cleaning with HojiChar for {len(texts)} texts.")

    cleaner = hojichar.Compose([
        DocumentNormalizer(),
        # Add other text-based filters here if they were intended
        # e.g., hojichar.filters.text_cleaning.CleanBadWords()
    ])

    input_documents = [hojichar.Document(text) for text in texts]

    if num_cores is None:
        num_cores = os.cpu_count() or 1  # Default to 1 core if os.cpu_count() is None

    logger.info(f"Using {num_cores} CPU cores for HojiChar processing.")

    cleaned_texts_content = []
    # Initialize Parallel with the cleaner and number of jobs
    with Parallel(cleaner, num_jobs=num_cores) as pfilter:
        # pfilter.imap_apply returns an iterator of processed Document objects
        for i, doc in enumerate(pfilter.imap_apply(input_documents)):
            cleaned_texts_content.append(doc.text)
            if (i + 1) % 100 == 0: # Log progress every 100 texts
                logger.info(f"HojiChar processed {i+1}/{len(texts)} texts in clean_texts...")

    logger.info(f"HojiChar cleaning complete in clean_texts. Processed {len(cleaned_texts_content)} texts.")
    return cleaned_texts_content

if __name__ == '__main__':
    # Example usage:
    # This part is for testing the module directly.
    # It requires HojiChar to be installed.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sample_texts = [
        "これは最初のサンプルテキストです。",
        "これは　２番目の　テキストです。　スペースが多いです。",
        "This is the third text with English words.",
        "重複重複重複テキスト", # Example for potential deduplication filters if added
        "髙島屋と髙島屋" # Example for normalization
    ]

    logger.info(f"Original texts: {sample_texts}")

    # Using default number of cores
    cleaned = clean_texts(sample_texts)

    logger.info(f"Cleaned texts: {cleaned}")

    # Example: Test with a specific number of cores
    # cleaned_single_core = clean_texts(sample_texts, num_cores=1)
    # logger.info(f"Cleaned texts (1 core): {cleaned_single_core}")

    assert len(cleaned) == len(sample_texts)
    # Basic check: Normalizer should handle full-width spaces, etc.
    # The exact output depends on DocumentNormalizer's behavior.
    # For "これは　２番目の　テキストです。　スペースが多いです。"
    # it might become "これは 2番目の テキストです。 スペースが多いです。"
    # or similar depending on the NFKC normalization.
    if len(sample_texts) > 1:
         # Assuming DocumentNormalizer changes full-width spaces to half-width
        expected_second_text_part = "2番目の" # Normalizer changes full-width numbers
        if expected_second_text_part not in cleaned[1]:
             logger.warning(f"Normalization of the second text might not be as expected: '{cleaned[1]}'")
        # Check for Takashimaya normalization (髙 -> 高)
        if "高島屋" not in cleaned[4]:
            logger.warning(f"Normalization of '髙島屋' might not be as expected: '{cleaned[4]}'")

    logger.info("Test run of text_cleaner.py complete.")
