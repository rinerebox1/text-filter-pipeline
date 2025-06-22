import json
import os
import hojichar
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

def main():
    # Step 1: Load data
    data_file = "docs/fineweb_100_samples.json"
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        # Suggestion to run the generation script might be removed if data is always pre-supplied
        # print("Please run 'python サンプルで100件取得する.py' to generate the data.")
        return

    print(f"Loading data from {data_file}...")
    texts = load_data(data_file)
    if not texts:
        print("No texts found in the data file.")
        return
    print(f"Loaded {len(texts)} texts.")
    if texts:
        print(f"First text sample: '{texts[0][:100]}...'")

    # Step 2: Filter text using HojiChar
    print("Filtering texts using HojiChar...")

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
    print(f"Using {num_cores} CPU cores for HojiChar processing.")

    filtered_texts_content = []
    with Parallel(cleaner, num_jobs=num_cores) as pfilter:
        # pfilter.imap_apply returns an iterator of processed Document objects
        for i, doc in enumerate(pfilter.imap_apply(input_documents)):
            filtered_texts_content.append(doc.text)
            if (i + 1) % 20 == 0:
                print(f"HojiChar processed {i+1}/{len(texts)} texts...")

    print(f"HojiChar filtering complete. Processed {len(filtered_texts_content)} texts.")

    # Save filtered texts to a JSONL file
    filtered_output_file = "filtered_texts.jsonl"
    with open(filtered_output_file, 'w', encoding='utf-8') as f:
        for text in filtered_texts_content:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"Filtered texts saved to {filtered_output_file}")

    if filtered_texts_content:
        print(f"First filtered text sample: '{filtered_texts_content[0][:100]}...'")

    # Step 3: Score filtered texts using GPU scorer
    print("Preparing to score filtered texts...")

    # Attempt to import the scorer class
    try:
        from gpu_scorer import Fineweb2EduJapaneseScoreClassifier
    except ImportError:
        print("Error: Could not import Fineweb2EduJapaneseScoreClassifier from gpu_scorer.py.")
        print("Please ensure transformers and torch are installed if you intend to run this step.")
        # In a real scenario where we expect this to run, we might exit.
        # For now, we'll just print a message and skip this step.
        Fineweb2EduJapaneseScoreClassifier = None

    if Fineweb2EduJapaneseScoreClassifier:
        # Load filtered texts from the JSONL file
        texts_to_score = []
        if os.path.exists(filtered_output_file):
            with open(filtered_output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        texts_to_score.append(json.loads(line)['text'])
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed line in {filtered_output_file}: {line.strip()}")

        if not texts_to_score:
            print("No texts loaded from filtered_texts.jsonl to score.")
        else:
            print(f"Loaded {len(texts_to_score)} texts for scoring from {filtered_output_file}.")

            model_name = "hotchpotch/fineweb-2-edu-japanese-classifier"
            # Defaulting to 'cpu' due to installation issues.
            # The class itself will check for CUDA availability if 'cuda' is passed.
            # Forcing 'cpu' here to prevent download attempts if torch is not fully installed.
            device_to_use = "cpu"

            print(f"Initializing classifier with device: {device_to_use}...")
            # Reduce batch_size and num_workers if on CPU and memory is a concern.
            # These values are from the original script, adjust as needed.
            # Batch size 1 might be very slow but safest for CPU memory.
            # Num workers should ideally be 0 or 1 on CPU for this kind of task.
            try:
                classifier = Fineweb2EduJapaneseScoreClassifier(
                    model_path=model_name,
                    device=device_to_use,
                    batch_size=32, # Smaller batch for CPU
                    num_workers=1,  # Fewer workers for CPU
                    show_progress=True,
                    # dtype can be float32 for CPU
                    dtype=hojichar.utils.infer_torch_dtype(device_to_use) # hojichar has a helper
                )

                print("Scoring texts...")
                # Note: This step will be very slow on CPU and require transformers/torch
                scored_results = classifier.predict(texts_to_score) # List of (is_edu, score)

                scored_output_file = "scored_texts.jsonl"
                with open(scored_output_file, 'w', encoding='utf-8') as f:
                    for i, text in enumerate(texts_to_score):
                        is_educational, score = scored_results[i]
                        f.write(json.dumps({"text": text, "is_educational": is_educational, "score": score}) + "\n")
                print(f"Scored texts saved to {scored_output_file}")
                if scored_results:
                    print(f"First scored text sample: Score={scored_results[0][1]:.2f}, Educational={scored_results[0][0]}, Text='{texts_to_score[0][:100]}...'")

            except NameError as ne: # Catch if torch or transformers aren't installed
                 if 'torch' in str(ne).lower() or 'transformers' in str(ne).lower():
                    print(f"Skipping scoring: Required library not found ({ne}). Please install torch and transformers.")
                 else:
                    raise ne # Re-raise if it's another NameError
            except Exception as e:
                print(f"Error during scoring: {e}")
                print("Skipping scoring step.")
    else:
        print("Skipping scoring step as classifier could not be imported (likely missing dependencies).")


if __name__ == "__main__":
    main()
