import os
import json
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmCategory,
    HarmBlockThreshold,
)

# Assuming the Gemini API key is set as an environment variable
# For local development, you might use:
# from dotenv import load_dotenv
# load_dotenv()
# API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=API_KEY)
# However, for a production/shared environment, it's better to configure
# this outside the code or assume it's pre-configured.
# For now, I'll assume genai is configured elsewhere or the user will handle it.

# Placeholder for API Key configuration.
# If you have a specific way to load API keys (e.g. from a config file or env var),
# that logic should be added here or handled by the calling application.
# For now, this will only work if the environment is already configured for Gemini.
# Initialize Gemini client
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set. Gemini calls may fail.")
client = genai.Client(api_key=API_KEY)

DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:  HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

def call_gemini_api(
    prompt: str,
    response_schema: dict,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2, # Lower temperature for more deterministic, schema-following output
    top_p: float = 0.95,
    top_k: int = 64,
    max_output_tokens: int = 8192, # Max for gemini-1.5-flash
) -> dict | None:
    """
    Calls the Gemini API with a given prompt and expects a JSON response
    structured according to the provided schema.

    Args:
        prompt: The input prompt for the Gemini model.
        response_schema: The JSON schema for the expected response format.
        model_name: The name of the Gemini model to use.
        temperature: Controls randomness. Lower is more deterministic.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        max_output_tokens: Maximum number of tokens in the response.

    Returns:
        A dictionary parsed from the Gemini API's JSON response, or None if an error occurs.
    """
    try:
        cfg = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
        )

        # Issue the request
        response = client.generative.generate_content(
            model=model_name,
            prompt=prompt,
            config=cfg,
            safety_settings=DEFAULT_SAFETY_SETTINGS,
        )

        # Parse the JSON text
        text = getattr(response, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from Gemini response: {e}")
                print(f"Response text was: {text}")
                return None
        else:
            print("Gemini response was empty or malformed.")
            return None

    except Exception as e:
        print(f"An error occurred while calling Gemini API: {e}")
        return None


# Specific functions will be added below, using the above generic caller.
# These will require the prompt templates and schemas, which are planned for the next step.
# For now, I will define them with placeholders for templates/schemas.

# Import prompt templates and schemas
# Assuming prompts.py is in the parent directory of utils, which is src/text-filter-pipeline/
# So, the import path from src.text-filter-pipeline.utils would be ..prompts
try:
    from ..prompts import (
        INPUT_DIFFICULTY_RATING_TEMPLATE,
        OUTPUT_DIFFICULTY_JSON_SCHEMA,
        INPUT_QUALITY_RATING_TEMPLATE,
        OUTPUT_QUALITY_JSON_SCHEMA,
        INPUT_CLASSIFICATION_TEMPLATE,
        OUTPUT_CLASSIFICATION_JSON_SCHEMA,
    )
except ImportError:
    # Fallback for cases where the script might be run directly or module structure is different
    # This might happen if 'src' is not in PYTHONPATH or running from a different working directory.
    # For package execution, the relative import `..prompts` should work.
    print("Could not import from ..prompts, trying direct import (may fail in package context)")
    from prompts import (  # type: ignore
        INPUT_DIFFICULTY_RATING_TEMPLATE,
        OUTPUT_DIFFICULTY_JSON_SCHEMA,
        INPUT_QUALITY_RATING_TEMPLATE,
        OUTPUT_QUALITY_JSON_SCHEMA,
        INPUT_CLASSIFICATION_TEMPLATE,
        OUTPUT_CLASSIFICATION_JSON_SCHEMA,
    )

def get_difficulty_rating(instruction: str) -> dict | None:
    """
    Assigns a difficulty level to an instruction using Gemini.
    """
    prompt = INPUT_DIFFICULTY_RATING_TEMPLATE.format(input=instruction)
    return call_gemini_api(prompt, OUTPUT_DIFFICULTY_JSON_SCHEMA)

def get_quality_rating(instruction: str) -> dict | None:
    """
    Checks the quality of an instruction using Gemini.
    """
    prompt = INPUT_QUALITY_RATING_TEMPLATE.format(input=instruction)
    return call_gemini_api(prompt, OUTPUT_QUALITY_JSON_SCHEMA)

def get_instruction_classification(instruction: str) -> dict | None:
    """
    Categorizes an instruction using Gemini.
    """
    prompt = INPUT_CLASSIFICATION_TEMPLATE.format(input=instruction)
    return call_gemini_api(prompt, OUTPUT_CLASSIFICATION_JSON_SCHEMA)

# Example of how these might be called once prompts are loaded:
if __name__ == '__main__':
    # This block is for testing and requires GEMINI_API_KEY to be set.
    # It also assumes that this script can find prompts.py (e.g., it's in the same directory
    # or the Python path is configured correctly).
    # To run this test:
    # 1. Make sure GEMINI_API_KEY is set in your environment.
    # 2. Ensure prompts.py is accessible (e.g., copy it to this directory for a quick test,
    #    or run this script as part of the package `python -m src.text_filter_pipeline.utils.gemini_utils`).
    if not API_KEY:
        print("Please set GEMINI_API_KEY to run tests.")
    else:
        # For __main__ execution, we might need to adjust Python path or use absolute imports if possible
        # For simplicity, if run directly, it might rely on `prompts.py` being in the same folder or PYTHONPATH.
        # The try-except for imports above attempts to handle this.

        sample_instruction = "Can you explain quantum physics in simple terms, including its historical development and key experiments, tailored for a high school student with some basic physics knowledge but no prior exposure to quantum concepts? Please also suggest some further reading."

        print("Testing Gemini Utils (requires GEMINI_API_KEY)...")

        print(f"\n--- Getting Difficulty Rating for: \"{sample_instruction}\" ---")
        difficulty = get_difficulty_rating(sample_instruction)
        print(f"Difficulty: {json.dumps(difficulty, indent=2)}")

        print(f"\n--- Getting Quality Rating for: \"{sample_instruction}\" ---")
        quality = get_quality_rating(sample_instruction)
        print(f"Quality: {json.dumps(quality, indent=2)}")

        print(f"\n--- Getting Instruction Classification for: \"{sample_instruction}\" ---")
        classification = get_instruction_classification(sample_instruction)
        print(f"Classification: {json.dumps(classification, indent=2)}")

        # Test with a different instruction
        sample_instruction_2 = "gimme pizza recipe fast"
        print(f"\n--- Getting Difficulty Rating for: \"{sample_instruction_2}\" ---")
        difficulty_2 = get_difficulty_rating(sample_instruction_2)
        print(f"Difficulty 2: {json.dumps(difficulty_2, indent=2)}")

        print(f"\n--- Getting Quality Rating for: \"{sample_instruction_2}\" ---")
        quality_2 = get_quality_rating(sample_instruction_2)
        print(f"Quality 2: {json.dumps(quality_2, indent=2)}")

        print(f"\n--- Getting Instruction Classification for: \"{sample_instruction_2}\" ---")
        classification_2 = get_instruction_classification(sample_instruction_2)
        print(f"Classification 2: {json.dumps(classification_2, indent=2)}")
