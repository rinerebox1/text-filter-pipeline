# Prompt and schema for Instruction Difficulty Rating
INPUT_DIFFICULTY_RATING_TEMPLATE = """
# Instruction

You first need to identify the given user intent and then label the difficulty level of the user query based on the content of the user query.

## User Query
```
{input}
```

## Output Format
Given the user query, in your output, you first need to identify the user intent and the knowledge needed to solve the task in the user query.
Then, rate the difficulty level of the user query as `very easy`, `easy`, `medium`, `hard`, or `very hard`.

Now, please output the user intent and difficulty level below in a json format by filling in the placeholders in []:
```
{{
    "intent": "The user wants to [....]",
    "knowledge": "To solve this problem, the models need to know [....]",
    "difficulty": "[very easy/easy/medium/hard/very hard]"
}}
```
""".lstrip()

OUTPUT_DIFFICULTY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "knowledge": {"type": "string"},
        "difficulty": {
            "type": "string",
            "enum": ["very easy", "easy", "medium", "hard", "very hard"],
        },
    },
    "required": ["intent", "knowledge", "difficulty"],
}


# Prompt and schema for Instruction Quality Evaluation
INPUT_QUALITY_RATING_TEMPLATE = """
# Instruction

You need to rate the quality of the user query based on its clarity, specificity, and coherence.

The rating scale is as follows:

- very poor: The query is unclear, vague, or incoherent. It lacks essential information and context.
- poor: The query is somewhat unclear or lacks important details. It requires significant clarification.
- average: The query is moderately clear and specific. It may require some additional information for a complete understanding.
- good: The query is clear, specific, and mostly well-formed. It provides sufficient context for understanding the user's intent.
- excellent: The query is very clear, specific, and well-articulated. It contains all the necessary information and context for providing a comprehensive response.

## User Query
```
{input}
```

## Output Format
Given the user query, you first need to give an assesement, highlighting the strengths and/or weaknesses of the user query.
Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]:
```
{{
    "explanation": "[...]",
    "quality": "[very poor/poor/average/good/excellent]"
}}
```
""".lstrip()

OUTPUT_QUALITY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "quality": {
            "type": "string",
            "enum": ["very poor", "poor", "average", "good", "excellent"],
        },
    },
    "required": ["explanation", "quality"],
}


# Prompt and schema for Instruction Classification
INPUT_CLASSIFICATION_TEMPLATE = """
# Instruction

Please label the task tags for the user query.

## User Query
```
{input}
```

## Tagging the user input
Please label the task tags for the user query. You will need to analyze the user query and select the most relevant task tag from the list below.

all_task_tags = [
    "Information seeking",  # Users ask for specific information or facts about various topics.
    "Reasoning",  # Queries require logical thinking, problem-solving, or processing of complex ideas.
    "Planning",  # Users need assistance in creating plans or strategies for activities and projects.
    "Editing",  # Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
    "Coding & Debugging",  # Users seek help with writing, reviewing, or fixing code in programming.
    "Math",  # Queries related to mathematical concepts, problems, and calculations.
    "Role playing",  # Users engage in scenarios requiring ChatGPT to adopt a character or persona.
    "Data analysis",  # Requests involve interpreting data, statistics, or performing analytical tasks.
    "Creative writing",  # Users seek assistance with crafting stories, poems, or other creative texts.
    "Advice seeking",  # Users ask for recommendations or guidance on various personal or professional issues.
    "Brainstorming",  # Involves generating ideas, creative thinking, or exploring possibilities.
    "Others"  # Any queries that do not fit into the above categories or are of a miscellaneous nature.
]

## Output Format:
Note that you can only select a single primary tag. Other applicable tags can be added to the list of other tags.
Now, please output your tags below in a json format by filling in the placeholders in <...>:
```
{{
    "primary_tag": "<primary tag>",
    "other_tags": ["<tag 1>", "<tag 2>", ... ]
}}
```
""".lstrip()


OUTPUT_CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_tag": {
            "type": "string",
            "enum": [
                "Information seeking",
                "Reasoning",
                "Planning",
                "Editing",
                "Coding & Debugging",
                "Math",
                "Role playing",
                "Data analysis",
                "Creative writing",
                "Advice seeking",
                "Brainstorming",
                "Others",
            ],
        },
        "other_tags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "Information seeking",
                    "Reasoning",
                    "Planning",
                    "Editing",
                    "Coding & Debugging",
                    "Math",
                    "Role playing",
                    "Data analysis",
                    "Creative writing",
                    "Advice seeking",
                    "Brainstorming",
                    "Others",
                ],
            },
        },
    },
    "required": ["primary_tag", "other_tags"],
}
