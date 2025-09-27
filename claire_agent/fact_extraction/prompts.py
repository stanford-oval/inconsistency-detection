"""Prompt templates used for fact extraction and filtering."""

EXTRACTION_PROMPT_TEMPLATE = (
    "# instruction\n"
    "You are an expert fact extractor tasked with identifying and listing atomic facts from a given text. "
    "Your goal is to produce a comprehensive list of facts that are explicitly stated or directly inferrable "
    "from the provided information.\n\n"
    "Instructions:\n"
    "1. Read the passage carefully.\n"
    "2. Extract all atomic facts from the information provided. An atomic fact is a single, indivisible piece of "
    "information that cannot be broken down further without losing its meaning or accuracy.\n"
    "3. Include only facts that are explicitly stated or can be directly and unambiguously inferred from the text.\n"
    "4. Do not add any external knowledge or assumptions not present in the given information.\n"
    "5. Ensure that each fact is self-contained and can be independently fact-checked.\n"
    "6. For each fact, identify the EXACT sentence or phrase from the original text that contains this information.\n\n"
    "Document your reasoning by populating the `FactExtractionProcess` fields in the provided schema.\n"
    "Group related facts by topic and evaluate each one for atomicity, redundancy, and evidentiary support.\n\n"
    "Return your response strictly as JSON that conforms to the `FactExtractionResponse` schema. Do not include any "
    "additional commentary or formatting outside of valid JSON.\n\n"
    "# input\n"
    "Here is the passage you need to analyze:\n\n"
    "<passage>\n"
    "{passage}\n"
    "</passage>\n"
)


FACT_FILTER_PROMPT_TEMPLATE = (
    "# instruction\n"
    "You are tasked with determining whether a statement from Wikipedia is worthy of verification. Your goal is to "
    "distinguish between statements that require verification and those that are common knowledge, subjective "
    "opinions, or statements about Wikipedia's structure.\n\n"
    "# input\n"
    "Here is the statement to analyze:\n"
    "<statement>\n"
    "{statement}\n"
    "</statement>\n\n"
    "Evaluate the statement using these criteria:\n"
    "1. Objectivity: Is the statement an objective claim that can be proven true or false?\n"
    "2. Non-triviality: Is the information beyond common knowledge or easily observable facts?\n"
    "3. Relevance: Does the statement relate to the subject matter and not to Wikipedia's structure or editing process?\n"
    "4. Context sufficiency: Does the statement provide enough context to be verified?\n\n"
    "Provide a concise but thorough justification that references each criterion and clearly states whether the "
    "statement should be verified.\n\n"
    "Respond strictly as JSON that conforms to the `FactFilterResponse` schema, filling in the `reasoning` and "
    "`determination` fields. Do not include any additional commentary or formatting outside of valid JSON.\n"
)


__all__ = [
    "EXTRACTION_PROMPT_TEMPLATE",
    "FACT_FILTER_PROMPT_TEMPLATE",
]
