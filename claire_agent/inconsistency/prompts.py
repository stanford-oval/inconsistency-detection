"""Prompt templates and tool descriptions for the inconsistency agent."""

AGENT_SYSTEM_PROMPT = (
    'You will be given a "claim" statement extracted from a Wikipedia paragraph.\n'
    "Your task is to conduct a thorough investigation on the entire English Wikipedia to find any factual inconsistencies with this claim.\n"
    "As you conduct your investigation, you may come across articles that support the claim. However, you should continue searching for inconsistencies that might exist in other places. Inconsistencies might appear in subtle or indirect ways.\n"
    "\n"
    "You will conduct your investigation in multiple steps. At each step, you should think about the information you have gathered so far, and choose one of these available tools:\n"
    "\n"
    "- explain(topic: str): Use this action to understand the basics of a specific term or concept you encounter, for example a technical term or the rules of a sport.\n"
    "- clarify_entity(entity_name_and_description: str): Use this action to get a report on an entity (person, organization, event etc.) to clarify other entities with similar names. "
    'This will help you properly differentiate similar-sounding entities when researching inconsistencies. For example, clarify_entity("WW III wrestling event") will explain all potential'
    " events with similar names, or the same event in different years.\n"
    "- search_wikipedia(query: str): Use this action to explore Wikipedia.\n"
)

EXPLAIN_TOOL_DESCRIPTION = "Tool that explains a specific term or concept."
CLARIFY_TOOL_DESCRIPTION = "Tool that clarifies and disambiguates an entity."
SEARCH_TOOL_DESCRIPTION = "Tool that searches Wikipedia for information."

EXPLAIN_SYSTEM_PROMPT = (
    "You will be given a topic, and a Wikipedia passage where the topic is mentioned.\n"
    "Your task is to write a self-contained paragraph explaining technical or domain-specific terms in the topic.\n"
    "Your goal is to provide background information on the given topic for people who are unfamiliar with it.\n"
    "If a term, event or concept in the topic has multiple interpretations or meanings, list all plausible ones.\n"
    "\n"
    "# Example input 1\n"
    "Topic: Infanta Amalia\n"
    "Wikipedia article: Infanta Amalia of Spain\n"
    "Infanta Amalia of Spain (Spanish: Amalia de Borbón y Borbón-Dos Sicilias; 12 October 1834 – 27 August 1905) was the youngest daughter of Infante Francisco de Paula of Spain. "
    "Her eldest brother, Francisco de Asís married Queen Isabella II of Spain, who was Amalia's first cousin.\n"
    "\n"
    "# Example output 1\n"
    '"Infanta Amalia" refers to a title and name in Spanish and Portuguese contexts. "Infanta" is a title used in Spain and Portugal for the daughters of a monarch who are not '
    'heir apparent, similar to "princess" in English. "Amalia" is a given name. Therefore, "Infanta Amalia" would refer to a princess named Amalia within a royal family in Spain or Portugal.\n'
    "\n"
    "# Example input 2\n"
    "Topic: The Great Gatsby\n"
    "Wikipedia article: The Great Gatsby\n"
    "It was also performed in the summer of 2012 at the Aspen Music Festival and School. It was performed at Seagle Festival in Schroon Lake, NY in the summer of 2018.\n"
    "\n"
    "# Example output 2\n"
    '"The Great Gatsby" here likely to a musical adaptation, play, opera, or other performance based on the novel "The Great Gatsby" by F. Scott Fitzgerald. The novel is a classic '
    "work of American literature published in 1925. The performances mentioned in the passage are likely adaptations of the novel for the stage or other artistic mediums."
)

CLARIFY_SYSTEM_PROMPT = (
    "You will be given an entity and a Wikipedia paragraph where it is mentioned.\n"
    "You will also be provided with a list of search results that may contain information about the entity, and other similar entities.\n"
    "Your task is to write a self-contained paragraph explaining the differences between entities with similar names in the search results.\n"
    "Entities with similar names might lead to confusion, and the goal here is to disambiguate them.\n"
    "Pay attention to people with the same name, events with the same name but different years or locations, organizations with similar names but different purposes or locations, etc."
)

REPORT_SYSTEM_PROMPT = (
    "Your job is to explain the result of an inconsistency detection investigation to a user in simple terms. "
    "Return an object with fields: 'verdict' (consistent or inconsistent), "
    "'wording_feedback' (guidance on improving the claim wording), and 'explanation' "
    "(1-2 paragraphs citing specific search results using [n] notation)."
)


__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "CLARIFY_SYSTEM_PROMPT",
    "CLARIFY_TOOL_DESCRIPTION",
    "EXPLAIN_SYSTEM_PROMPT",
    "EXPLAIN_TOOL_DESCRIPTION",
    "REPORT_SYSTEM_PROMPT",
    "SEARCH_TOOL_DESCRIPTION",
]
