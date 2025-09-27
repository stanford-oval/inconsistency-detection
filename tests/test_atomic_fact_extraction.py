import pytest

from claire_agent import ClaimExtractor
from utils.logger import logger


@pytest.mark.asyncio
async def test_extract_atomic_facts():
    extractor = ClaimExtractor(
        engine="gpt-5-mini",
        model_provider="azure_openai",
        reasoning_effort="low",
    )

    extracted_facts = await extractor.extract(
        passage=(
            "Title: Haruki Murakami > Biography\n\n"
            "Haruki Murakami (村上 春樹, Murakami Haruki; born January 12, 1949[1]) is a Japanese writer. "
            "His novels, essays, and short stories have been best-sellers in Japan and internationally, with his work "
            "translated into 50 languages[2] and having sold millions of copies outside Japan.[3][4] He has received "
            "numerous awards for his work, including the Gunzo Prize for New Writers, the World Fantasy Award, the "
            "Tanizaki Prize, Yomiuri Prize for Literature, the Frank O'Connor International Short Story Award, the "
            "Noma Literary Prize, the Franz Kafka Prize, the Kiriyama Prize for Fiction, the Goodreads Choice Awards "
            "for Best Fiction, the Jerusalem Prize, and the Princess of Asturias Awards."
        ),
    )

    assert isinstance(extracted_facts, list)
    assert all(isinstance(fact, str) for fact in extracted_facts)
    assert len(extracted_facts) > 0, "No facts were extracted."

    logger.info("Extracted Facts:")
    for fact in extracted_facts:
        logger.info(f"- {fact}")
