import pytest

from claire_agent import InconsistencyAgent


@pytest.mark.asyncio
async def test_inconsistency_agent():
    agent = InconsistencyAgent(
        "gpt-5-mini",
        "azure_openai",
        num_results_per_query=3,
        reasoning_effort="low",
    )

    report = await agent.analyze_claim(
        "Water boils at 100 degrees Celsius.",
        passage=(
            "Title: Water Boiling > Basics\n\n"
            "Water typically boils at 100 degrees Celsius when measured at standard atmospheric pressure."
        ),
    )

    assert report is not None
    assert report.search_results and len(report.search_results) > 0, (
        "There should be at least one search result"
    )
    assert report.verdict in {"consistent", "inconsistent"}, (
        "Verdict should be either 'consistent' or 'inconsistent'"
    )
    assert isinstance(report.explanation, str) and report.explanation.strip(), (
        "Explanation should be non-empty"
    )
    assert isinstance(report.wording_feedback, str) and report.wording_feedback.strip(), (
        "Wording feedback should be non-empty"
    )


@pytest.mark.asyncio
async def test_analyze_passage_for_inconsistencies():
    agent = InconsistencyAgent(
        "gpt-5-mini",
        "azure_openai",
        num_results_per_query=3,
        reasoning_effort="low",
    )

    reports = await agent.analyze_passage_for_inconsistencies(
        passage=(
            "Title: Haruki Murakami > Biography\n\n"
            "Haruki Murakami (村上 春樹, Murakami Haruki; born January 12, 1949[1]) is a Japanese writer. "
            "His novels, essays, and short stories have been best-sellers in Japan and internationally."
        ),
    )

    assert isinstance(reports, list)
    assert len(reports) >= 2, "At least two claims should be extracted"
    for r in reports:
        assert r.search_results and len(r.search_results) > 0, (
            "There should be at least one search result"
        )
        assert r.verdict in {"consistent", "inconsistent"}, (
            "Verdict should be either 'consistent' or 'inconsistent'"
        )
        assert isinstance(r.explanation, str) and r.explanation.strip(), (
            "Explanation should be non-empty"
        )
        assert isinstance(r.wording_feedback, str) and r.wording_feedback.strip(), (
            "Wording feedback should be non-empty"
        )
