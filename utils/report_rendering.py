"""Utilities for rendering inconsistency reports and related content."""

from __future__ import annotations

import re

import markdown
from markdownify import markdownify as md
from rich import box
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from claire_agent import InconsistencyReport
from retrieval.document_block import Block


console = Console()


def _convert_custom_markdown_table_to_html(markdown_table_string: str) -> str:
    """Convert a custom Markdown table string into HTML.

    Args:
        markdown_table_string: Input string with a custom ``<Table>`` wrapper
            produced by ``_triplet_serialize``.

    Returns:
        An HTML table equivalent of the provided Markdown string.
    """
    content = markdown_table_string.strip().removeprefix("<Table>").removesuffix("</Table>").strip()
    lines = content.split("\n")

    if len(lines) < 2:
        return "<table><!-- Malformed custom table input --></table>"

    header_line = lines[0]
    data_lines = lines[2:]

    header_cells_match = re.findall(r"\|\s*(.*?)\s*(?=\|)", header_line)
    if not header_cells_match:
        return "<table><!-- Invalid header format --></table>"
    header_cells = header_cells_match
    num_cols = len(header_cells)

    html_header = "<thead>\n  <tr>\n"
    for header in header_cells:
        html_header += f"    <th>{header}</th>\n"
    html_header += "  </tr>\n</thead>"

    html_body = "<tbody>\n"
    for row_line in data_lines:
        row_line = row_line.strip()
        if not row_line:
            continue

        html_body += "  <tr>\n"
        if ":" in row_line and not row_line.startswith("|"):
            parts = row_line.split(":", 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            html_body += f"    <td>{key}</td>\n"
            colspan_attr = f' colspan="{num_cols - 1}"' if num_cols > 1 else ""
            if num_cols > 1 or colspan_attr == "":
                html_body += f"    <td{colspan_attr}>{value}</td>\n"
        elif row_line.startswith("|") and row_line.endswith("|"):
            row_cells_match = re.findall(r"\|\s*(.*?)\s*(?=\|)", row_line)
            row_cells = row_cells_match
            row_cells.extend([""] * (num_cols - len(row_cells)))
            for i in range(num_cols):
                cell_content = row_cells[i] if i < len(row_cells) else ""
                html_body += f"    <td>{cell_content}</td>\n"
        else:
            html_body += f'    <td colspan="{num_cols}">{row_line}</td>\n'

        html_body += "  </tr>\n"
    html_body += "</tbody>"

    html_table = f"<table>\n{html_header}\n{html_body}\n</table>"
    return html_table


def _markdown_to_html(markdown_string: str) -> str:
    """Convert Markdown content, including custom tables, to HTML.

    Args:
        markdown_string: Markdown string that may include custom ``<Table>`` blocks.

    Returns:
        HTML representation of the provided Markdown string.
    """

    def replace_table_match(match: re.Match[str]) -> str:
        custom_table_block = match.group(0)
        return _convert_custom_markdown_table_to_html(custom_table_block)

    processed_markdown = re.sub(
        r"<Table>.*?</Table>",
        replace_table_match,
        markdown_string,
        flags=re.DOTALL,
    )

    html_output = markdown.markdown(processed_markdown, extensions=["extra", "tables"])
    return html_output


def _html_to_markdown(html_string: str) -> str:
    """Convert HTML to Markdown using consistent formatting.

    Args:
        html_string: HTML input to be converted.

    Returns:
        A Markdown formatted string preserving headings in ATX style.
    """
    return md(html_string, heading_style="ATX")


def _clean_markdown(md_text: str) -> str:
    """Normalize Markdown content by round-tripping through HTML.

    Args:
        md_text: Markdown string that may contain irregular formatting.

    Returns:
        A cleaned Markdown string with consistent formatting.
    """
    html_text = _markdown_to_html(md_text)
    cleaned_md = _html_to_markdown(html_text)
    return cleaned_md


def _render_search_evidence(blocks: list[Block]) -> None:
    """Render a list of search result blocks inside Rich panels."""
    if not blocks:
        console.print(Text("No search evidence was captured for this claim.", style="italic dim"))
        return

    console.rule(Text("Passages that Were Looked At", style="bold cyan"))

    for index, block in enumerate(blocks, start=1):
        markdown_content = _clean_markdown(block.content)

        panel_body: RenderableType = Markdown(markdown_content)

        title_text = Text(f"[{index}] {block.full_title}")
        if block.url:
            title_text.stylize(f"link {block.url}")

        console.print(
            Panel(
                panel_body,
                title=title_text,
                border_style="cyan",
                box=box.SQUARE,
                expand=True,
            )
        )


def render_inconsistency_report(report: InconsistencyReport) -> None:
    """Render an inconsistency report to the terminal using Rich components.

    Args:
        report: Structured report containing verdict and supporting metadata.
    """
    verdict_color = "green" if report.verdict == "consistent" else "red"

    console.rule(Text("Inconsistency Report", style=f"bold {verdict_color}"))

    console.print(
        Panel(
            Text(report.claim_text, style="bold"),
            title="Claim",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True,
        )
    )

    console.print(
        Panel(
            Text(report.claim_passage),
            title="Claim Was Extracted From",
            border_style="magenta",
            box=box.ROUNDED,
            expand=True,
        )
    )
    verdict_panel = Panel(
        Text(report.verdict.upper(), justify="center", style=f"bold {verdict_color}"),
        title="Verdict",
        border_style=verdict_color,
        box=box.ROUNDED,
        expand=False,
    )
    console.print(verdict_panel, justify="center")

    console.print(
        Panel(
            Text(report.explanation),
            title="Why?",
            border_style=verdict_color,
            box=box.ROUNDED,
            expand=True,
        )
    )

    console.print(
        Panel(
            Text(report.wording_feedback),
            title="How the Claim Could Be Reworded for Clarity",
            border_style="yellow",
            box=box.ROUNDED,
            expand=True,
        )
    )

    _render_search_evidence(report.search_results)
    console.print()


__all__ = [
    "render_inconsistency_report",
]
