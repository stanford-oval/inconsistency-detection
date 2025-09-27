from datetime import datetime
import re
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)


class Block(BaseModel):
    """An indexing/retrieval unit. Can be a paragraph, list, linearized table, or linearized Infobox."""

    document_title: str = Field(..., description="The title of the document")
    section_title: str = Field(
        ...,
        description="The hierarchical section title of the block excluding `document_title`, e.g. 'Land > Central Campus'. Section title can be empty, for instance the first section of Wikipedia articles.",
    )
    content: str = Field(..., description="The content of the block, usually in Markdown format")
    last_edit_date: Optional[datetime] = Field(
        None,
        description="The last edit date of the block in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS",
    )
    url: Optional[str] = Field(None, description="The URL of the block")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Disallow extra fields
    )

    @field_serializer("last_edit_date")
    def serialize_datetime(self, v: datetime):
        return v.strftime("%Y-%m-%d") if v else None

    @property
    def date_human_readable(self) -> Optional[str]:
        return self.last_edit_date.strftime("%B %d, %Y") if self.last_edit_date else None

    @property
    def full_title(self) -> str:
        if not self.section_title:
            return self.document_title
        return self.document_title + " > " + self.section_title

    @property
    def combined_text(self) -> str:
        return self.full_title + " " + self.content

    @property
    def id(self) -> int:
        return abs(hash(self.combined_text))

    @field_validator("last_edit_date", mode="before")
    def parse_last_edit_date(cls, last_edit_date):
        if isinstance(last_edit_date, str):
            try:
                return Block.convert_string_to_datetime(last_edit_date)
            except ValueError:
                raise ValueError(
                    f"Invalid date format for `last_edit_date`: '{last_edit_date}'. It should be in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ"
                )
        return last_edit_date

    @staticmethod
    def convert_string_to_datetime(date_string: str) -> datetime:
        """Convert a string to a datetime object.

        The string can be in the following formats:
        - YYYY-MM-DDTHH:MM:SSZ
        - YYYY-MM-DDTHH:MM:SS
        - YYYY-MM-DD HH:MM:SS
        - YYYY-MM-DD
        This conversion will ignore the timezone information.
        """
        date_formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
        ]
        for date_format in date_formats:
            try:
                return datetime.strptime(date_string, date_format)
            except ValueError:
                continue
        raise ValueError(
            f"Invalid date format for `last_edit_date`: '{date_string}'. It should be in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS"
        )

    @staticmethod
    def block_list_to_string(
        blocks: list["Block"], start_index: int = 1, truncate_text: bool = False
    ) -> str:
        def maybe_truncate(text: str) -> str:
            max_length = 200
            if not truncate_text or len(text) <= max_length:
                return text

            prefix = text[:max_length]
            cutoff = None
            for match in re.finditer(r"\s", prefix):
                cutoff = match.start()

            if cutoff is not None and cutoff > 0 and cutoff >= max_length // 2:
                truncated = text[:cutoff]
            else:
                truncated = prefix

            truncated = truncated.rstrip()
            if not truncated:
                truncated = prefix.rstrip() or prefix

            remaining_characters = len(text) - len(truncated)
            return f"{truncated} ... [{remaining_characters} more characters]"

        return "\n\n".join(
            f"[{i + start_index}] {maybe_truncate(b.combined_text)}" for i, b in enumerate(blocks)
        )
