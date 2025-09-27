from pydantic import BaseModel, ConfigDict, Field

from retrieval.document_block import Block


class Claim(BaseModel):
    claim_id: str = Field(...)
    claim_text: str = Field(...)
    context_block: Block = Field(
        ..., description="The original block the claim has been extracted from"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Disallow extra fields
    )
