from pydantic import BaseModel
from typing import List


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    text: str
    entities: List[Entity]
