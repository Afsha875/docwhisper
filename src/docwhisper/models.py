"""Shared datatypes. This file is the contract between modules — keep it tiny."""

from dataclasses import dataclass


@dataclass
class Chunk:
    id: str  # "<relpath>:<position>"
    path: str  # relative to the indexed root
    position: int
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, higher is better
