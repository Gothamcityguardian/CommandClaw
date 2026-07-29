"""
Data structures for the four-quadrant knowledge map.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Quadrant(str, Enum):
    KNOWN_KNOWN = "known_known"         # know it, aware of it
    KNOWN_UNKNOWN = "known_unknown"     # don't know it, aware of the gap
    UNKNOWN_KNOWN = "unknown_known"     # know it, haven't connected it yet
    UNKNOWN_UNKNOWN = "unknown_unknown" # don't know it, unaware of the gap


QUADRANT_LABELS = {
    Quadrant.KNOWN_KNOWN:     "Known Knowns",
    Quadrant.KNOWN_UNKNOWN:   "Known Unknowns",
    Quadrant.UNKNOWN_KNOWN:   "Unknown Knowns",
    Quadrant.UNKNOWN_UNKNOWN: "Unknown Unknowns",
}

QUADRANT_COLORS = {
    Quadrant.KNOWN_KNOWN:     "green",
    Quadrant.KNOWN_UNKNOWN:   "yellow",
    Quadrant.UNKNOWN_KNOWN:   "cyan",
    Quadrant.UNKNOWN_UNKNOWN: "red",
}


@dataclass
class KnowledgeItem:
    topic: str
    quadrant: Quadrant
    confidence: float       # 0.0–1.0
    evidence: str = ""
    web_researched: bool = False


@dataclass
class KnowledgeMap:
    items: list[KnowledgeItem] = field(default_factory=list)

    def by_quadrant(self, q: Quadrant) -> list[KnowledgeItem]:
        return [i for i in self.items if i.quadrant == q]

    def overall_confidence(self) -> float:
        if not self.items:
            return 0.0
        return sum(i.confidence for i in self.items) / len(self.items)

    def add_from_dict(self, data: dict):
        quadrant_keys = {
            "known_knowns":     Quadrant.KNOWN_KNOWN,
            "known_unknowns":   Quadrant.KNOWN_UNKNOWN,
            "unknown_knowns":   Quadrant.UNKNOWN_KNOWN,
            "unknown_unknowns": Quadrant.UNKNOWN_UNKNOWN,
        }
        for key, quadrant in quadrant_keys.items():
            for entry in data.get(key, []):
                if isinstance(entry, dict):
                    self.items.append(KnowledgeItem(
                        topic=entry.get("topic", ""),
                        quadrant=quadrant,
                        confidence=float(entry.get("confidence", 0.5)),
                        evidence=entry.get("evidence", ""),
                    ))
                elif isinstance(entry, str):
                    default_conf = 0.8 if quadrant == Quadrant.KNOWN_KNOWN else 0.2
                    self.items.append(KnowledgeItem(
                        topic=entry,
                        quadrant=quadrant,
                        confidence=default_conf,
                    ))
