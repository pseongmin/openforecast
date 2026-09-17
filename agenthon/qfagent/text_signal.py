"""Turn the frozen text corpus into scenario knobs — conservatively.

The playbook's measured failure mode is tone-averaging a salted corpus, so this
layer only ever *widens* on inferred signal and never narrows. A drift is applied
only when the same direction is stated by several dated documents.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from .engine import Scenario

HAWKISH = re.compile(r"\b(hike|tighten|raise(?:s|d)? rates|inflation risks? (?:remain|elevated)|further increases?)\b", re.I)
DOVISH = re.compile(r"\b(cut|ease|lower(?:s|ed)? rates|accommodat|downside risks? to growth|pause)\b", re.I)
SHOCK = re.compile(r"\b(emergency|crisis|default|collapse|intervention|unscheduled|shock|war|sanction)\b", re.I)
DECISION_IN_WINDOW = re.compile(r"\b(meeting|decision|vote|referendum|election)\b", re.I)


@dataclass(frozen=True)
class Document:
    path: str
    text: str


def read_corpus(text_dir: pathlib.Path, max_docs: int = 400) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(text_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".json", ".jsonl", ".csv"}:
            continue
        try:
            docs.append(Document(str(path.relative_to(text_dir)), path.read_text(errors="ignore")))
        except OSError:
            continue
        if len(docs) >= max_docs:
            break
    return docs


def scenarios_from_corpus(docs: list[Document], asset_ids: list[str]) -> tuple[list[Scenario], list[str]]:
    """Return (scenarios, notes). Empty scenarios = pure numerical forecast."""
    if not docs:
        return [], ["corpus empty or absent"]
    hawk = sum(len(HAWKISH.findall(d.text)) for d in docs)
    dove = sum(len(DOVISH.findall(d.text)) for d in docs)
    shock = sum(len(SHOCK.findall(d.text)) for d in docs)
    decision = sum(len(DECISION_IN_WINDOW.findall(d.text)) for d in docs)
    notes = [f"{len(docs)} documents; hawkish={hawk} dovish={dove} shock={shock} decision-words={decision}"]

    scenarios: list[Scenario] = []
    # Direction: only if lopsided. Inferred, so it may move the centre but must not narrow.
    total = hawk + dove
    if total >= 6 and max(hawk, dove) / total >= 0.7:
        sign = 1.0 if hawk > dove else -1.0
        strength = min(0.5, 0.25 + 0.05 * (max(hawk, dove) - min(hawk, dove)) / 10)
        scenarios.append(Scenario("directional", 0.55, drift_in_sd=sign * strength, spread_multiple=1.1))
        scenarios.append(Scenario("no-move", 0.45, drift_in_sd=0.0, spread_multiple=1.0))
        notes.append(f"directional tilt {sign:+.0f} at {strength:.2f} sd (inferred; spread widened, not narrowed)")
    # Shock vocabulary: fatten the tails rather than move the centre.
    if shock >= 4 or decision >= 8:
        scenarios.append(Scenario("stress", 0.2, drift_in_sd=0.0, spread_multiple=1.8))
        if not any(s.name == "no-move" for s in scenarios):
            scenarios.append(Scenario("calm", 0.8, drift_in_sd=0.0, spread_multiple=1.0))
        notes.append("stress branch added (tail widening only)")
    return scenarios, notes
