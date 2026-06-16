"""Triage: classify each source file by ROLE, so content is routed, not blindly dumped.

The starter pipeline concatenated every file in the client folder into every prompt, which
pulls in decoy figures (general market commentary) and internal docs. Here we classify each
file's role and route it: client-fact sources are extracted; per-client guidance is carried
through; global policy is handled by encoded rules; decoys are dropped.

Classification is deterministic by filename (the source names are stable). The principle is
"route, don't drop": only high-confidence decoys are excluded, everything else is kept.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class Role(str, Enum):
    DB = "db"  # structured system of record
    MEETING_NOTES = "meeting_notes"  # decisions + live values
    REPORT_REQUEST = "report_request"  # scope / instruction
    GUIDANCE = "guidance"  # fde_notes: rules + per-client hints
    IMAGE = "image"  # statement image, needs OCR
    DECOY = "decoy"  # general market/portfolio docs — excluded
    UNKNOWN = "unknown"  # unrecognised file: kept, but routed to an exploratory pass + flagged


# Filenames whose content must never reach extraction or generation.
_DECOY_STEMS = {"platform_market_update", "portfolio_pack"}
# template_spec drives the report definition, not client content — not extracted here.
_IGNORED_STEMS = {"template_spec"}


def classify(path: Path) -> Role | None:
    """Return the role for a file, or None if it should be ignored entirely."""
    stem = path.stem.lower()
    suffix = path.suffix.lower()

    if stem in _IGNORED_STEMS:
        return None
    if stem in _DECOY_STEMS:
        return Role.DECOY
    if suffix in {".png", ".jpg", ".jpeg"}:
        return Role.IMAGE
    # Treat any .json as the custody db: the canonical file is client_data_db.json, and no client
    # folder in the example data carries a second JSON. If a future client ever ships a non-db JSON
    # this heuristic would misroute it — at that point, match on the stem alone.
    if stem == "client_data_db" or suffix == ".json":
        return Role.DB
    if stem == "meeting_notes":
        return Role.MEETING_NOTES
    if stem == "report_request":
        return Role.REPORT_REQUEST
    if stem == "fde_notes":
        return Role.GUIDANCE
    # Unrecognised file: route it to the exploratory unknown-document pass rather than folding it
    # into the structured extraction (where novel content can be silently missed) or dropping it.
    return Role.UNKNOWN


def triage_folder(client_dir: Path) -> dict[Role, list[Path]]:
    """Group a client folder's files by role (excluding ignored and, separately, decoys)."""
    grouped: dict[Role, list[Path]] = {}
    for path in sorted(client_dir.iterdir()):
        if not path.is_file():
            continue
        role = classify(path)
        if role is None:
            continue
        grouped.setdefault(role, []).append(path)
    return grouped
