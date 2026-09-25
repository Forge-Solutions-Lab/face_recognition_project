"""Member Registry and Management Module.

Provides the central roster of recognized persons and lookup utilities.
New members can be registered simply by appending to the MEMBERS list without
modifying any other system files.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class Member(TypedDict):
    id: str
    name: str


# ── Registered Member Catalog ────────────────────────────────────────────────
MEMBERS: List[Member] = [
    {"id": "6601001", "name": "สมชาย"},
    {"id": "6601002", "name": "สมหญิง"},
]


# ── Lookup Helpers ────────────────────────────────────────────────────────────
def get_all_members() -> List[Member]:
    """Return a copy of all registered members."""
    return list(MEMBERS)


def get_member_by_id(member_id: str) -> Optional[Member]:
    """Retrieve a member dict by their unique ID string.

    Args:
        member_id: The unique ID string to search for (e.g. '6601001').

    Returns:
        Matching Member dictionary or None if not found.
    """
    cleaned_id = member_id.strip()
    for member in MEMBERS:
        if member["id"] == cleaned_id:
            return member
    return None


def format_member_dir_name(member_id: str, member_name: str) -> str:
    """Standardize the directory name for saving face samples.

    Args:
        member_id: Unique member ID.
        member_name: Member display name.

    Returns:
        Formatted folder name, e.g. '6601001_สมชาย'.
    """
    clean_id = member_id.strip()
    clean_name = member_name.strip().replace(" ", "_")
    return f"{clean_id}_{clean_name}"
