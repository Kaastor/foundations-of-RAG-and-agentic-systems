"""User profiles, access policies, retention hints, and simple route guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raglab.domain.models import UserContext
from raglab.storage.json_store import read_json


ROLE_DISCLOSURES: dict[str, tuple[str, ...]] = {
    "external": ("public",),
    "distributor_support": ("public", "partner_shareable"),
    "field_support": ("public", "partner_shareable", "internal"),
    "engineering": ("public", "partner_shareable", "internal"),
    "compliance": ("public", "partner_shareable", "internal", "restricted"),
}


def user_from_role(role: str, user_id: str | None = None, region: str = "global", high_stakes: bool = False) -> UserContext:
    """Create a simple user context from a role."""
    disclosures = ROLE_DISCLOSURES.get(role, ("public",))
    return UserContext(
        user_id=user_id or role,
        role=role,
        region=region,
        allowed_disclosures=disclosures,
        high_stakes=high_stakes,
    )


def load_named_user(path: str | Path, name: str) -> UserContext:
    """Load a user profile from the bundled examples file."""
    payload = read_json(path)
    for row in payload:
        if row["user_id"] == name:
            disclosures = tuple(row.get("allowed_disclosures", ROLE_DISCLOSURES.get(row["role"], ("public",))))
            return UserContext(
                user_id=row["user_id"],
                role=row["role"],
                region=row.get("region", "global"),
                language=row.get("language", "en"),
                allowed_disclosures=disclosures,
                source_preference=row.get("source_preference", "authority"),
                high_stakes=row.get("high_stakes", False),
            )
    raise KeyError(f"Unknown user profile: {name}")


def can_view(disclosure: str, user: UserContext) -> bool:
    """Check whether the user's disclosure class allows a document."""
    return disclosure in set(user.allowed_disclosures)


def allow_action(action_name: str, state: dict[str, Any]) -> bool:
    """A tiny governance predicate used by the agent controller.

    The goal is not comprehensive policy enforcement. It is to make the policy
    hook explicit and easy to inspect in code and traces.
    """
    if action_name.startswith("side_effect:") and not state.get("approved", False):
        return False
    if state.get("requested_disclosure") == "internal" and state.get("user_role") == "external":
        return False
    return True


def retention_tag_for_trace(request_kind: str) -> str:
    """Assign a toy retention tag so docs can point at a concrete policy hook."""
    return "short_lived" if request_kind in {"agent", "answer"} else "standard"
