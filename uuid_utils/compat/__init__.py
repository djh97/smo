"""Compatibility surface used by langchain-core."""

from uuid import UUID, uuid4


def uuid7(timestamp=None, nanos=None) -> UUID:
    """Fallback UUID generator compatible with uuid_utils.compat.uuid7."""
    return uuid4()


__all__ = ["UUID", "uuid7"]
