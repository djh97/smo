"""Local fallback shim for uuid_utils on Windows/Python 3.9.

This project only needs uuid_utils.compat.uuid7 via langchain-core for tracing IDs.
The upstream compiled wheel can fail to load in some local environments, so we
provide a small pure-Python fallback that is sufficient for local development.
"""

from uuid import UUID, uuid4


def uuid7(timestamp=None, nanos=None) -> UUID:
    """Fallback UUID generator.

    This is not a true UUIDv7 implementation; it only provides a stable UUID
    object for local development when the compiled uuid_utils wheel is unusable.
    """
    return uuid4()


__all__ = ["UUID", "uuid7"]
