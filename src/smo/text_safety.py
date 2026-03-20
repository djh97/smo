from __future__ import annotations

import unicodedata


def sanitize_provider_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    cleaned_chars: list[str] = []
    for char in normalized:
        codepoint = ord(char)
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        if codepoint < 32 and char not in "\n\r\t":
            continue
        cleaned_chars.append(char)
    return "".join(cleaned_chars).strip()
