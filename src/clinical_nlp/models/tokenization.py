from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]
