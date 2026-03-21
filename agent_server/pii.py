"""
PII redaction for maintenance knowledge before storage.

Five patterns covering common PII in manufacturing environments:
phone numbers, emails, SSNs, badge numbers, and phone extensions.
"""

import re
from typing import List, Tuple, Union

PII_PATTERNS: List[Union[Tuple[str, str], Tuple[str, str, int]]] = [
    # Phone numbers: 555-123-4567, 555.123.4567, 5551234567
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE_REDACTED]"),
    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),
    # Social Security Numbers: 123-45-6789
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
    # Badge numbers: badge #12345, badge 1234
    (r"\bbadge\s*#?\s*\d{4,8}\b", "[BADGE_REDACTED]", re.IGNORECASE),
    # Phone extensions: ext 1234, extension 12345
    (r"\b(?:ext|extension)\s*\.?\s*\d{3,5}\b", "[EXT_REDACTED]", re.IGNORECASE),
]


def redact_pii(text: str) -> str:
    """Redact personally identifiable information from text."""
    result = text
    for pattern_info in PII_PATTERNS:
        if len(pattern_info) == 3:
            pattern, replacement, flags = pattern_info
            result = re.sub(pattern, replacement, result, flags=flags)
        else:
            pattern, replacement = pattern_info
            result = re.sub(pattern, replacement, result)
    return result
