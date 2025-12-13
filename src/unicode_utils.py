"""
Unicode normalization utilities for handling emoji encoding issues.

This module provides robust functions to fix emoji encoding problems that occur
when AI APIs return escaped unicode sequences like \\uD83C\\uDFA7 instead of 🎧.

The main function `normalize_unicode()` should be called immediately after
parsing any JSON response from AI APIs to ensure emojis are properly decoded.
"""

import re
import json
import logging

logger = logging.getLogger(__name__)


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode encoding in text, converting escaped sequences to actual characters.

    This fixes issues where emojis appear as \\uD83C\\uDFA7 instead of 🎧.
    Handles multiple levels of escaping and various edge cases.

    IMPORTANT: This function is careful NOT to corrupt already-valid UTF-8 text
    that contains emojis like keycap numbers (1️⃣, 2️⃣, etc.).

    Args:
        text: Input text that may contain escaped unicode sequences

    Returns:
        Text with all unicode escapes properly decoded to actual characters

    Examples:
        >>> normalize_unicode("Hello \\\\uD83C\\\\uDFA7 World")
        'Hello 🎧 World'
        >>> normalize_unicode("Already fine 🎧")
        'Already fine 🎧'
        >>> normalize_unicode("1️⃣ Keycap emoji")
        '1️⃣ Keycap emoji'
    """
    if not isinstance(text, str):
        return text

    if not text:
        return text

    # Check if there are any escaped unicode sequences to fix
    # Must be a literal backslash followed by 'u' and 4 hex digits
    if not re.search(r'\\u[0-9a-fA-F]{4}', text):
        return text

    # Method 1: Use regex replacement - safest for mixed content
    # This only replaces actual \uXXXX escape sequences without corrupting
    # existing properly-encoded UTF-8 characters (like keycap emojis)
    try:
        # Pattern matches \uXXXX (with single backslash in string)
        pattern = r'\\u([0-9a-fA-F]{4})'

        def replace_unicode(match):
            try:
                codepoint = int(match.group(1), 16)
                return chr(codepoint)
            except (ValueError, OverflowError):
                return match.group(0)

        decoded = re.sub(pattern, replace_unicode, text)

        # Handle surrogate pairs by doing a second pass
        # Surrogate pairs appear as two consecutive \uXXXX sequences
        # e.g., \uD83C\uDFA7 becomes two characters that need combining
        # After first pass, surrogates will be actual surrogate chars
        try:
            # Encode to UTF-16 with surrogatepass to handle any surrogate pairs
            decoded = decoded.encode('utf-16', 'surrogatepass').decode('utf-16')
        except (UnicodeDecodeError, UnicodeEncodeError):
            # If that fails, the text is fine as-is
            pass

        return decoded

    except Exception as e:
        logger.debug(f"Regex method failed: {e}, trying json.loads")

    # Method 2: Try json.loads trick for JSON-style escapes
    try:
        # Wrap in quotes and parse as JSON string
        # This handles standard JSON unicode escapes
        decoded = json.loads(f'"{text}"')
        return decoded
    except (json.JSONDecodeError, ValueError):
        pass

    # If all else fails, return original
    logger.warning(f"Unicode normalization failed, returning original")
    return text


def normalize_dict_unicode(data: dict) -> dict:
    """
    Recursively normalize unicode in all string values of a dictionary.

    Args:
        data: Dictionary potentially containing escaped unicode in values

    Returns:
        Dictionary with all string values normalized
    """
    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = normalize_unicode(value)
        elif isinstance(value, dict):
            result[key] = normalize_dict_unicode(value)
        elif isinstance(value, list):
            result[key] = normalize_list_unicode(value)
        else:
            result[key] = value
    return result


def normalize_list_unicode(data: list) -> list:
    """
    Recursively normalize unicode in all string values of a list.

    Args:
        data: List potentially containing escaped unicode in values

    Returns:
        List with all string values normalized
    """
    if not isinstance(data, list):
        return data

    result = []
    for item in data:
        if isinstance(item, str):
            result.append(normalize_unicode(item))
        elif isinstance(item, dict):
            result.append(normalize_dict_unicode(item))
        elif isinstance(item, list):
            result.append(normalize_list_unicode(item))
        else:
            result.append(item)
    return result


def normalize_json_response(json_data) -> any:
    """
    Normalize unicode in parsed JSON data from AI API responses.

    This is the main function to call after json.loads() on AI responses.
    It handles dicts, lists, and strings appropriately.

    Args:
        json_data: Parsed JSON data (dict, list, or string)

    Returns:
        The same structure with all unicode escapes normalized

    Example:
        >>> response = json.loads(ai_response)
        >>> normalized = normalize_json_response(response)
    """
    if isinstance(json_data, str):
        return normalize_unicode(json_data)
    elif isinstance(json_data, dict):
        return normalize_dict_unicode(json_data)
    elif isinstance(json_data, list):
        return normalize_list_unicode(json_data)
    else:
        return json_data


# Convenience alias
fix_emoji_encoding = normalize_unicode
