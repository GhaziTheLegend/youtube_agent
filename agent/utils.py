"""Small shared helpers used across modules that parse LLM JSON output."""


def strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` even when told not to."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text.strip()
