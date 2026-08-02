"""
Smoke test: confirms your GEMINI_API_KEY works. This is the Gemini
equivalent of test_claude.py - use whichever LLM provider you have a key for.

Run directly:
    python tests/connectivity/test_gemini.py

Note on the SDK: Google renamed/replaced their Python package. This uses the
current one, `google-genai` (import as `from google import genai`) - NOT the
older `google-generativeai` package, which is deprecated.

Note on model name: "flash" models get version-bumped fairly often (e.g.
gemini-2.0-flash -> gemini-2.5-flash -> newer). If GEMINI_MODEL below returns
a "model not found" error, check ai.google.dev/gemini-api/docs/models for
the current flash model name and update the constant.
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"  # update if this becomes unavailable


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY not set in .env. Get one free at aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents="Reply with exactly: Gemini API is connected.",
    )
    print(response.text)


if __name__ == "__main__":
    main()
