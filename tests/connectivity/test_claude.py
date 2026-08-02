"""
Smoke test: confirms your ANTHROPIC_API_KEY works before you build anything
on top of it. Run directly, not via pytest:

    python tests/connectivity/test_claude.py
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in."
        )

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[{"role": "user", "content": "Reply with exactly: Claude API is connected."}],
    )
    print(response.content[0].text)


if __name__ == "__main__":
    main()
