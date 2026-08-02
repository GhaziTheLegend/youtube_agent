"""
Smoke test: generates a short voiceover clip using edge-tts (free, no API key
needed) to confirm your environment can produce audio before Phase 3.

Run directly:
    python tests/connectivity/test_tts.py

Then play tests/connectivity/output_test.mp3 to confirm it sounds right.
"""

import asyncio

import edge_tts

OUTPUT_PATH = "tests/connectivity/output_test.mp3"
VOICE = "en-US-GuyNeural"  # matches the default in channels/techreviews.yaml
TEXT = "This is a test of the voiceover pipeline. If you can hear this clearly, you're good to go."


async def main():
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_PATH)
    print(f"✅ Saved test voiceover to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
