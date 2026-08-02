# YouTube Shorts AI Agent

Faceless, AI-voiceover YouTube Shorts pipeline: research → script → produce → publish → learn from results.
See `ROADMAP.md` (or the roadmap doc from our conversation) for the full phased plan.

## Phase 1 status: ✅ Research module

What's new since Phase 0:
- `agent/llm.py` — provider-agnostic LLM wrapper (Gemini or Claude, whichever key you have)
- `agent/db/` — SQLite database (via SQLAlchemy) tracking ideas through their lifecycle
- `agent/research.py` — generates video ideas, validates them, saves them for review
- `tests/test_research.py` — automated tests with a mocked LLM (no API key/cost needed to run these)

### Try it (once your `.env` has GEMINI_API_KEY or ANTHROPIC_API_KEY)
```bash
python -m agent.cli research techreviews      # generates a batch of ideas
python -m agent.cli list-ideas techreviews    # review them
python -m agent.cli approve-idea 1            # approve one for Phase 2 scripting
```

### Run the test suite (no API key needed - uses a mocked LLM)
```bash
pytest tests/ -v
```

## Phase 2 status: ✅ Scripting module

What's new since Phase 1:
- `agent/scripting.py` — turns an approved idea into a full shot-by-shot script (hook, scenes with visual cues, CTA)
- Duration is computed from word count in code, never trusted from the LLM directly
- **Guard rail**: `generate_script()` refuses to run against an idea that isn't `approved` — this is enforced in code, not just a convention, so later phases can trust the chain
- `tests/test_scripting.py` — covers the guard rail explicitly, plus duration calculation and DB state transitions

### Try it
```bash
python -m agent.cli research techreviews
python -m agent.cli approve-idea 1
python -m agent.cli script techreviews 1       # will refuse if idea 1 isn't approved
python -m agent.cli show-script 1
python -m agent.cli approve-script 1
python -m agent.cli list-scripts techreviews
```

### Idea/Script lifecycle so far
```
Idea:   pending_review -> approved -> scripted
Script:              pending_review -> approved
```

## Phase 3A status: ✅ Voiceover (production, part 1 of 4)

What's new since Phase 2:
- `agent/production/voiceover.py` — synthesizes per-line audio via edge-tts and builds an exact timeline (start/end seconds for every line, plus word-level timing for future caption burn-in)
- **Key design choice**: word timing comes from edge-tts's own boundary events, not a separate Whisper transcription pass — the TTS engine already knows where it put each word
- **Key design choice**: audio duration is measured from the real written file (via `mutagen`), never estimated — same "compute, don't trust" principle as Phase 2's script-length estimate
- `tests/test_voiceover.py` — the network call is mocked, but real ffmpeg-generated audio files are used, so duration-reading is tested against genuine audio, not a faked number

### Try it for real (this needs actual network access - won't work in a restricted sandbox)
```bash
python -m agent.cli produce-voiceover techreviews 1
```
This requires script `1` to have status `approved`. It writes:
- `output/techreviews/script_1/audio/*.mp3` — one file per line
- `output/techreviews/script_1/timeline.json` — exact timing for every line + word

Listen to the output files to sanity-check pacing and pronunciation before moving on - Phase 3B (visuals) and 3C (assembly) will build directly on top of `timeline.json`, so it's worth catching voice issues now rather than after the whole video is assembled.

### If edge-tts is blocked on your network (corporate/school firewall, etc.)

`agent/production/voiceover.py` is a provider-agnostic dispatcher — the same
pattern as `agent/llm.py`. Two backends exist:

| Provider | File | Network? | Word timing |
|---|---|---|---|
| `edge_tts` (default) | `tts_edge.py` | Required (unofficial MS API) | Native, free |
| `piper` | `tts_piper.py` | **None** - fully local | Via local Whisper pass |

To switch a channel to Piper:

1. Install the extra deps (already in `requirements.txt`): `piper-tts`, `faster-whisper`
2. Download a voice model using our **integrity-verified** downloader (not
   `piper.download_voices` directly - that one doesn't check whether the
   download was actually complete, which matters if your network ever
   truncates long connections):
   ```bash
   python -m agent.cli download-voice en_US-lessac-medium
   ```
   This verifies each file's byte count against the server's declared size
   and automatically retries if a download comes back truncated.
3. In the channel's YAML config:
   ```yaml
   production:
     voice_id: en_US-lessac-medium
     tts_provider: piper
   ```
4. Run `produce-voiceover` as normal - no code changes needed.

First run per voice will also download a small local Whisper model
(~140MB, one-time, cached after) for word-timing extraction.




## Setup

1. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up your secrets**
   ```bash
   cp .env.example .env
   ```
   Then fill in `.env`:
   - `ANTHROPIC_API_KEY` — get one at console.anthropic.com
   - `PEXELS_API_KEY` — free, needed from Phase 3 onward (pexels.com/api)
   - YouTube keys — not needed yet, see "Get this started early" below

3. **Verify the config system works**
   ```bash
   python -m agent.cli show-config techreviews
   ```
   You should see the validated config printed as JSON. Try breaking a field
   in `channels/techreviews.yaml` (e.g. set `content_pillars: []`) and re-run
   to see validation catch it.

4. **Run the connectivity smoke tests**
   ```bash
   python tests/connectivity/test_claude.py   # needs ANTHROPIC_API_KEY
   python tests/connectivity/test_tts.py      # no key needed (edge-tts is free)
   ```

## Get this started early (Phase 4 needs it, but approval can be slow)

Set up your Google Cloud project and YouTube Data API v3 access now, even
though we won't write the upload code until Phase 4:

1. Go to console.cloud.google.com, create a new project.
2. Enable "YouTube Data API v3" under APIs & Services.
3. Configure the OAuth consent screen (External, add your channel's Google
   account as a test user while in testing mode).
4. Create OAuth client credentials (type: Desktop app), download the JSON,
   save it as `secrets/client_secret.json` (create the `secrets/` folder —
   it's git-ignored).
5. Note your daily quota (10,000 units by default; an upload costs ~1,600
   units, so ~6 uploads/day per project). We'll handle multi-channel quota
   planning in Phase 6.

## Adding a new channel

Copy `channels/techreviews.yaml` to `channels/<your_channel_id>.yaml` and
edit every field. Run `python -m agent.cli show-config <your_channel_id>`
to confirm it validates.

## Project structure

```
agent/
  config.py       - channel config schema + loader (Phase 0)
  cli.py          - command-line entry point
  research.py     - Phase 1
  scripting.py    - Phase 2
  production/     - Phase 3 (voiceover, visuals, assembly, captions)
  metadata.py     - Phase 4
  publisher.py    - Phase 4
  analytics.py    - Phase 5
  db/             - data models (Phase 1+)
channels/         - one YAML config per channel
tests/connectivity/ - manual API smoke tests
```
