"""
Tests for the research module. These mock agent.research.generate() so they
run instantly, for free, offline, and deterministically - no real API calls.
This is the standard pattern for testing anything that wraps an external API:
test YOUR logic (prompt building, parsing, validation, DB writes) against a
fixed fake response, and rely on the one-off connectivity scripts in
tests/connectivity/ to confirm the real API actually works.

Run with:
    pytest tests/test_research.py -v
"""

import json
from unittest.mock import patch

from agent import research
from agent.config import ChannelConfig, ProductionConfig
from agent.db import session as db_session
from agent.db.models import Idea

# A minimal valid config, built directly (not from YAML) so this test doesn't
# depend on the example channel file staying unchanged.
SAMPLE_CONFIG = ChannelConfig(
    channel_id="testchannel",
    channel_name="Test Channel",
    niche="AI tools for students",
    content_pillars=["AI tool reviews", "productivity hacks"],
    target_audience="university students",
    production=ProductionConfig(voice_id="en-US-GuyNeural"),
)

# A fake but schema-valid LLM response, standing in for a real Gemini/Claude call.
FAKE_LLM_RESPONSE = json.dumps(
    [
        {
            "title": "3 AI Tools That Actually Save You Time (Not Hype)",
            "hook": "I tested 12 AI study tools so you don't have to.",
            "angle": "Ranking by actual time saved, not feature lists.",
            "why_now": "New semester = peak search interest in study tools.",
            "search_interest": "High",
            "competition": "Medium",
        },
        {
            "title": "The Free AI Tool Students Are Sleeping On",
            "hook": "This one's free and better than the paid version.",
            "angle": "Direct head-to-head comparison with a popular paid tool.",
            "why_now": "Recent free-tier update makes this newly relevant.",
            "search_interest": "Medium",
            "competition": "Low",
        },
    ]
)


def test_generate_ideas_parses_and_validates_response():
    with patch("agent.research.generate", return_value=FAKE_LLM_RESPONSE):
        ideas = research.generate_ideas(SAMPLE_CONFIG)

    assert len(ideas) == 2
    assert ideas[0].title.startswith("3 AI Tools")
    assert ideas[0].search_interest == "High"
    assert ideas[1].competition == "Low"


def test_generate_ideas_handles_code_fenced_response():
    """LLMs sometimes wrap JSON in ```json ... ``` despite instructions not to."""
    fenced = f"```json\n{FAKE_LLM_RESPONSE}\n```"
    with patch("agent.research.generate", return_value=fenced):
        ideas = research.generate_ideas(SAMPLE_CONFIG)
    assert len(ideas) == 2


def test_generate_ideas_raises_on_invalid_json():
    with patch("agent.research.generate", return_value="not json at all"):
        try:
            research.generate_ideas(SAMPLE_CONFIG)
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass


def test_generate_ideas_skips_invalid_items_but_keeps_valid_ones():
    mixed = json.dumps(
        [
            json.loads(FAKE_LLM_RESPONSE)[0],
            {"title": "Missing required fields"},  # invalid - will be dropped
        ]
    )
    with patch("agent.research.generate", return_value=mixed):
        ideas = research.generate_ideas(SAMPLE_CONFIG)
    assert len(ideas) == 1


def test_run_research_saves_ideas_to_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_session.init_db()  # fresh DB in tmp_path, isolated from your real data/agent.db

    with patch("agent.research.generate", return_value=FAKE_LLM_RESPONSE):
        rows = research.run_research(SAMPLE_CONFIG)

    assert len(rows) == 2
    assert all(row.status == "pending_review" for row in rows)
    assert all(row.channel_id == "testchannel" for row in rows)

    with db_session.get_session() as sess:
        count = sess.query(Idea).filter(Idea.channel_id == "testchannel").count()
    assert count == 2
