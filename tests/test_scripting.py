"""
Tests for the scripting module. Mocks the LLM, same pattern as
test_research.py: fast, free, offline, deterministic.

Run with:
    pytest tests/test_scripting.py -v
"""

import json
from unittest.mock import patch

import pytest

from agent import scripting
from agent.config import ChannelConfig, ProductionConfig, ScriptConfig
from agent.db import session as db_session
from agent.db.models import Idea, Script

SAMPLE_CONFIG = ChannelConfig(
    channel_id="testchannel",
    channel_name="Test Channel",
    niche="AI tools for students",
    content_pillars=["AI tool reviews"],
    target_audience="university students",
    script=ScriptConfig(target_duration_seconds=45),
    production=ProductionConfig(voice_id="en-US-GuyNeural"),
)

FAKE_SCRIPT_RESPONSE = json.dumps(
    {
        "hook": "Everyone's using ChatGPT wrong for studying. Here's the fix.",
        "scenes": [
            {"text": "Most students just paste notes and ask for a summary.", "visual_cue": "phone screen, notes app"},
            {"text": "That skips the part where you actually learn anything.", "visual_cue": "person looking confused at screen"},
            {"text": "Instead, ask it to quiz you back. Recall beats rereading.", "visual_cue": "close-up of quiz app interface"},
        ],
        "cta": "Try it before your next exam and thank me later.",
    }
)


def _make_idea(status="approved") -> Idea:
    return Idea(
        id=1,
        channel_id="testchannel",
        title="Stop Using ChatGPT Wrong",
        hook="Everyone studies with AI wrong.",
        angle="Recall-based prompting instead of summarization.",
        why_now="Exam season.",
        search_interest="High",
        competition="Medium",
        status=status,
    )


def test_generate_script_from_approved_idea():
    idea = _make_idea(status="approved")
    with patch("agent.scripting.generate", return_value=FAKE_SCRIPT_RESPONSE):
        result = scripting.generate_script(SAMPLE_CONFIG, idea)

    assert result.hook.startswith("Everyone's using ChatGPT")
    assert len(result.scenes) == 3
    assert result.scenes[0].line_number == 1
    assert result.estimated_duration_seconds > 0


def test_generate_script_refuses_unapproved_idea():
    idea = _make_idea(status="pending_review")
    with patch("agent.scripting.generate", return_value=FAKE_SCRIPT_RESPONSE):
        with pytest.raises(RuntimeError, match="not 'approved'"):
            scripting.generate_script(SAMPLE_CONFIG, idea)


def test_generate_script_refuses_rejected_idea():
    idea = _make_idea(status="rejected")
    with patch("agent.scripting.generate", return_value=FAKE_SCRIPT_RESPONSE):
        with pytest.raises(RuntimeError):
            scripting.generate_script(SAMPLE_CONFIG, idea)


def test_generate_script_allows_rescripting_already_scripted_idea():
    """Regenerating a script for an idea that already has one should be
    allowed (e.g. you didn't like the first draft) - only pending_review
    should be blocked."""
    idea = _make_idea(status="scripted")
    with patch("agent.scripting.generate", return_value=FAKE_SCRIPT_RESPONSE):
        result = scripting.generate_script(SAMPLE_CONFIG, idea)
    assert result.hook


def test_duration_is_computed_not_trusted_from_llm():
    """Confirms we're computing duration from word count ourselves, not
    parsing some duration field the LLM might hallucinate."""
    idea = _make_idea()
    with patch("agent.scripting.generate", return_value=FAKE_SCRIPT_RESPONSE):
        result = scripting.generate_script(SAMPLE_CONFIG, idea)

    # Recompute expected duration by hand and compare
    expected_hook = round(len(idea.hook.split()) * 0 + len("Everyone's using ChatGPT wrong for studying. Here's the fix.".split()) / 150 * 60, 1)
    assert result.hook_duration_seconds == expected_hook


def test_run_scripting_saves_script_and_advances_idea_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_session.init_db()

    with db_session.get_session() as sess:
        idea = Idea(
            channel_id="testchannel",
            title="Stop Using ChatGPT Wrong",
            hook="Everyone studies with AI wrong.",
            angle="Recall-based prompting instead of summarization.",
            why_now="Exam season.",
            search_interest="High",
            competition="Medium",
            status="approved",
        )
        sess.add(idea)
        sess.commit()
        idea_id = idea.id

    with patch("agent.scripting.generate", return_value=FAKE_SCRIPT_RESPONSE):
        generated, script_id = scripting.run_scripting(SAMPLE_CONFIG, idea_id)

    with db_session.get_session() as sess:
        idea_row = sess.get(Idea, idea_id)
        assert idea_row.status == "scripted"

        script_row = sess.get(Script, script_id)
        assert script_row.status == "pending_review"
        assert script_row.idea_id == idea_id
        scenes = json.loads(script_row.scenes_json)
        assert len(scenes) == 3


def test_run_scripting_raises_for_wrong_channel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_session.init_db()

    with db_session.get_session() as sess:
        idea = Idea(
            channel_id="a_different_channel",
            title="X", hook="X", angle="X", why_now="X",
            search_interest="High", competition="Low", status="approved",
        )
        sess.add(idea)
        sess.commit()
        idea_id = idea.id

    with pytest.raises(RuntimeError, match="No idea"):
        scripting.run_scripting(SAMPLE_CONFIG, idea_id)
