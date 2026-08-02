"""
Scripting module - Phase 2.

Flow: approved Idea -> prompt (built around the idea's specific angle) -> LLM
-> validated ScriptDraft -> duration estimated from word count (NOT trusted
from the LLM - see note below) -> saved to DB with status="pending_review".

Guard rail: this module refuses to script an idea that hasn't been approved
(Phase 1's human checkpoint). This isn't just a convention - it's enforced in
generate_script() below, so Phase 4's publisher can eventually trust that
anything with a script went through a real approval, without re-checking.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from agent.config import ChannelConfig
from agent.db.models import Idea, Script
from agent.db.session import get_session
from agent.llm import generate
from agent.utils import strip_code_fences

# Average narration pace for punchy Shorts-style delivery. This is a starting
# assumption - once Phase 3 generates real TTS audio, you'll get ACTUAL
# durations and can compare them against this estimate to calibrate it per
# voice/channel. For now it's just used to sanity-check script length before
# you spend TTS credits on something 3x too long.
WORDS_PER_MINUTE = 150


class ScriptDraftScene(BaseModel):
    text: str
    visual_cue: str = Field(..., description="What's on screen during this line")


class ScriptDraft(BaseModel):
    """What we ask the LLM for. Deliberately does NOT include duration - we
    compute that ourselves from word count rather than trusting an LLM to do
    arithmetic, which is a common source of silently-wrong numbers."""
    hook: str = Field(..., description="First 5-8 seconds - the pattern interrupt")
    scenes: list[ScriptDraftScene] = Field(..., min_length=1)
    cta: str = Field(..., description="Call to action / outro line")


class ScriptScene(BaseModel):
    """A scene after duration has been computed - what actually gets stored."""
    line_number: int
    text: str
    visual_cue: str
    est_duration_seconds: float


class GeneratedScript(BaseModel):
    idea_id: int
    hook: str
    hook_duration_seconds: float
    scenes: list[ScriptScene]
    cta: str
    cta_duration_seconds: float
    estimated_duration_seconds: float
    target_duration_seconds: float

    @property
    def within_target(self) -> bool:
        """True if estimated length is within 25% of the channel's target -
        loose on purpose, since word-count-based estimates are approximate."""
        if self.target_duration_seconds <= 0:
            return True
        ratio = self.estimated_duration_seconds / self.target_duration_seconds
        return 0.75 <= ratio <= 1.25


PROMPT_TEMPLATE = """You are a scriptwriter for a faceless YouTube Shorts channel.

CHANNEL
- Niche: {niche}
- Tone: {tone}
- Target duration: ~{target_duration} seconds (keep this tight - Shorts, not long-form)

THIS VIDEO'S IDEA (already approved - build the script specifically around this)
- Title: {idea_title}
- Hook concept: {idea_hook}
- Angle (the distinct take this video takes - THIS IS THE CORE OF THE SCRIPT,
  not a side note): {idea_angle}
- Why it's timely: {idea_why_now}

REQUIREMENTS
- The script MUST deliver on the specific angle above throughout, not just in
  the hook. A script that states facts without executing the angle fails the
  brief, and also risks the "no unique insight" monetization issue the
  channel is designed to avoid.
- must_include: {must_include}
- must_avoid: {must_avoid}
- Each scene should be short (1-2 sentences) - Shorts pacing is fast.
- Include a visual_cue for every line describing what's on screen (this
  drives which stock/AI visuals get fetched in Phase 3, so be specific -
  "close-up of hands typing" not just "typing").

OUTPUT FORMAT
Return ONLY a JSON object (no markdown fences, no commentary) with exactly
these keys:
- "hook": string
- "scenes": array of objects, each with "text" and "visual_cue"
- "cta": string
"""


def _build_prompt(config: ChannelConfig, idea: Idea) -> str:
    return PROMPT_TEMPLATE.format(
        niche=config.niche,
        tone=config.script.tone,
        target_duration=config.script.target_duration_seconds,
        idea_title=idea.title,
        idea_hook=idea.hook,
        idea_angle=idea.angle,
        idea_why_now=idea.why_now,
        must_include=", ".join(config.script.must_include) or "none specified",
        must_avoid=", ".join(config.script.must_avoid) or "none specified",
    )


def _estimate_duration(text: str) -> float:
    words = len(text.split())
    return round(words / WORDS_PER_MINUTE * 60, 1)


def generate_script(config: ChannelConfig, idea: Idea) -> GeneratedScript:
    """Call the LLM, validate its response, compute real durations.
    Raises RuntimeError if the idea isn't approved, or if the LLM response
    can't be trusted (invalid JSON / fails schema)."""
    if idea.status not in ("approved", "scripted"):
        raise RuntimeError(
            f"Idea {idea.id} has status={idea.status!r}, not 'approved'. "
            f"Run: python -m agent.cli approve-idea {idea.id}"
        )

    prompt = _build_prompt(config, idea)
    raw = generate(prompt, json_mode=True)
    cleaned = strip_code_fences(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM did not return valid JSON. Raw response:\n{raw}") from e

    try:
        draft = ScriptDraft.model_validate(parsed)
    except ValidationError as e:
        raise RuntimeError(f"LLM response failed schema validation:\n{e}") from e

    hook_duration = _estimate_duration(draft.hook)
    cta_duration = _estimate_duration(draft.cta)

    scenes: list[ScriptScene] = []
    scenes_total = 0.0
    for i, scene in enumerate(draft.scenes, start=1):
        duration = _estimate_duration(scene.text)
        scenes_total += duration
        scenes.append(
            ScriptScene(
                line_number=i,
                text=scene.text,
                visual_cue=scene.visual_cue,
                est_duration_seconds=duration,
            )
        )

    total = round(hook_duration + scenes_total + cta_duration, 1)

    return GeneratedScript(
        idea_id=idea.id,
        hook=draft.hook,
        hook_duration_seconds=hook_duration,
        scenes=scenes,
        cta=draft.cta,
        cta_duration_seconds=cta_duration,
        estimated_duration_seconds=total,
        target_duration_seconds=config.script.target_duration_seconds,
    )


def save_script(config: ChannelConfig, script: GeneratedScript) -> int:
    """Persist the script and advance the source idea to status='scripted'.
    Returns the new script's DB id."""
    with get_session() as session:
        row = Script(
            idea_id=script.idea_id,
            channel_id=config.channel_id,
            hook=script.hook,
            scenes_json=json.dumps([s.model_dump() for s in script.scenes]),
            cta=script.cta,
            estimated_duration_seconds=script.estimated_duration_seconds,
            target_duration_seconds=script.target_duration_seconds,
            status="pending_review",
        )
        session.add(row)

        idea = session.get(Idea, script.idea_id)
        if idea is not None:
            idea.status = "scripted"

        session.commit()
        session.refresh(row)
        return row.id


def run_scripting(config: ChannelConfig, idea_id: int) -> tuple[GeneratedScript, int]:
    """Full Phase 2 flow: load idea -> generate -> save. Returns (script, db_id)."""
    with get_session() as session:
        idea = session.get(Idea, idea_id)
        if idea is None or idea.channel_id != config.channel_id:
            raise RuntimeError(f"No idea with id={idea_id} for channel {config.channel_id!r}")
        session.expunge(idea)

    script = generate_script(config, idea)
    script_id = save_script(config, script)
    return script, script_id
