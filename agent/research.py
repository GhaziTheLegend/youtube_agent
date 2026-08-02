"""
Research module - Phase 1.

Flow: ChannelConfig -> prompt -> LLM -> validated list[VideoIdea] -> saved to
DB with status="pending_review" -> printed for YOU to approve (Phase 1 stays
manual; Phase 6 later wires this into a scheduler, but the human checkpoint
here should stay even then, per the roadmap).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from agent.config import ChannelConfig
from agent.db.models import Idea
from agent.db.session import get_session
from agent.llm import generate
from agent.utils import strip_code_fences


# ---------------------------------------------------------------------------
# Structured output schema - what we ask the LLM for, and what we validate
# its response against before trusting it enough to write to the database.
# ---------------------------------------------------------------------------

class VideoIdea(BaseModel):
    title: str
    hook: str = Field(..., description="The first line/visual that grabs attention")
    angle: str = Field(..., description="The distinct opinion/structure/insight - not just facts")
    why_now: str = Field(..., description="Why this topic is timely/relevant right now")
    search_interest: str = Field(..., pattern="^(High|Medium|Low)$")
    competition: str = Field(..., pattern="^(High|Medium|Low)$")


PROMPT_TEMPLATE = """You are a content strategist for a faceless YouTube Shorts channel.

CHANNEL
- Niche: {niche}
- Content pillars: {content_pillars}
- Target audience: {target_audience}
- Tone: {tone}

TASK
Generate {count} video ideas for this channel's next batch of Shorts.

CRITICAL REQUIREMENT: Each idea must have a distinct angle, opinion, or structural
hook - NOT just a restatement of facts. YouTube's monetization policy for faceless
channels specifically penalizes content that just repeats information with no
unique insight, so "why_now" and "angle" must describe something genuinely
distinctive about how THIS video would cover the topic, not just what the topic is.

{seed_keywords_block}

OUTPUT FORMAT
Return ONLY a JSON array (no markdown fences, no commentary) of exactly {count}
objects, each with these exact keys:
- "title": string
- "hook": string (the opening line or visual)
- "angle": string (the distinct take/structure/opinion)
- "why_now": string (timeliness/relevance)
- "search_interest": one of "High", "Medium", "Low"
- "competition": one of "High", "Medium", "Low"
"""


def _build_prompt(config: ChannelConfig) -> str:
    seed_keywords_block = ""
    if config.research.seed_keywords:
        keywords = ", ".join(config.research.seed_keywords)
        seed_keywords_block = f"Consider these seed keywords/topics: {keywords}"

    return PROMPT_TEMPLATE.format(
        niche=config.niche,
        content_pillars=", ".join(config.content_pillars),
        target_audience=config.target_audience,
        tone=config.script.tone,
        count=config.research.ideas_per_cycle,
        seed_keywords_block=seed_keywords_block,
    )


def generate_ideas(config: ChannelConfig) -> list[VideoIdea]:
    """Call the LLM and return a validated list of VideoIdea. Raises on failure
    rather than silently returning garbage - a research batch you can't trust
    is worse than no batch."""
    prompt = _build_prompt(config)
    raw = generate(prompt, json_mode=True)
    cleaned = strip_code_fences(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM did not return valid JSON. Raw response:\n{raw}"
        ) from e

    if not isinstance(parsed, list):
        raise RuntimeError(f"Expected a JSON array, got: {type(parsed)}")

    ideas: list[VideoIdea] = []
    errors: list[str] = []
    for i, item in enumerate(parsed):
        try:
            ideas.append(VideoIdea.model_validate(item))
        except ValidationError as e:
            errors.append(f"Item {i}: {e}")

    if errors:
        # Don't fail the whole batch for one bad item - just tell the user
        # what got dropped. If everything failed, that's still an error.
        print(f"⚠️  {len(errors)} idea(s) failed validation and were skipped:")
        for err in errors:
            print(f"   - {err}")

    if not ideas:
        raise RuntimeError("No valid ideas were returned by the LLM.")

    return ideas


def save_ideas(config: ChannelConfig, ideas: list[VideoIdea]) -> list[int]:
    """Persist ideas to the DB with status=pending_review. Returns their IDs."""
    saved_ids: list[int] = []
    with get_session() as session:
        for idea in ideas:
            row = Idea(
                channel_id=config.channel_id,
                title=idea.title,
                hook=idea.hook,
                angle=idea.angle,
                why_now=idea.why_now,
                search_interest=idea.search_interest,
                competition=idea.competition,
                status="pending_review",
            )
            session.add(row)
            session.flush()  # populate row.id without committing yet
            saved_ids.append(row.id)
        session.commit()
    return saved_ids


def run_research(config: ChannelConfig) -> list[Idea]:
    """Full Phase 1 flow: generate -> validate -> save -> return DB rows."""
    ideas = generate_ideas(config)
    ids = save_ideas(config, ideas)
    with get_session() as session:
        rows = session.query(Idea).filter(Idea.id.in_(ids)).all()
        session.expunge_all()  # detach so caller can use rows after session closes
    return rows
