"""
Channel configuration schema.

This is the machine-readable version of the "fill-in-the-blank" template we
designed earlier. Every channel gets one YAML file under channels/, and this
module loads + validates it into a typed Python object every other module
can rely on.

Why pydantic instead of just `yaml.safe_load()` into a dict?
- Typos/missing fields fail LOUDLY at startup, not silently at 2am mid-pipeline.
- Every other module gets autocomplete + type checking on `config.niche` etc.
  instead of `config["niche"]` (which can KeyError anywhere).
- This is the same pattern you'll see in real production systems: validate
  external input (config files, API responses) at the boundary, trust it
  everywhere after that.
"""

from __future__ import annotations

from datetime import time
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums - constrain fields to known-valid values instead of free-text strings
# ---------------------------------------------------------------------------

class VisualsSource(str, Enum):
    STOCK_FOOTAGE = "stock_footage"
    AI_GENERATED = "ai_generated"
    SCREEN_RECORDING = "screen_recording"


class TTSProvider(str, Enum):
    EDGE_TTS = "edge_tts"       # free, network-dependent, gives word timing natively
    PIPER = "piper"              # free, fully local/offline, needs faster-whisper for word timing
    ELEVENLABS = "elevenlabs"   # paid, higher quality


class Visibility(str, Enum):
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"


# ---------------------------------------------------------------------------
# Sub-sections (each maps to a section of the original template document)
# ---------------------------------------------------------------------------

class UploadSlot(BaseModel):
    """One recurring upload time, e.g. Monday 18:00 in Asia/Karachi."""
    day: str = Field(..., description="Monday, Tuesday, ... or 'daily'")
    time: time
    timezone: str = "Asia/Karachi"

    @field_validator("day")
    @classmethod
    def validate_day(cls, v: str) -> str:
        valid = {"monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday", "daily"}
        if v.lower() not in valid:
            raise ValueError(f"day must be one of {valid}, got {v!r}")
        return v.lower()


class ResearchConfig(BaseModel):
    ideas_per_cycle: int = Field(default=8, ge=1, le=30)
    seed_keywords: list[str] = Field(default_factory=list)
    sources: list[str] = Field(
        default_factory=lambda: ["llm_synthesis"],
        description="e.g. llm_synthesis, google_trends, reddit, news_rss",
    )


class ScriptConfig(BaseModel):
    target_duration_seconds: int = Field(default=45, ge=15, le=180)
    tone: str = "energetic, punchy, direct"
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    original_angle_required: bool = Field(
        default=True,
        description=(
            "If True, the scripting prompt MUST inject a distinct angle/"
            "opinion/structure, not just recite facts. Keep this True - "
            "see YouTube's 2026 inauthentic-content policy notes in the "
            "project roadmap."
        ),
    )


class ProductionConfig(BaseModel):
    voice_id: str = Field(..., description="Voice name/ID for the TTS provider")
    tts_provider: TTSProvider = TTSProvider.EDGE_TTS
    visuals_source: VisualsSource = VisualsSource.STOCK_FOOTAGE
    aspect_ratio: str = "9:16"
    captions_enabled: bool = True
    caption_style: str = "bold, centered, word-by-word highlight"
    background_music_volume_db: float = Field(
        default=-20.0, description="Relative to voiceover; keep voice dominant"
    )


class MetadataConfig(BaseModel):
    title_formula: str = "[Hook] | [Keyword]"
    tags_strategy: list[str] = Field(default_factory=list)
    category: str = "22"  # YouTube category ID, default "People & Blogs"


class PublishingConfig(BaseModel):
    schedule: list[UploadSlot] = Field(default_factory=list)
    default_visibility: Visibility = Visibility.PRIVATE
    require_manual_approval: bool = Field(
        default=True,
        description="Keep True until you trust pipeline output quality (see roadmap Phase 4).",
    )


class ChannelGoals(BaseModel):
    target_subscribers_90d: int | None = None
    target_avg_views_per_short: int | None = None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class ChannelConfig(BaseModel):
    channel_id: str = Field(..., description="Short slug, e.g. 'techreviews' - used as DB key")
    channel_name: str
    niche: str
    content_pillars: list[str] = Field(
        ..., min_length=1, description="3-5 recurring content buckets"
    )
    target_audience: str
    language: str = "en"

    research: ResearchConfig = Field(default_factory=ResearchConfig)
    script: ScriptConfig = Field(default_factory=ScriptConfig)
    production: ProductionConfig
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    publishing: PublishingConfig = Field(default_factory=PublishingConfig)
    goals: ChannelGoals = Field(default_factory=ChannelGoals)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChannelConfig":
        """Load and validate a channel_config.yaml file into a ChannelConfig."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No channel config found at {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


def load_channel(channel_id: str, channels_dir: str | Path = "channels") -> ChannelConfig:
    """Convenience loader: load_channel('techreviews') -> channels/techreviews.yaml"""
    path = Path(channels_dir) / f"{channel_id}.yaml"
    return ChannelConfig.from_yaml(path)
