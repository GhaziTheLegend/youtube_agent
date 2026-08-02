"""
Entry point for the agent.

Usage:
    python -m agent.cli show-config techreviews
    python -m agent.cli research techreviews
    python -m agent.cli list-ideas techreviews
    python -m agent.cli approve-idea 3
    python -m agent.cli script techreviews 3
    python -m agent.cli show-script 1
    python -m agent.cli list-scripts techreviews
    python -m agent.cli approve-script 1
"""

import json

import click

from agent.config import load_channel
from agent.db.models import Idea, Script
from agent.db.session import get_session


@click.group()
def cli():
    """YouTube Shorts AI Agent."""
    pass


@cli.command("show-config")
@click.argument("channel_id")
def show_config(channel_id: str):
    """Load channels/<channel_id>.yaml, validate it, and print it."""
    config = load_channel(channel_id)
    click.echo(f"✅ Loaded and validated config for '{config.channel_name}'\n")
    click.echo(config.model_dump_json(indent=2))


@cli.command("research")
@click.argument("channel_id")
def research(channel_id: str):
    """Generate a fresh batch of video ideas for a channel and save them
    to the database with status=pending_review."""
    from agent.research import run_research  # imported here so `show-config`
    # (and anyone without an LLM key yet) doesn't need llm.py to import cleanly

    config = load_channel(channel_id)
    click.echo(f"🔎 Generating {config.research.ideas_per_cycle} ideas for '{config.channel_name}'...\n")
    rows = run_research(config)

    for row in rows:
        click.echo(f"[{row.id}] {row.title}")
        click.echo(f"     Hook:        {row.hook}")
        click.echo(f"     Angle:       {row.angle}")
        click.echo(f"     Why now:     {row.why_now}")
        click.echo(f"     Interest/Competition: {row.search_interest} / {row.competition}")
        click.echo()

    click.echo(f"✅ Saved {len(rows)} ideas. Review them, then:")
    click.echo(f"   python -m agent.cli approve-idea <id>")


@cli.command("list-ideas")
@click.argument("channel_id")
@click.option("--status", default="pending_review", help="Filter by status")
def list_ideas(channel_id: str, status: str):
    """List saved ideas for a channel, filtered by status (default: pending_review)."""
    with get_session() as session:
        rows = (
            session.query(Idea)
            .filter(Idea.channel_id == channel_id, Idea.status == status)
            .order_by(Idea.created_at.desc())
            .all()
        )
        if not rows:
            click.echo(f"No ideas found with status={status!r} for channel {channel_id!r}.")
            return
        for row in rows:
            click.echo(f"[{row.id}] ({row.status}) {row.title}")


@cli.command("approve-idea")
@click.argument("idea_id", type=int)
def approve_idea(idea_id: int):
    """Mark an idea as approved (this is the human checkpoint from the roadmap -
    Phase 2 scripting should only ever run against approved ideas)."""
    with get_session() as session:
        idea = session.get(Idea, idea_id)
        if idea is None:
            click.echo(f"No idea with id={idea_id}")
            return
        idea.status = "approved"
        session.commit()
        click.echo(f"✅ Idea {idea_id} approved: {idea.title}")


def _print_script(hook: str, scenes: list[dict], cta: str,
                   estimated: float, target: float) -> None:
    click.echo(f"HOOK: {hook}\n")
    for scene in scenes:
        click.echo(f"  [{scene['line_number']}] {scene['text']}")
        click.echo(f"      🎬 {scene['visual_cue']}  (~{scene['est_duration_seconds']}s)")
    click.echo(f"\nCTA: {cta}\n")

    ratio = estimated / target if target else 1.0
    flag = "✅" if 0.75 <= ratio <= 1.25 else "⚠️ "
    click.echo(f"{flag} Estimated duration: {estimated}s (target: {target}s)")


@cli.command("script")
@click.argument("channel_id")
@click.argument("idea_id", type=int)
def script(channel_id: str, idea_id: int):
    """Generate a full script from an approved idea."""
    from agent.scripting import run_scripting  # deferred import, same reasoning as `research`

    config = load_channel(channel_id)
    try:
        generated, script_id = run_scripting(config, idea_id)
    except RuntimeError as e:
        click.echo(f"❌ {e}")
        return

    click.echo(f"✅ Script #{script_id} generated for idea {idea_id}\n")
    _print_script(
        generated.hook,
        [s.model_dump() for s in generated.scenes],
        generated.cta,
        generated.estimated_duration_seconds,
        generated.target_duration_seconds,
    )
    click.echo(f"\nReview it, then: python -m agent.cli approve-script {script_id}")


@cli.command("show-script")
@click.argument("script_id", type=int)
def show_script(script_id: int):
    """Print a previously generated script."""
    with get_session() as session:
        row = session.get(Script, script_id)
        if row is None:
            click.echo(f"No script with id={script_id}")
            return
        scenes = json.loads(row.scenes_json)
        _print_script(row.hook, scenes, row.cta, row.estimated_duration_seconds, row.target_duration_seconds)


@cli.command("list-scripts")
@click.argument("channel_id")
@click.option("--status", default=None, help="Filter by status (default: all)")
def list_scripts(channel_id: str, status: str | None):
    """List scripts for a channel, optionally filtered by status."""
    with get_session() as session:
        query = session.query(Script).filter(Script.channel_id == channel_id)
        if status:
            query = query.filter(Script.status == status)
        rows = query.order_by(Script.created_at.desc()).all()
        if not rows:
            click.echo(f"No scripts found for channel {channel_id!r}" + (f" with status={status!r}" if status else ""))
            return
        for row in rows:
            click.echo(f"[{row.id}] (idea {row.idea_id}, {row.status}) ~{row.estimated_duration_seconds}s - {row.hook[:60]}")


@cli.command("approve-script")
@click.argument("script_id", type=int)
def approve_script(script_id: int):
    """Mark a script as approved (human checkpoint 2 - Phase 3 production
    should only ever run against approved scripts)."""
    with get_session() as session:
        row = session.get(Script, script_id)
        if row is None:
            click.echo(f"No script with id={script_id}")
            return
        row.status = "approved"
        session.commit()
        click.echo(f"✅ Script {script_id} approved.")


@cli.command("produce-voiceover")
@click.argument("channel_id")
@click.argument("script_id", type=int)
def produce_voiceover(channel_id: str, script_id: int):
    """Synthesize the voiceover audio for an approved script.
    Writes one mp3 per line to output/<channel>/script_<id>/audio/ and a
    timeline.json describing exact timing - visuals/assembly (next) read this."""
    import asyncio
    import json as json_module
    from pathlib import Path

    from agent.production.voiceover import build_timeline, synthesize_script, total_duration

    config = load_channel(channel_id)

    with get_session() as session:
        row = session.get(Script, script_id)
        if row is None or row.channel_id != channel_id:
            click.echo(f"❌ No script with id={script_id} for channel {channel_id!r}")
            return
        if row.status != "approved":
            click.echo(f"❌ Script {script_id} has status={row.status!r}, not 'approved'.")
            click.echo(f"   Run: python -m agent.cli approve-script {script_id}")
            return
        hook, cta = row.hook, row.cta
        scenes = [(s["line_number"], s["text"]) for s in json.loads(row.scenes_json)]

    output_dir = Path("output") / channel_id / f"script_{script_id}" / "audio"
    click.echo(f"🎙️  Synthesizing voiceover with voice '{config.production.voice_id}'...")

    clips = asyncio.run(
        synthesize_script(
            hook, scenes, cta, config.production.voice_id, output_dir,
            provider=config.production.tts_provider,
        )
    )
    timeline = build_timeline(clips)

    timeline_path = output_dir.parent / "timeline.json"
    timeline_path.write_text(json_module.dumps(timeline, indent=2))

    for entry in timeline:
        click.echo(f"  [{entry['label']:>10}] {entry['start']:>5.1f}s - {entry['end']:>5.1f}s  \"{entry['text'][:50]}\"")

    click.echo(f"\n✅ Total duration: {total_duration(clips)}s")
    click.echo(f"   Audio files: {output_dir}/")
    click.echo(f"   Timeline:    {timeline_path}")


@cli.command("download-voice")
@click.argument("voice")
@click.option("--download-dir", default="models/piper", help="Where to save the voice model")
@click.option("--max-retries", default=5, help="Retries per file on truncated/failed download")
def download_voice_cmd(voice: str, download_dir: str, max_retries: int):
    """Download a Piper voice model with integrity verification. Use this
    instead of `piper.download_voices` directly if your network silently
    truncates long downloads (verified against Content-Length, retries on
    mismatch) - e.g. python -m agent.cli download-voice en_US-lessac-medium"""
    from pathlib import Path

    from agent.production.tts_downloader import download_voice

    try:
        download_voice(voice, Path(download_dir), max_retries=max_retries)
        click.echo(f"\n✅ Voice '{voice}' ready in {download_dir}/")
    except Exception as e:
        click.echo(f"\n❌ {e}")


if __name__ == "__main__":
    cli()

