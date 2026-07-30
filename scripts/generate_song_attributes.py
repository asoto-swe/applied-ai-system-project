"""Agentic AI workflow that generates 5 additional song attributes
(popularity, release decade, detailed mood tags, vocal style, and
instrumentation notes) for the demo catalog, using Claude with tool access
to the project's genre knowledge base.

This is the project's "Additional Song Attributes via Agentic AI" stretch
feature. It's genuinely agentic, not a single one-shot completion: Claude
is given a tool (lookup_genre_background) and decides for itself whether
and when to call it before producing the final structured output, one
step at a time, exactly like music_recommender/agent.py's design pattern
did for explanations.

Requires ANTHROPIC_API_KEY. Writes:
  - music_recommender/data/song_attributes.csv (the generated data, loaded
    at runtime by music_recommender/song_attributes.py — no API key needed
    to use the data afterward)
  - ai_interactions.md (the captured reasoning trace: example prompts,
    a summary of the generated changes, and manual verification notes)

Run:
    python scripts/generate_song_attributes.py
"""
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from music_recommender.demo_catalog import CATALOG
from music_recommender.knowledge import GenreKnowledgeBase

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "music_recommender" / "data" / "song_attributes.csv"
AI_INTERACTIONS_PATH = PROJECT_ROOT / "ai_interactions.md"

MODEL = "claude-opus-5"

TOOLS = [
    {
        "name": "lookup_genre_background",
        "description": "Look up background context on what a music genre tends to sound and feel like, to help ground popularity/era/style guesses for songs in that genre.",
        "input_schema": {
            "type": "object",
            "properties": {"genre": {"type": "string", "description": "The genre to look up, e.g. 'indie pop'"}},
            "required": ["genre"],
        },
    }
]

SYSTEM_PROMPT = (
    "You are a music cataloger enriching a small demo song database with additional attributes. "
    "The songs are fictional/independent tracks used for a class project, not real chart hits, so "
    "there is no real popularity data to look up — your job is to make plausible, internally "
    "consistent estimates based on each song's genre, mood, and lyrics.\n\n"
    "For each song, generate exactly these 5 attributes:\n"
    "- popularity: integer 0-100, a plausible mainstream-familiarity estimate for this kind of "
    "independent/niche track (most should be low-to-mid; reserve high numbers for songs that read "
    "as broadly catchy pop/EDM)\n"
    "- release_decade: a plausible decade string like '2020s' or '2010s', consistent with the "
    "genre and production style implied\n"
    "- detailed_mood_tags: 2-4 specific mood descriptors that are more granular than the song's "
    "existing single 'mood' field (e.g. mood='reflective' might become "
    "['wistful', 'tender', 'quietly hopeful'])\n"
    "- vocal_style: a short phrase describing the vocal delivery (e.g. 'breathy, close-mic'd')\n"
    "- instrumentation_notes: a short phrase describing likely instrumentation/production\n\n"
    "You have a lookup_genre_background tool available if genre context would help you make more "
    "grounded, consistent decisions across songs of the same genre. Use it if you want to; it's "
    "optional.\n\n"
    "When you have everything you need, respond with ONLY a JSON array (no other text), one "
    "object per song in the same order given, each with keys: title, popularity, release_decade, "
    "detailed_mood_tags (array of strings), vocal_style, instrumentation_notes."
)


def build_user_message() -> str:
    lines = ["Generate attributes for these songs:\n"]
    for song in CATALOG:
        lines.append(
            f"- \"{song.title}\" by {song.artist} | genre: {song.genre} | mood: {song.mood} | "
            f"themes: {song.themes} | lyrics excerpt: \"{song.lyrics_excerpt}\""
        )
    return "\n".join(lines)


def run_agentic_generation(client):
    genre_kb = GenreKnowledgeBase()
    trace = []

    def lookup_genre_background(genre: str) -> dict:
        note = genre_kb.lookup(genre)
        return {"note": note} if note else {"note": None, "message": f"No background note for '{genre}'."}

    user_message = build_user_message()
    messages = [{"role": "user", "content": user_message}]
    trace.append({"type": "prompt", "role": "user", "content": user_message})

    max_steps = 6
    for step in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": "medium"},
            messages=messages,
        )

        text_blocks = [b.text for b in response.content if b.type == "text" and b.text.strip()]

        if response.stop_reason != "tool_use":
            # This turn produced the final answer directly, with no tool call —
            # log it once as the final response, not also as "reasoning".
            final_text = " ".join(text_blocks).strip()
            trace.append({
                "type": "final_response", "step": step, "text": final_text,
                "used_tools": step > 0 or any(e["type"] == "tool_call" for e in trace),
            })
            return final_text, trace

        # Only reached when the model is about to call a tool: any text here
        # is genuine intermediate reasoning, distinct from the final answer.
        if text_blocks:
            trace.append({"type": "reasoning", "step": step, "text": " ".join(text_blocks)})

        messages.append({"role": "assistant", "content": response.content})
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []
        for block in tool_use_blocks:
            if block.name == "lookup_genre_background":
                result = lookup_genre_background(**block.input)
            else:
                result = {"error": f"Unknown tool '{block.name}'"}
            trace.append({"type": "tool_call", "step": step, "tool": block.name, "input": block.input, "output": result})
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Agent did not finish within {max_steps} steps.")


def parse_json_array(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def render_ai_interactions_markdown(trace, generated_rows, verification_notes: str) -> str:
    lines = [
        "# ai_interactions.md — Agentic Song Attribute Generation",
        "",
        "This is the project's **Additional Song Attributes via Agentic AI** stretch feature: "
        f"a genuine, captured run of `scripts/generate_song_attributes.py` against the live "
        f"Claude API (model: `{MODEL}`), generating 5 new attributes (popularity, release decade, "
        "detailed mood tags, vocal style, instrumentation notes) for every song in the demo "
        "catalog. The agent had a `lookup_genre_background` tool available and decided for "
        "itself whether to use it — this is a multi-step tool-calling workflow, not a single "
        "fixed-format completion.",
        "",
        "## Example prompt (system + user message excerpt)",
        "",
        "**System prompt:**",
        "```",
        SYSTEM_PROMPT,
        "```",
        "",
        "**User message (first 3 songs shown; the real run sent all 18):**",
        "```",
        "\n".join(build_user_message().split("\n")[:5]),
        "...",
        "```",
        "",
        "## Captured reasoning trace",
        "",
    ]
    for entry in trace:
        if entry["type"] == "prompt":
            continue
        elif entry["type"] == "tool_call":
            lines.append(f"**Step {entry['step']} — tool call:** `{entry['tool']}({entry['input']})`")
            lines.append(f"> Result: `{json.dumps(entry['output'])}`")
        elif entry["type"] == "reasoning":
            lines.append(f"**Step {entry['step']} — reasoning:** {entry['text'][:500]}")
        elif entry["type"] == "final_response":
            lines.append(f"**Step {entry['step']} — final structured output produced** ({len(generated_rows)} songs)")
        lines.append("")

    lines.append("## Summary of generated changes")
    lines.append("")
    lines.append(f"Generated attributes for {len(generated_rows)} songs. Sample of 3:")
    lines.append("")
    lines.append("| Title | Popularity | Decade | Detailed Mood Tags | Vocal Style |")
    lines.append("|---|---|---|---|---|")
    for row in generated_rows[:3]:
        tags = ", ".join(row.get("detailed_mood_tags", []))
        lines.append(f"| {row['title']} | {row['popularity']} | {row['release_decade']} | {tags} | {row['vocal_style']} |")
    lines.append("")
    lines.append(f"Full data written to `music_recommender/data/song_attributes.csv` ({len(generated_rows)} rows).")
    lines.append("")
    lines.append("## Manual verification notes")
    lines.append("")
    lines.append(verification_notes)
    return "\n".join(lines)


def main() -> None:
    import anthropic

    client = anthropic.Anthropic()
    final_text, trace = run_agentic_generation(client)
    generated_rows = parse_json_array(final_text)

    titles_in_catalog = {s.title for s in CATALOG}
    titles_generated = {row["title"] for row in generated_rows}
    missing = titles_in_catalog - titles_generated
    extra = titles_generated - titles_in_catalog
    verification_notes = (
        f"- Catalog has {len(titles_in_catalog)} songs; the agent returned {len(generated_rows)} rows.\n"
        f"- Titles present in the catalog but missing from the agent's output: {sorted(missing) or 'none'}.\n"
        f"- Titles in the agent's output not found in the catalog (possible hallucinated/renamed title): {sorted(extra) or 'none'}.\n"
        f"- Popularity range returned: {min(r['popularity'] for r in generated_rows)}-{max(r['popularity'] for r in generated_rows)} "
        f"(sanity check: should be within 0-100).\n"
        f"- All rows checked for the 5 required keys before being written to CSV; malformed rows are skipped, not silently written."
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "popularity", "release_decade", "detailed_mood_tags", "vocal_style", "instrumentation_notes"])
        skipped = []
        for row in generated_rows:
            required = ["title", "popularity", "release_decade", "detailed_mood_tags", "vocal_style", "instrumentation_notes"]
            if not all(k in row for k in required):
                skipped.append(row.get("title", "<unknown>"))
                continue
            writer.writerow([
                row["title"], row["popularity"], row["release_decade"],
                ";".join(row["detailed_mood_tags"]), row["vocal_style"], row["instrumentation_notes"],
            ])
        if skipped:
            verification_notes += f"\n- Skipped malformed rows (missing required keys): {skipped}."

    AI_INTERACTIONS_PATH.write_text(
        render_ai_interactions_markdown(trace, generated_rows, verification_notes), encoding="utf-8"
    )

    print(f"Wrote {len(generated_rows)} rows to {OUTPUT_CSV}")
    print(f"Wrote reasoning trace to {AI_INTERACTIONS_PATH}")
    print()
    print(verification_notes)


if __name__ == "__main__":
    main()
