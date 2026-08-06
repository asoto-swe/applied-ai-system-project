"""Streamlit UI for TuneMatch.

A thin front end over the same pipeline `app.py` drives from the CLI —
`TasteProfile` in, `MusicRecommender.recommend(...)` out. No recommendation
logic lives here; this only collects input, calls the existing package, and
renders the result. Styling lives entirely in the CSS block below and in
`.streamlit/config.toml`.

Run with: streamlit run ui.py
"""
import html
import os

import streamlit as st

from music_recommender.data import TasteProfile
from music_recommender.demo_catalog import CATALOG
from music_recommender.ranking_strategies import STRATEGIES, get_strategy
from music_recommender.recommender import MusicRecommender

st.set_page_config(page_title="TuneMatch", page_icon="\U0001F3B5", layout="centered")

MODE_DESCRIPTIONS = {
    "balanced": "Blends tags, semantic meaning, and predicted sound evenly.",
    "genre-first": "Weighs an exact genre match above everything else.",
    "mood-first": "Weighs how a song feels over its genre label.",
    "energy-similarity": "Weighs predicted energy/valence/acousticness fit most.",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(129, 140, 248, 0.16), transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(94, 234, 212, 0.10), transparent 40%),
        linear-gradient(160deg, #0b1023 0%, #10152e 45%, #1b1140 100%);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] > .main { padding-top: 1rem; }

/* Hero header */
.tm-hero { text-align: center; padding: 0.5rem 0 1.25rem 0; }
.tm-eq { display: flex; justify-content: center; gap: 5px; height: 30px; align-items: flex-end; margin-bottom: 0.6rem; }
.tm-eq span {
    width: 5px; border-radius: 3px;
    background: linear-gradient(180deg, #5eead4, #818cf8);
    animation: tm-bounce 1.1s ease-in-out infinite;
}
.tm-eq span:nth-child(1) { height: 10px; animation-delay: -1.0s; }
.tm-eq span:nth-child(2) { height: 22px; animation-delay: -0.8s; }
.tm-eq span:nth-child(3) { height: 14px; animation-delay: -0.6s; }
.tm-eq span:nth-child(4) { height: 28px; animation-delay: -0.4s; }
.tm-eq span:nth-child(5) { height: 8px;  animation-delay: -0.2s; }
.tm-eq span:nth-child(6) { height: 20px; animation-delay: 0s; }
.tm-eq span:nth-child(7) { height: 12px; animation-delay: -0.5s; }
@keyframes tm-bounce {
    0%, 100% { transform: scaleY(0.4); opacity: 0.7; }
    50% { transform: scaleY(1); opacity: 1; }
}
.tm-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #a5f3ea, #a78bfa 60%, #60a5fa);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    letter-spacing: -0.02em;
}
.tm-tagline { color: #9aa3c7; font-size: 0.95rem; margin-top: 0.35rem; }

/* Status pill */
.tm-status {
    border-radius: 999px; padding: 0.5rem 1.1rem; font-size: 0.85rem;
    border: 1px solid rgba(148, 163, 253, 0.25);
    background: rgba(255, 255, 255, 0.04);
    color: #c3caed; text-align: center; margin-bottom: 1.3rem;
}
.tm-status b { color: #5eead4; }

/* Form panel */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(148, 163, 253, 0.18);
    border-radius: 20px;
    padding: 1.6rem 1.6rem 1rem 1.6rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(10, 6, 40, 0.35);
}
div[data-testid="stTextInput"] input, div[data-baseweb="select"] > div {
    background: rgba(11, 16, 35, 0.65) !important;
    border: 1px solid rgba(148, 163, 253, 0.25) !important;
    border-radius: 10px !important;
    color: #e6e9f5 !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #5eead4 !important;
    box-shadow: 0 0 0 1px #5eead4 !important;
}
label[data-testid="stWidgetLabel"] p { color: #b9c0e6 !important; font-weight: 500; font-size: 0.88rem; }

[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #5eead4, #818cf8);
    color: #0b1023; border: none; border-radius: 999px;
    font-weight: 600; letter-spacing: 0.02em; padding: 0.55rem 1.6rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 4px 18px rgba(94, 234, 212, 0.25);
}
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 22px rgba(129, 140, 248, 0.35);
}
[data-testid="stFormSubmitButton"] button p { color: #0b1023 !important; }

/* Recommendation cards */
.tm-results-heading {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600;
    color: #e6e9f5; margin: 1.6rem 0 0.9rem 0; font-size: 1.15rem;
}
.tm-card {
    position: relative; overflow: hidden;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(148, 163, 253, 0.16);
    border-radius: 16px; padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    backdrop-filter: blur(8px);
}
.tm-card::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, #5eead4, #818cf8);
}
.tm-card-row { display: flex; align-items: center; gap: 0.9rem; }
.tm-rank {
    font-family: 'Space Grotesk', sans-serif; font-size: 0.8rem; color: #6f7bab;
    min-width: 1.5rem;
}
.tm-card-title { font-weight: 600; color: #f1f3fb; font-size: 1.02rem; }
.tm-card-meta { color: #9aa3c7; font-size: 0.85rem; margin-top: 0.1rem; }
.tm-score-wrap { margin-left: auto; display: flex; flex-direction: column; align-items: flex-end; gap: 0.3rem; }
.tm-score-label {
    font-size: 0.6rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #8891c4;
}
.tm-score {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 0.95rem; color: #0b1023;
    background: linear-gradient(90deg, #5eead4, #a78bfa);
    border-radius: 999px; padding: 0.25rem 0.75rem; white-space: nowrap;
}
.tm-meter-track {
    width: 100%; height: 6px; border-radius: 999px;
    background: rgba(148, 163, 253, 0.14); margin-top: 0.8rem; overflow: hidden;
}
.tm-meter-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #5eead4, #a78bfa); }
.tm-explanation { color: #c3caed; font-size: 0.9rem; line-height: 1.55; margin-top: 0.6rem; }
.tm-score-note { color: #8891c4; font-size: 0.82rem; line-height: 1.5; margin: -0.3rem 0 0.9rem 0; }

[data-testid="stExpander"] {
    border: 1px solid rgba(148, 163, 253, 0.16) !important;
    border-radius: 14px !important;
    background: rgba(255, 255, 255, 0.02);
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_recommender(mode: str) -> MusicRecommender:
    return MusicRecommender(ranking_strategy=get_strategy(mode))


def split_csv(raw: str) -> list:
    return [item.strip() for item in raw.split(",") if item.strip()]


st.markdown(
    """
    <div class="tm-hero">
        <div class="tm-eq"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
        <p class="tm-title">TuneMatch</p>
        <p class="tm-tagline">find the songs that match your vibe, not just your tags</p>
    </div>
    """,
    unsafe_allow_html=True,
)

live_voyage = bool(os.environ.get("VOYAGE_API_KEY"))
live_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
if live_voyage and live_claude:
    status_html = "<b>Live mode</b> &mdash; semantic retrieval (Voyage) and AI-generated explanations (Claude) are active."
else:
    missing = [k for k, live in (("VOYAGE_API_KEY", live_voyage), ("ANTHROPIC_API_KEY", live_claude)) if not live]
    status_html = f"<b>Chill fallback mode</b> &mdash; {', '.join(missing)} not set. Tag matching and template explanations still fully work."
st.markdown(f'<div class="tm-status">{status_html}</div>', unsafe_allow_html=True)

with st.form("taste_profile"):
    name = st.text_input("Your Name", value="Alex")

    col1, col2 = st.columns(2)
    with col1:
        genres = st.text_input("Genres You Like", value="rock,lofi", help="Comma-separated, e.g. 'hip hop,indie pop'.")
        themes = st.text_input("Themes You Like", value="", help="Comma-separated, e.g. 'heartbreak,nostalgia'.")
    with col2:
        moods = st.text_input("Moods You're After", value="happy,intense", help="Comma-separated, e.g. 'chill,reflective'.")
        artists = st.text_input("Favorite Artists", value="", help="Comma-separated, e.g. 'The xx,Phoebe Bridgers'.")

    col3, col4 = st.columns(2)
    with col3:
        mode = st.selectbox("Ranking Mode", options=sorted(STRATEGIES), index=sorted(STRATEGIES).index("balanced"))
        st.caption(MODE_DESCRIPTIONS.get(mode, ""))
    with col4:
        limit = st.slider("Number of Recommendations", min_value=1, max_value=8, value=3)

    submitted = st.form_submit_button("Find My Sound \U0001F3A7")

if submitted:
    profile = TasteProfile(
        name=name or "Listener",
        preferred_genres=split_csv(genres),
        preferred_moods=split_csv(moods),
        preferred_themes=split_csv(themes),
        favorite_artists=split_csv(artists),
    )

    try:
        recommender = get_recommender(mode)
        with st.spinner("Retrieving, ranking, and explaining..."):
            recommendations = recommender.recommend(profile, CATALOG, limit=limit)
    except ValueError as exc:
        st.error(str(exc))
    else:
        if not recommendations:
            st.info("No strong matches for that taste profile — try broadening your genres or moods.")
        else:
            st.markdown(
                f'<p class="tm-results-heading">Recommendations for {html.escape(profile.name)} '
                f'&middot; ‘{mode}’ mode</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="tm-score-note">Higher match score = stronger fit. Scores are relative to this '
                'list, not a fixed 0&ndash;10 or percentage scale (and the underlying scale shifts by ranking '
                'mode) &mdash; the bar under each score shows how that pick compares to your top match here.</p>',
                unsafe_allow_html=True,
            )
            top_score = max(rec.score for rec in recommendations)
            for rank, rec in enumerate(recommendations, start=1):
                meter_pct = max(4, min(100, round(rec.score / top_score * 100)))
                st.markdown(
                    f"""
                    <div class="tm-card">
                        <div class="tm-card-row">
                            <span class="tm-rank">{rank:02d}</span>
                            <div>
                                <div class="tm-card-title">{html.escape(rec.title)}</div>
                                <div class="tm-card-meta">{html.escape(rec.artist)} &middot; {html.escape(rec.genre)}</div>
                            </div>
                            <div class="tm-score-wrap">
                                <span class="tm-score-label">match score</span>
                                <span class="tm-score">{rec.score:.2f}</span>
                            </div>
                        </div>
                        <div class="tm-meter-track"><div class="tm-meter-fill" style="width: {meter_pct}%;"></div></div>
                        <div class="tm-explanation">{html.escape(rec.explanation)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with st.expander(f"\U0001F3B6 Browse the Demo Catalog ({len(CATALOG)} songs)"):
    st.dataframe(
        [
            {
                "Title": song.title,
                "Artist": song.artist,
                "Genre": song.genre,
                "Mood": song.mood,
                "Themes": ", ".join(song.themes),
            }
            for song in CATALOG
        ],
        hide_index=True,
        use_container_width=True,
    )
