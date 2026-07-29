# TuneMatch: A Hybrid RAG Music Recommender

## Original Project (Module 1-3)

This project extends **TuneMatch** (also referred to as *VibeFinder 1.0* in its model card), a Module 3 mini-project at `ai110-module3show-musicrecommendersimulation-starter`. The original TuneMatch was a deliberately non-ML, fully explainable recommender: a listener described their taste as a favorite genre, mood, target energy level, and an acoustic preference, and the system scored every song in a hand-built 17-song catalog against that profile using hand-tuned weighted rules (exact genre/mood match, energy closeness, an acousticness nudge) plus a diversity re-ranking penalty to avoid recommending five songs by the same artist. It returned a ranked top-5 list with a transparent, per-feature "why" for every pick, and its README/model card documented real biases found through adversarial stress-testing — most notably that its heaviest-weighted signal (energy) created a "safe middle" filter-bubble effect. The explicit design priority was explainability over predictive accuracy.

## Title and Summary

**TuneMatch** has been rebuilt into a full applied AI system that recommends music by *meaning*, not just by matching tags. Instead of exact-string genre/mood comparisons alone, it retrieves songs using a hybrid of categorical tag matching and semantic similarity (so a song can surface because its lyrics and themes feel like what you described, even if no tag matches literally), refines the ranking with a small trained model that predicts the kind of sound (energetic vs. mellow, upbeat vs. melancholic, acoustic vs. electronic) your stated taste implies, and has an LLM write a grounded, personalized explanation from that retrieved evidence instead of filling in a canned template.

It matters because this is the same problem real-world recommenders (Spotify, YouTube Music) solve, built at a scale where every design decision — what happens when a genre isn't in the catalog, when Claude is unavailable, when a listener's taste is confusing — is visible, testable, and explained rather than hidden inside a black box.

## Architecture Overview

The full system diagram is at [`diagrams/architecture.mmd`](diagrams/architecture.mmd) (Mermaid source; also duplicated at `diagrams/system_diagram.mmd`). In short:

- **Retriever** (`music_recommender/retriever.py`) — combines exact tag matching (genre/mood/theme/artist) with semantic similarity over song meaning, using Voyage AI embeddings. A song can be retrieved purely on semantic closeness, without any literal tag overlap.
- **Specialized Model** (`music_recommender/taste_model.py`) — a small `scikit-learn` model, trained offline on a curated dataset, that predicts a target audio profile (energy/valence/acousticness) from a listener's stated genres and moods. Its prediction becomes a third ranking signal.
- **Agent** (`music_recommender/recommender.py`, `MusicRecommender`) — orchestrates the Retriever and Specialized Model, blends their signals into a single score, and asks Claude to generate the final explanation grounded in that evidence (`music_recommender/explainer.py`).
- **Reliability Evaluator** — every one of the three AI dependencies above (Voyage embeddings, the trained model, Claude) is wrapped in a guardrail: if it fails or is unavailable, the system logs a warning and falls back to a deterministic alternative instead of crashing.
- **Tester** — an automated `pytest` suite (correctness) plus a dedicated reliability/consistency script (behavior) verify the Retriever, Specialized Model, and Agent independently, at dev time.
- **Human checkpoint** — the listener reviews the ranked recommendations and can accept or refine their stated taste, feeding back into the profile for the next request.

Data flows in one direction per request: **taste description → Taste Profile → Agent (delegates to Retriever + Specialized Model) → Claude explanation → ranked output → shown to the listener**, with a feedback loop back to the profile.

## Setup Instructions

1. **Clone the repo and enter the folder:**
   ```bash
   git clone <your-repo-url> tunematch
   cd tunematch
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Enable the live AI paths.** Without any keys set, the system runs entirely on its deterministic fallbacks (categorical tag matching, template explanations) — it is fully functional out of the box. To exercise the real semantic retrieval and LLM-generated explanations:
   ```bash
   export VOYAGE_API_KEY="your-voyage-key"       # free key at https://dash.voyageai.com
   export ANTHROPIC_API_KEY="your-anthropic-key" # from https://console.anthropic.com
   ```

5. **(Optional) Retrain the specialized model.** A trained artifact is already committed at `music_recommender/models/taste_affinity_model.joblib`. To regenerate it (and see its held-out evaluation score):
   ```bash
   python scripts/train_taste_model.py
   ```

6. **Run the demo:**
   ```bash
   python app.py
   ```

7. **Run the automated test suite:**
   ```bash
   python -m pytest -v
   ```

8. **Run the reliability report** (the project's "does the AI give consistent answers" evaluation — see Testing Summary below):
   ```bash
   python scripts/evaluate_reliability.py
   ```

## Sample Interactions

The following are real, unedited outputs captured from this codebase (no `VOYAGE_API_KEY` / `ANTHROPIC_API_KEY` set, so retrieval ran on categorical matching and the explanation used the template fallback — see Design Decisions for why that's still a meaningful demonstration). These use the 8-song catalog from `scripts/evaluate_reliability.py`, which sets explicit energy/valence/acousticness per song; running `python app.py` uses a smaller 2-song demo catalog with default (0.5/0.5/0.5) audio features, so its Maya score is a slightly different 8.43 for the same reason.

**1. A clean, multi-signal match**

Input: `TasteProfile(name='Maya', preferred_genres=['indie pop'], preferred_moods=['reflective'], preferred_themes=['nostalgia', 'solitude'], favorite_artists=['Phoebe Bridgers'])`

```
Garden Song by Phoebe Bridgers (score: 8.46)
Maya, this song is a good fit because it matches your interest in indie pop.
its reflective mood fits your taste. its themes of nostalgia, solitude align
with your preferences. you already like Phoebe Bridgers. its overall sound
(how energetic, upbeat, and acoustic it is) closely matches what your stated
taste predicts you'd enjoy. The lyrics excerpt "I miss the way we used to be"
also reinforce the connection.
```

**2. A conflicting/adversarial profile** — genre `metal` paired with mood `calm`, which don't co-occur in the training data

Input: `TasteProfile(name='Deshawn', preferred_genres=['metal'], preferred_moods=['calm'], preferred_themes=[], favorite_artists=[])`

```
Ironclad by Blackforge (score: 3.87)
Deshawn, this song is a good fit because it matches your interest in metal.
The lyrics excerpt "Forged in fire, I don't break" also reinforce the connection.
```

The system doesn't force a mood match that doesn't exist — it correctly falls back to the one real signal (genre) rather than inventing a connection.

**3. A genre absent from the catalog and training data** — `kpop`

Input: `TasteProfile(name='Priya', preferred_genres=['kpop'], preferred_moods=['happy'], preferred_themes=[], favorite_artists=[])`

```
Sunrise City by Neon Echo (score: 3.97)
Priya, this song is a good fit because its happy mood fits your taste.
The lyrics excerpt "Everything feels possible today" also reinforce the connection.
```

Again, no crash and no fabricated genre match — the system quietly falls back to the mood signal that actually exists.

## Design Decisions

- **Three independent AI signals, blended, not chained.** Categorical tag matching, Voyage semantic similarity, and the trained taste-affinity model are all computed and weighted independently (`SEMANTIC_WEIGHT = 3.0`, `AUDIO_WEIGHT = 1.5` in `recommender.py`), rather than having one gate the others. Trade-off: more moving parts and more failure surface, but each one degrades on its own — a Voyage outage doesn't take down the taste-affinity model or Claude.
- **Voyage AI for embeddings, not a custom endpoint.** Anthropic has no first-party embeddings API; Voyage AI is Anthropic's own recommended provider. Trade-off: one more API key and dependency, isolated entirely inside `embeddings.py` so the rest of the system doesn't know or care which embedding provider is behind it.
- **The specialized model refines ranking, it doesn't gate retrieval.** A song's predicted audio-feature fit only affects *where it lands* in the results, never *whether it's retrieved at all*. Trade-off: this keeps retrieval behavior predictable and testable, at the cost of the specialized model being unable to surface a song purely on sound similarity if it has zero tag or semantic match.
- **A small, curated, synthetic training set (29 rows) for the specialized model**, not a scraped or licensed dataset. This was a deliberate scope decision — it makes the model reproducible and fast to retrain in seconds — at the honest cost of generalization: 29 hand-labeled (genre, mood) pairs cannot capture the real diversity of how those combinations sound. This is a demonstration of the *architecture*, not a production-grade model, and the README/model card say so directly rather than let a reviewer assume otherwise.
- **LLM-generated explanations with a template fallback, not LLM-only.** Grounding Claude's explanation in the retrieved evidence rather than hand-formatting a string is what actually makes this a RAG system — a retrieved fact that never reaches the model's output isn't "retrieval-augmented generation," it's just retrieval. The template fallback exists so the system is still fully functional, testable, and free to run without an Anthropic API key.
- **Every AI dependency fails the same way: log a warning, fall back, keep going.** Voyage embeddings, the taste-affinity model, and Claude explanations all use the identical guardrail shape (`try`/`except` + `logging.warning` + deterministic fallback). Trade-off: uniform, easy-to-audit failure handling, at the cost that a silently-expired API key looks the same as "no key was ever configured" unless someone reads the logs.

## Testing Summary

**What worked:** 15/15 `pytest` tests pass (`tests/test_recommender.py`, `test_retriever.py`, `test_taste_model.py`, `test_reliability.py`), and the dedicated reliability report (`scripts/evaluate_reliability.py`) passes 16/16 checks across five profiles — one normal and four adversarial (conflicting genre/mood, a genre absent from the catalog, minimal taste signal, and a broad multi-genre profile). Consistency was verified directly: the same query run three times in a row produces byte-identical rankings and scores every time, which is the core guarantee an "AI recommender" needs to be trustworthy rather than random. The specialized model was confirmed working live (not just in its fallback path) — installing `scikit-learn`/`joblib` for real and re-running `python app.py` changed the top recommendation's score from 7.0 to 8.43 once the trained model's prediction started contributing, which is direct evidence the pipeline is actually using it, not just carrying dead code.

**What didn't work initially:** `scripts/evaluate_reliability.py` failed with `ModuleNotFoundError: No module named 'music_recommender'` the first time it was run directly (`python scripts/evaluate_reliability.py`), because a script inside a subdirectory isn't automatically able to import the project's top-level package. Fixed with a small `sys.path` bootstrap at the top of the script; both `python scripts/evaluate_reliability.py` and `python -m scripts.evaluate_reliability` now work.

**What we learned:** testing an AI system is not the same as testing ordinary code. The consistency check deliberately does *not* compare Claude's generated explanation text between runs — an LLM is expected to phrase the same grounded facts differently each time, and treating that as a bug would be testing the wrong thing. Instead, it checks the part of the pipeline that genuinely should be deterministic (which songs get retrieved, in what order, with what score) and leaves explanation quality to a separate, looser heuristic (`check_explanation_groundedness`: non-empty, addresses the listener by name, isn't absurdly long). Knowing which parts of an AI pipeline *should* be deterministic and testing only those that way — rather than either testing everything strictly or nothing at all — was the main testing lesson of this project.

### Reliability & Human-Evaluation Summary

This project proves it works through three of the four standard reliability approaches: **automated tests** (`tests/`, `pytest`), **logging and error handling** (every AI dependency logs a warning and falls back deterministically — see `retriever.py`, `recommender.py`, `taste_model.py`), and **human evaluation** (manual review of real system output against explicit criteria, table below). Confidence scoring wasn't added as a separate mechanism, but `semantic_score` and `audio_fit_score` already function as per-recommendation confidence signals surfaced in the final score.

**15/15 automated tests passed; 16/16 automated reliability checks passed** (consistency, graceful degradation, the empty-profile guardrail, and explanation groundedness, across 5 profiles). **Manual review of 6 representative interactions: 5/6 passed cleanly, 1 revealed a genuine limitation** — the specialized model doesn't extrapolate well to genre/mood combinations absent from its training data; it regresses toward the dataset's average instead of reflecting either stated preference. That's an honest consequence of a deliberately small, 29-row synthetic training set (see Design Decisions), not a code defect.

| Test Input | Evaluation Criteria | Result |
|---|---|---|
| Maya: `indie pop` / `reflective` / nostalgia, solitude / Phoebe Bridgers | Explanation cites the actual genre, mood, and artist matches; no crash | **Pass** |
| Deshawn: `metal` / `calm` (conflicting — no song in the catalog has a calm mood) | Does not fabricate a mood match that doesn't exist; falls back to the real genre match | **Pass** |
| Priya: `kpop` / `happy` (genre absent from both the catalog and training data) | Does not fabricate a kpop match; falls back to the real mood match | **Pass** |
| Empty profile (no genre, mood, theme, or artist) | Raises a clear, catchable error instead of crashing or returning nonsense | **Pass** — raises `ValueError: Taste profile cannot be empty` |
| Jordan: favorite artist only, no genre/mood/theme | Still produces a relevant recommendation via the one real signal; doesn't claim an audio-fit or semantic match it has no basis for | **Pass** |
| Specialized model: `classical` / `euphoric` (contradictory, unseen combination) | Predicted audio profile should lean toward at least one of the two stated preferences | **Fail** — predicted `{energy: 0.42, valence: 0.52, acousticness: 0.54}`, close to the training set's overall average rather than classical's calm/acoustic character or euphoric's high energy. With no training example pairing these two tags, the model has nothing to interpolate from and hedges toward the mean. |

## Reflection

Building this taught me that "an AI system" is rarely one model — it's a small pipeline of specialized pieces (a retriever, a trained model, a generator) each doing one narrow job well, wired together with enough guardrails that any single piece can fail without taking down the whole system. The hardest part wasn't getting any one component to work; it was deciding what happens when it *doesn't* — what a recommender should say when a listener's taste doesn't match anything in the catalog, or when an external API is down. Designing for those failure paths from the start, rather than bolting them on afterward, turned out to matter more for making this feel like a real system than any single feature did.

*(The graded responsible-AI reflection — collaboration process, a helpful AI suggestion, a flawed one, and the system's limitations — is in [`model_card.md`](model_card.md), not here.)*
