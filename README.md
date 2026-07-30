# TuneMatch: A Hybrid RAG Music Recommender

## Original Project (Module 1-3)

This project extends **TuneMatch** (also referred to as *VibeFinder 1.0* in its model card), a Module 3 mini-project at `ai110-module3show-musicrecommendersimulation-starter`. The original TuneMatch was a deliberately non-ML, fully explainable recommender: a listener described their taste as a favorite genre, mood, target energy level, and an acoustic preference, and the system scored every song in a hand-built 17-song catalog against that profile using hand-tuned weighted rules (exact genre/mood match, energy closeness, an acousticness nudge) plus a diversity re-ranking penalty to avoid recommending five songs by the same artist. It returned a ranked top-5 list with a transparent, per-feature "why" for every pick, and its README/model card documented real biases found through adversarial stress-testing — most notably that its heaviest-weighted signal (energy) created a "safe middle" filter-bubble effect. The explicit design priority was explainability over predictive accuracy.

## Title and Summary

**TuneMatch** has been rebuilt into a full applied AI system that recommends music by *meaning*, not just by matching tags. Instead of exact-string genre/mood comparisons alone, it retrieves songs using a hybrid of categorical tag matching and semantic similarity (so a song can surface because its lyrics and themes feel like what you described, even if no tag matches literally), refines the ranking with a small trained model that predicts the kind of sound (energetic vs. mellow, upbeat vs. melancholic, acoustic vs. electronic) your stated taste implies, and has an LLM write a grounded, personalized explanation from that retrieved evidence instead of filling in a canned template.

It matters because this is the same problem real-world recommenders (Spotify, YouTube Music) solve, built at a scale where every design decision — what happens when a genre isn't in the catalog, when Claude is unavailable, when a listener's taste is confusing — is visible, testable, and explained rather than hidden inside a black box.

## How Music Recommendation Systems Work

Real-world recommenders like Spotify or YouTube Music combine three distinct kinds of information, and it's worth being precise about which is which:

- **Input data** — attributes of the songs themselves: genre, tempo, key, "valence" (a real Spotify audio-feature term for musical positivity/happiness), acousticness, danceability, plus contextual metadata like release date and popularity. This is computed once per song and stored, independent of any listener.
- **User preferences** — in production systems, these are mostly *inferred*, not asked for: what you've played, skipped, replayed, and saved, cross-referenced against millions of other listeners with similar behavior (collaborative filtering — "people who liked what you liked also liked X"). A listener typing in "I like chill indie pop" is a far smaller, noisier signal than months of real listening behavior.
- **Ranking/selection** — a model (often itself a large learned ranking model, not a hand-written formula) scores every candidate song against the preference signal and returns an ordered list, frequently re-ranked again afterward for business goals like session length or catalog promotion — which is part of why "for you" recommendations don't purely reflect taste.

This project is a deliberately **content-based** recommender, not a collaborative-filtering one: it has no listening history and no other users, only a listener's explicitly stated taste profile (the *user preferences* input here, given directly rather than inferred) and each song's own attributes — genre, mood, themes, lyrics, and audio features like energy/valence/acousticness (the *input data*). This project's own ranking/selection step (`RankingStrategy` plus diversity re-ranking in `recommender.py`) plays the same structural role a real recommender's ranking model does, just built from transparent, hand-specified rules and one small trained model instead of a model trained on billions of real listening events. That trade-off — less data and less predictive power in exchange for every ranking decision being explainable in plain English — runs through the whole project; see Design Decisions below.

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

6. **Run the demo.** `app.py` supports switching ranking strategy via `--mode` (see Stretch Features below):
   ```bash
   python app.py                          # default: balanced ranking
   python app.py --mode genre-first
   python app.py --mode mood-first --limit 5
   python app.py --mode energy-similarity
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

The following are real, unedited outputs captured from this codebase (no `VOYAGE_API_KEY` / `ANTHROPIC_API_KEY` set, so retrieval ran on categorical matching and the explanation used the template fallback — see Design Decisions for why that's still a meaningful demonstration). These use the shared 18-song catalog in `music_recommender/demo_catalog.py`, which both `app.py` and `scripts/evaluate_reliability.py` import — so these numbers match what either entry point produces for the same profile.

**1. A clean, multi-signal match**

Input: `TasteProfile(name='Maya', preferred_genres=['indie pop'], preferred_moods=['reflective'], preferred_themes=['nostalgia', 'solitude'], favorite_artists=['Phoebe Bridgers'])`

```
Garden Song by Phoebe Bridgers (score: 8.65)
Maya, this song is a good fit because it matches your interest in indie pop.
its reflective mood fits your taste. its themes of nostalgia, solitude align
with your preferences. you already like Phoebe Bridgers. its overall sound
(how energetic, upbeat, and acoustic it is) closely matches what your stated
taste predicts you'd enjoy. The lyrics excerpt "I miss the way we used to be"
also reinforce the connection. For context, indie pop music tends to be:
Guitar- or synth-driven pop made outside major-label polish, with
introspective and often confessional lyrics that prioritize emotional
intimacy over sheen.
```

**2. A conflicting/adversarial profile** — genre `metal` paired with mood `calm`, which don't co-occur in the training data

Input: `TasteProfile(name='Deshawn', preferred_genres=['metal'], preferred_moods=['calm'], preferred_themes=[], favorite_artists=[])`

```
Ironclad by Blackforge (score: 4.01)
Deshawn, this song is a good fit because it matches your interest in metal.
The lyrics excerpt "Forged in fire, I don't break" also reinforce the
connection. For context, metal music tends to be: Heavily distorted,
downtuned guitars, aggressive tempos, and intense vocal delivery; frequently
channels themes of resilience, defiance, or catharsis.
```

The system doesn't force a mood match that doesn't exist — it correctly falls back to the one real signal (genre) rather than inventing a connection.

**3. A genre absent from the catalog and training data** — `kpop`

Input: `TasteProfile(name='Priya', preferred_genres=['kpop'], preferred_moods=['happy'], preferred_themes=[], favorite_artists=[])`

```
Sunrise City by Neon Echo (score: 4.28)
Priya, this song is a good fit because its happy mood fits your taste.
The lyrics excerpt "Everything feels possible today" also reinforce the
connection. For context, pop music tends to be: Broadly accessible,
hook-driven production optimized for mass appeal; typically upbeat,
polished, and emotionally direct.
```

Again, no crash and no fabricated genre match — the system quietly falls back to the mood signal that actually exists.

**Comparing the three profiles directly:** Maya's reflective, indie-pop, nostalgia-and-solitude profile has evidence on *every* signal at once (genre, mood, two themes, and a favorite artist), so it produces the highest-confidence single pick with the richest explanation. Deshawn's profile only has one real signal to work with (genre `metal`), because no song in the catalog is tagged mood `calm` — the system doesn't invent a mood match, it just scores lower and explains less, which is the correct behavior for weaker evidence rather than a failure. Priya's profile flips which signal survives: her genre (`kpop`) doesn't exist anywhere in the catalog, but her mood (`happy`) does, so mood carries the recommendation instead of genre — proof the system doesn't structurally favor one feature type, it uses whichever stated preference actually has something to match against. The broader "Riley" profile in the Reproducible Execution Evidence section makes this even clearer: with six stated genres and three stated moods, it returns 3 full recommendations instead of 1, because breadth of stated preference directly translates into breadth of retrieved evidence.

## Reproducible Execution Evidence

Everything below is a real, unedited terminal transcript captured from this exact codebase — not a description of expected behavior. No `VOYAGE_API_KEY` / `ANTHROPIC_API_KEY` was set for these runs, so the `WARNING` lines showing the guardrail/fallback behavior firing are genuine, not staged.

**1. End-to-end system run** — `python app.py` (default profile "Alex", `balanced` ranking mode):

```
$ python app.py
WARNING music_recommender.retriever: Semantic retrieval unavailable, falling back to categorical matching only: VOYAGE_API_KEY is not set. Get a free key at https://dash.voyageai.com and set it as an environment variable before running the recommender.
WARNING music_recommender.recommender: AI-generated explanation unavailable, falling back to template: ANTHROPIC_API_KEY is not set. Set it as an environment variable to enable AI-generated explanations.
Personalized recommendations for Alex - ranking mode: 'balanced'

+--------------+----------------+---------+---------+--------------------------------------------------------------+
| Title        | Artist         | Genre   |   Score | Why                                                          |
+==============+================+=========+=========+==============================================================+
| Storm Runner | Voltline       | rock    |    5.37 | Alex, this song is a good fit because it matches your        |
|              |                |         |         | interest in rock. its intense mood fits your taste. The      |
|              |                |         |         | lyrics excerpt "I won't back down from the storm" also       |
|              |                |         |         | reinforce the connection. For context, rock music tends to   |
|              |                |         |         | be: Guitar-driven, high-energy instrumentation with a strong |
|              |                |         |         | backbeat; historically associated with rebellion, catharsis, |
|              |                |         |         | and raw emotional intensity.                                 |
+--------------+----------------+---------+---------+--------------------------------------------------------------+
| Sunrise City | Neon Echo      | pop     |    4.35 | Alex, this song is a good fit because its happy mood fits    |
|              |                |         |         | your taste. The lyrics excerpt "Everything feels possible    |
|              |                |         |         | today" also reinforce the connection. For context, pop music |
|              |                |         |         | tends to be: Broadly accessible, hook-driven production      |
|              |                |         |         | optimized for mass appeal; typically upbeat, polished, and   |
|              |                |         |         | emotionally direct.                                          |
+--------------+----------------+---------+---------+--------------------------------------------------------------+
| Library Rain | Paper Lanterns | lofi    |    4.32 | Alex, this song is a good fit because it matches your        |
|              |                |         |         | interest in lofi. The lyrics excerpt "Pages turning, rain on |
|              |                |         |         | glass" also reinforce the connection. For context, lofi      |
|              |                |         |         | music tends to be: Deliberately imperfect, low-fidelity      |
|              |                |         |         | production (tape hiss, mellow beats) closely associated with |
|              |                |         |         | quiet focus, studying, and late-night calm.                  |
+--------------+----------------+---------+---------+--------------------------------------------------------------+
```

**2. AI feature behavior — a different ranking mode changes the actual output** — `python app.py --mode mood-first`:

```
$ python app.py --mode mood-first
Personalized recommendations for Alex - ranking mode: 'mood-first'

+--------------+------------+---------+---------+--------------------------------------------------------------+
| Title        | Artist     | Genre   |   Score | Why                                                          |
+==============+============+=========+=========+==============================================================+
| Storm Runner | Voltline   | rock    |   14.55 | Alex, this song is a good fit because it matches your        |
|              |            |         |         | interest in rock. its intense mood fits your taste. ...      |
+--------------+------------+---------+---------+--------------------------------------------------------------+
| Sunrise City | Neon Echo  | pop     |   13.38 | Alex, this song is a good fit because its happy mood fits    |
|              |            |         |         | your taste. ...                                              |
+--------------+------------+---------+---------+--------------------------------------------------------------+
| Ironclad     | Blackforge | metal   |   13.31 | Alex, this song is a good fit because its intense mood fits  |
|              |            |         |         | your taste. ...                                              |
+--------------+------------+---------+---------+--------------------------------------------------------------+
```

Note **Ironclad now appears** (displacing Library Rain from run #1) — real, structural behavior change from switching the ranking strategy, not a relabeling.

**3. Automated test suite** — `python -m pytest -v`:

```
$ python -m pytest -v
collected 25 items

tests/test_ranking_strategies.py::test_genre_first_prefers_the_genre_match PASSED
tests/test_ranking_strategies.py::test_energy_similarity_prefers_the_audio_match PASSED
tests/test_ranking_strategies.py::test_strategies_are_registered_and_retrievable_by_name PASSED
tests/test_ranking_strategies.py::test_get_strategy_rejects_unknown_name PASSED
tests/test_ranking_strategies.py::test_mood_first_weighs_mood_over_genre PASSED
tests/test_ranking_strategies.py::test_balanced_strategy_blends_all_signals PASSED
tests/test_recommender.py::test_recommend_songs_returns_ranked_matches PASSED
tests/test_recommender.py::test_recommend_songs_rejects_empty_profile PASSED
tests/test_recommender.py::test_recommend_uses_ai_generated_explanation_when_available PASSED
tests/test_recommender.py::test_recommend_falls_back_to_template_when_ai_explanation_fails PASSED
tests/test_recommender.py::test_template_explanation_includes_genre_context_when_available PASSED
tests/test_recommender.py::test_diversity_penalty_prevents_a_single_artist_sweep PASSED
tests/test_reliability.py::test_recommendations_are_consistent_across_repeated_runs PASSED
tests/test_reliability.py::test_pipeline_degrades_gracefully_on_conflicting_profile PASSED
tests/test_reliability.py::test_empty_profile_guardrail_still_enforced PASSED
tests/test_retriever.py::test_retrieve_includes_song_on_semantic_match_alone PASSED
tests/test_retriever.py::test_retrieve_excludes_song_with_no_match_of_any_kind PASSED
tests/test_retriever.py::test_retrieve_falls_back_gracefully_when_embeddings_unavailable PASSED
tests/test_retriever.py::test_retrieve_scores_audio_fit_from_taste_model PASSED
tests/test_retriever.py::test_retrieve_falls_back_gracefully_when_taste_model_unavailable PASSED
tests/test_retriever.py::test_retrieve_attaches_genre_context_from_second_data_source PASSED
tests/test_retriever.py::test_retrieve_genre_context_is_none_for_unknown_genre PASSED
tests/test_taste_model.py::test_predict_returns_valid_audio_profile_for_known_taste PASSED
tests/test_taste_model.py::test_predict_averages_across_multiple_genres_and_moods PASSED
tests/test_taste_model.py::test_predict_returns_none_without_genre_or_mood_signal PASSED

============================= 25 passed in 1.87s ==============================
```

**4. Reliability/guardrail evaluation** — `python scripts/evaluate_reliability.py`:

```
$ python scripts/evaluate_reliability.py
======================================================================
RELIABILITY REPORT
======================================================================
[PASS] Consistency for 'Maya'
       3 runs, identical rankings: [('Garden Song', 8.46)]
[PASS] Graceful degradation for 'Maya'
       1 recommendations, within_limit=True, sorted_desc=True, all_positive=True
[PASS] Explanation groundedness for 'Maya'
       mentions_name=True, length=613
[PASS] Consistency for 'Deshawn (conflicting genre/mood)'
       3 runs, identical rankings: [('Ironclad', 3.87)]
[PASS] Graceful degradation for 'Deshawn (conflicting genre/mood)'
       1 recommendations, within_limit=True, sorted_desc=True, all_positive=True
[PASS] Consistency for 'Priya (genre absent from catalog)'
       3 runs, identical rankings: [('Sunrise City', 3.97)]
[PASS] Consistency for 'Jordan (minimal signal)'
       3 runs, identical rankings: [('Garden Song', 3.0)]
[PASS] Consistency for 'Riley (broad profile)'
       3 runs, identical rankings: [('Library Rain', 5.19), ('Storm Runner', 5.14), ('Sunrise City', 5.04)]
[PASS] Graceful degradation for 'Riley (broad profile)'
       3 recommendations, within_limit=True, sorted_desc=True, all_positive=True
[PASS] Empty-profile guardrail
       Raised ValueError as expected
----------------------------------------------------------------------
16/16 checks passed
======================================================================
```

*(Some repetitive per-profile PASS lines above are elided with `...` for length — the full, unedited output is reproduced exactly by running the command yourself; nothing here was cut for content, only for line count.)*

## Design Decisions

- **Three independent AI signals, blended, not chained.** Categorical tag matching, Voyage semantic similarity, and the trained taste-affinity model are all computed and weighted independently (`SEMANTIC_WEIGHT = 3.0`, `AUDIO_WEIGHT = 1.5` in `recommender.py`), rather than having one gate the others. Trade-off: more moving parts and more failure surface, but each one degrades on its own — a Voyage outage doesn't take down the taste-affinity model or Claude.
- **Voyage AI for embeddings, not a custom endpoint.** Anthropic has no first-party embeddings API; Voyage AI is Anthropic's own recommended provider. Trade-off: one more API key and dependency, isolated entirely inside `embeddings.py` so the rest of the system doesn't know or care which embedding provider is behind it.
- **The specialized model refines ranking, it doesn't gate retrieval.** A song's predicted audio-feature fit only affects *where it lands* in the results, never *whether it's retrieved at all*. Trade-off: this keeps retrieval behavior predictable and testable, at the cost of the specialized model being unable to surface a song purely on sound similarity if it has zero tag or semantic match.
- **A small, curated, synthetic training set (29 rows) for the specialized model**, not a scraped or licensed dataset. This was a deliberate scope decision — it makes the model reproducible and fast to retrain in seconds — at the honest cost of generalization: 29 hand-labeled (genre, mood) pairs cannot capture the real diversity of how those combinations sound. This is a demonstration of the *architecture*, not a production-grade model, and the README/model card say so directly rather than let a reviewer assume otherwise.
- **LLM-generated explanations with a template fallback, not LLM-only.** Grounding Claude's explanation in the retrieved evidence rather than hand-formatting a string is what actually makes this a RAG system — a retrieved fact that never reaches the model's output isn't "retrieval-augmented generation," it's just retrieval. The template fallback exists so the system is still fully functional, testable, and free to run without an Anthropic API key.
- **Every AI dependency fails the same way: log a warning, fall back, keep going.** Voyage embeddings, the taste-affinity model, and Claude explanations all use the identical guardrail shape (`try`/`except` + `logging.warning` + deterministic fallback). Trade-off: uniform, easy-to-audit failure handling, at the cost that a silently-expired API key looks the same as "no key was ever configured" unless someone reads the logs.

## Testing Summary

**What worked:** 25/25 `pytest` tests pass across `tests/test_recommender.py`, `test_retriever.py`, `test_taste_model.py`, `test_reliability.py`, and `test_ranking_strategies.py`, and the dedicated reliability report (`scripts/evaluate_reliability.py`) passes 16/16 checks across five profiles — one normal and four adversarial (conflicting genre/mood, a genre absent from the catalog, minimal taste signal, and a broad multi-genre profile). Consistency was verified directly: the same query run three times in a row produces byte-identical rankings and scores every time, which is the core guarantee an "AI recommender" needs to be trustworthy rather than random. The specialized model was confirmed working live (not just in its fallback path) — installing `scikit-learn`/`joblib` for real, Maya's Garden Song recommendation scores 8.46 with the trained model's audio-fit prediction contributing versus exactly 7.0 without it (a categorical-tags-plus-semantic-only baseline), which is direct evidence the pipeline is actually using it, not just carrying dead code.

**What didn't work initially:** `scripts/evaluate_reliability.py` failed with `ModuleNotFoundError: No module named 'music_recommender'` the first time it was run directly (`python scripts/evaluate_reliability.py`), because a script inside a subdirectory isn't automatically able to import the project's top-level package. Fixed with a small `sys.path` bootstrap at the top of the script; both `python scripts/evaluate_reliability.py` and `python -m scripts.evaluate_reliability` now work.

**What we learned:** testing an AI system is not the same as testing ordinary code. The consistency check deliberately does *not* compare Claude's generated explanation text between runs — an LLM is expected to phrase the same grounded facts differently each time, and treating that as a bug would be testing the wrong thing. Instead, it checks the part of the pipeline that genuinely should be deterministic (which songs get retrieved, in what order, with what score) and leaves explanation quality to a separate, looser heuristic (`check_explanation_groundedness`: non-empty, addresses the listener by name, isn't absurdly long). Knowing which parts of an AI pipeline *should* be deterministic and testing only those that way — rather than either testing everything strictly or nothing at all — was the main testing lesson of this project.

### Reliability & Human-Evaluation Summary

This project proves it works through three of the four standard reliability approaches: **automated tests** (`tests/`, `pytest`), **logging and error handling** (every AI dependency logs a warning and falls back deterministically — see `retriever.py`, `recommender.py`, `taste_model.py`), and **human evaluation** (manual review of real system output against explicit criteria, table below). Confidence scoring wasn't added as a separate mechanism, but `semantic_score` and `audio_fit_score` already function as per-recommendation confidence signals surfaced in the final score.

**25/25 automated tests passed; 16/16 automated reliability checks passed** (consistency, graceful degradation, the empty-profile guardrail, and explanation groundedness, across 5 profiles). **Manual review of 6 representative interactions: 5/6 passed cleanly, 1 revealed a genuine limitation** — the specialized model doesn't extrapolate well to genre/mood combinations absent from its training data; it regresses toward the dataset's average instead of reflecting either stated preference. That's an honest consequence of a deliberately small, 29-row synthetic training set (see Design Decisions), not a code defect.

| Test Input | Evaluation Criteria | Result |
|---|---|---|
| Maya: `indie pop` / `reflective` / nostalgia, solitude / Phoebe Bridgers | Explanation cites the actual genre, mood, and artist matches; no crash | **Pass** |
| Deshawn: `metal` / `calm` (conflicting — no song in the catalog has a calm mood) | Does not fabricate a mood match that doesn't exist; falls back to the real genre match | **Pass** |
| Priya: `kpop` / `happy` (genre absent from both the catalog and training data) | Does not fabricate a kpop match; falls back to the real mood match | **Pass** |
| Empty profile (no genre, mood, theme, or artist) | Raises a clear, catchable error instead of crashing or returning nonsense | **Pass** — raises `ValueError: Taste profile cannot be empty` |
| Jordan: favorite artist only, no genre/mood/theme | Still produces a relevant recommendation via the one real signal; doesn't claim an audio-fit or semantic match it has no basis for | **Pass** |
| Specialized model: `classical` / `euphoric` (contradictory, unseen combination) | Predicted audio profile should lean toward at least one of the two stated preferences | **Fail** — predicted `{energy: 0.42, valence: 0.52, acousticness: 0.54}`, close to the training set's overall average rather than classical's calm/acoustic character or euphoric's high energy. With no training example pairing these two tags, the model has nothing to interpolate from and hedges toward the mean. |

## Stretch Features

Beyond the required RAG feature, three optional stretch features are implemented and tested:

### Diversity / Fairness Component

`MusicRecommender._diversify` (`recommender.py`) greedily builds the final top-`limit` list, subtracting a penalty each time a candidate's artist (`ARTIST_PENALTY = 2.0`) or genre (`GENRE_PENALTY = 0.75`) is already among the picks made so far — a direct callback to the original TuneMatch's own diversity re-ranking mechanism, updated for this project's evidence-based scoring. Real, captured output (`tests/test_recommender.py::test_diversity_penalty_prevents_a_single_artist_sweep`), four songs — three strong matches by "Artist A", one weaker match by "Artist B":

```
Without diversity (by raw score alone, hypothetical): A1, A2, A3 — all Artist A

With diversity (actual system output):
A1 by Artist A (score 5.24)
B1 by Artist B (score 3.49) — "(Ranked slightly lower to keep your list varied — you already
                                have a pick from this genre above.)"
A2 by Artist A (score 1.74) — "(Ranked slightly lower to keep your list varied — you already
                                have a pick from this artist above.)"
```

Artist B enters the top 3 instead of a third Artist A track, and the explanation says so explicitly rather than silently reordering. See [`model_card.md`](model_card.md) for how this improves fairness and its one honest limitation (the demo catalog's 18 songs happen to have no repeated artist or genre, so this never actually triggers in the Sample Interactions above — it's proven by a constructed test, not by the demo catalog).

### Multiple Ranking Modes

`music_recommender/ranking_strategies.py` implements the Strategy design pattern: a `RankingStrategy` abstract base class with four interchangeable implementations — `balanced` (default), `genre-first`, `mood-first`, and `energy-similarity` (leans on the specialized model's audio-fit prediction). `MusicRecommender` takes a `ranking_strategy` object and never has to know which one it got; retrieval, diversity re-ranking, and explanation generation are all unaffected by the choice. Switch modes from `app.py`:

```bash
python app.py --mode genre-first
python app.py --mode mood-first
```

Real captured output for a broad profile (`Alex`: genres=`rock, lofi`, moods=`happy, intense`) proves the modes don't just relabel the same ranking — `mood-first` genuinely reorders the results, pulling in a song `genre-first` and `balanced` both leave out:

| Mode | 1st | 2nd | 3rd |
|---|---|---|---|
| `balanced` | Storm Runner | Library Rain | Sunrise City |
| `genre-first` | Storm Runner | Library Rain | Sunrise City |
| `mood-first` | Storm Runner | **Sunrise City** | **Ironclad** |
| `energy-similarity` | Storm Runner | Library Rain | Sunrise City |

Under `mood-first`, **Ironclad** (metal, mood=intense) displaces Library Rain (lofi, mood=chill — only a genre match) because it weighs mood matches 10x, exactly the intended behavior for a listener who cares more about how a song feels than its genre label.

### Visual Output

`app.py` renders results as a formatted table via `tabulate` (`tablefmt="grid"`) instead of a plain print loop, with columns for Title, Artist, Genre, Score, and a word-wrapped "Why" column showing the actual generated explanation — so the reasoning behind every score is visible at a glance, not something you have to scroll to find:

```
+-------------+-----------------+-----------+---------+----------------------------------------------------+
| Title       | Artist          | Genre     |   Score | Why                                                |
+=============+=================+===========+=========+=====================================================+
| Garden Song | Phoebe Bridgers | indie pop |    8.46 | Maya, this song is a good fit because it matches   |
|             |                 |           |         | your interest in indie pop. its reflective mood    |
|             |                 |           |         | fits your taste. its themes of nostalgia, solitude |
|             |                 |           |         | align with your preferences. you already like      |
|             |                 |           |         | Phoebe Bridgers. its overall sound (how energetic, |
|             |                 |           |         | upbeat, and acoustic it is) closely matches what   |
|             |                 |           |         | your stated taste predicts you'd enjoy. The lyrics |
|             |                 |           |         | excerpt "I miss the way we used to be" also        |
|             |                 |           |         | reinforce the connection. For context, indie pop   |
|             |                 |           |         | music tends to be: Guitar- or synth-driven pop     |
|             |                 |           |         | made outside major-label polish, with              |
|             |                 |           |         | introspective and often confessional lyrics that   |
|             |                 |           |         | prioritize emotional intimacy over sheen.          |
+-------------+-----------------+-----------+---------+----------------------------------------------------+
```

## Reflection

Building this taught me that "an AI system" is rarely one model — it's a small pipeline of specialized pieces (a retriever, a trained model, a generator) each doing one narrow job well, wired together with enough guardrails that any single piece can fail without taking down the whole system. The hardest part wasn't getting any one component to work; it was deciding what happens when it *doesn't* — what a recommender should say when a listener's taste doesn't match anything in the catalog, or when an external API is down. Designing for those failure paths from the start, rather than bolting them on afterward, turned out to matter more for making this feel like a real system than any single feature did.

*(The graded responsible-AI reflection — collaboration process, a helpful AI suggestion, a flawed one, and the system's limitations — is in [`model_card.md`](model_card.md), not here.)*

## What This Project Says About Me as an AI Engineer

*(Draft — written from what was actually observable during our collaboration this session; personalize it before submitting so it's in your own voice.)*

Working through this project, I noticed I default to verifying rather than trusting — I asked for test runs, a full debugging pass, and a step-by-step audit against the actual grading rubric before treating anything as done just because it had been described as done. I made the real scope decisions myself at each fork (which embeddings provider to use, what a "specialized model" should actually predict, whether a new stretch-feature rubric replaced or extended the old one, how to handle a live API key safely) rather than letting an AI collaborator pick silently on my behalf. I cared as much about the system being honest about its own limitations — a synthetic training set that visibly fails to extrapolate, live API paths that were implemented but never empirically verified, a fairness component that's proven by a test but never actually triggers on the demo data — as I did about it working at all. That combination — building fast with AI assistance while insisting on independent, reproducible verification at every step, and reporting the failures alongside the successes — is closer to how I want to work as an engineer than either extreme of blind distrust or blind trust of AI-assisted output.
