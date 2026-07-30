import csv

from music_recommender.data import Song
from music_recommender.song_attributes import SongAttributesStore


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "popularity", "release_decade", "detailed_mood_tags", "vocal_style", "instrumentation_notes"])
        for row in rows:
            writer.writerow(row)


def test_enrich_applies_attributes_for_matching_title(tmp_path, monkeypatch):
    csv_path = tmp_path / "song_attributes.csv"
    _write_csv(csv_path, [["Garden Song", "42", "2020s", "wistful;tender", "breathy", "acoustic guitar"]])
    monkeypatch.setattr("music_recommender.song_attributes.ATTRIBUTES_PATH", csv_path)

    store = SongAttributesStore()
    song = Song(title="Garden Song", artist="Phoebe Bridgers", genre="indie pop", mood="reflective")
    enriched = store.enrich([song])

    assert enriched[0].popularity == 42
    assert enriched[0].release_decade == "2020s"
    assert enriched[0].detailed_mood_tags == ["wistful", "tender"]
    assert enriched[0].vocal_style == "breathy"
    assert enriched[0].instrumentation_notes == "acoustic guitar"
    # Original Song object must be untouched (enrich returns new objects).
    assert song.popularity is None


def test_enrich_leaves_defaults_for_unmatched_title(tmp_path, monkeypatch):
    csv_path = tmp_path / "song_attributes.csv"
    _write_csv(csv_path, [["Some Other Song", "10", "2010s", "calm", "soft", "piano"]])
    monkeypatch.setattr("music_recommender.song_attributes.ATTRIBUTES_PATH", csv_path)

    store = SongAttributesStore()
    song = Song(title="Garden Song", artist="Phoebe Bridgers", genre="indie pop", mood="reflective")
    enriched = store.enrich([song])

    assert enriched[0].popularity is None
    assert enriched[0].detailed_mood_tags == []


def test_enrich_degrades_gracefully_when_csv_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr("music_recommender.song_attributes.ATTRIBUTES_PATH", missing_path)

    store = SongAttributesStore()
    song = Song(title="Garden Song", artist="Phoebe Bridgers", genre="indie pop", mood="reflective")
    enriched = store.enrich([song])

    assert enriched[0].popularity is None
    assert enriched == [song]
