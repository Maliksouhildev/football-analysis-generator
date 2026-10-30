import pytest
from src.ingestion.replay_ingestor import ReplayIngestor
from src.scraper.autonomous_match_resolver import AutonomousMatchResolver


def test_filename_metadata_parsing():
    fname = "bar.v.rma.full.laliga.25.10.2026.fullmatchsports.com.mp4"
    meta = ReplayIngestor.parse_match_metadata_from_filename(fname)
    assert meta.home == "Barcelona"
    assert meta.away == "Real Madrid"
    assert meta.match_date == "2026-10-25"
    assert meta.competition == "esp.1"

    # Backward compatibility with 3-tuple unpacking
    home, away, score = meta
    assert home == "Barcelona"
    assert away == "Real Madrid"


def test_goal_event_parsing():
    resolver = AutonomousMatchResolver()
    sum_data = {
        "keyEvents": [
            {
                "clock": {"value": 2460.0, "displayValue": "41'"},
                "type": {"text": "Goal"},
                "text": "Goal! Barcelona 0, Real Madrid 1. Kylian Mbappé (Real Madrid) right footed shot from the centre of the box to the bottom left corner.",
            }
        ]
    }
    events = resolver._parse_espn_key_events(sum_data, event_id="123", home="Barcelona", away="Real Madrid")
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type.value == "goal"
    assert ev.primary_player.name == "Kylian Mbappé"
    assert ev.team.name == "Real Madrid"
