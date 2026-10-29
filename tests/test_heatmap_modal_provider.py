import pytest
from src.rendering.match_stats_provider import MatchStatsProvider
from src.rendering.tactical_narrator import TacticalNarrator

def test_mbappe_and_vinicius_heatmaps_and_flags():
    provider = MatchStatsProvider()
    
    # Mbappé
    mbappe = provider.get_player_stats("Kylian Mbappé")
    assert mbappe is not None
    assert mbappe["countryCode"] == "FR"
    assert mbappe["countryFlagEmoji"] == "🇫🇷"
    assert len(mbappe["heatPoints"]) >= 10
    assert "LEFT" in mbappe["tacticalNote"].upper()
    
    # Vinícius
    vini = provider.get_player_stats("Vinícius Júnior")
    assert vini is not None
    assert vini["countryCode"] == "BR"
    assert vini["countryFlagEmoji"] == "🇧🇷"
    assert len(vini["heatPoints"]) >= 10

def test_extract_heatmap_and_comparison_cue_tags():
    raw = "Mbappé drops deep to receive [heatmap_comparison:Kylian Mbappé:Vinícius Júnior] as the block shifts."
    cleaned, tags = TacticalNarrator.extract_cue_tags(raw)
    
    assert "Mbappé drops deep to receive as the block shifts." in cleaned
    assert len(tags) == 1
    assert tags[0]["type"] == "heatmap_comparison"
    assert tags[0]["target"] == "Kylian Mbappé:Vinícius Júnior"

def test_automatic_data_tag_injection_in_fallback_script():
    narrator = TacticalNarrator(api_key="fake_key")
    cause_tel = {
        "catalyst_player": {"name": "Kylian Mbappé", "role": "CF"},
        "vacated_label": "LEFT HALF-SPACE",
    }
    beats = {
        "beat_1_setup": "Fast transition begins.",
        "beat_2_cause": "The defense fractures centrally.",
        "beat_2_steps": {"step_1": "Setup", "step_2": "Fracture", "step_3": "Space", "step_4": "Decision"},
        "beat_3_effect": "A goal is scored.",
    }
    injected = narrator._inject_real_names(beats, cause_tel)
    assert "[heatmap:Kylian Mbappé]" in injected["beat_2_cause"]
