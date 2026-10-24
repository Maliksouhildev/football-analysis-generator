"""
src/rendering/sfx_generator.py
Procedural Sound Effects Synthesizer for Tactical Broadcasts.

Generates broadcast-grade 48kHz stereo WAV sound effects using pure NumPy & SciPy:
1. sfx_freeze_whoosh.wav: Sub-bass cinematic drop (140Hz -> 38Hz) with filtered air whoosh.
2. sfx_tactical_tick.wav: Crisp 1800Hz damped UI tick.
3. sfx_ball_kick.wav: Punchy low-mid transient thud (110Hz -> 45Hz) with 8ms click.
4. sfx_goal_impact.wav: Deep stadium bass boom with harmonic chord swell.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

logger = logging.getLogger("rendering.sfx_generator")

SAMPLE_RATE = 48000


def _to_stereo_pcm16(mono_signal: np.ndarray, peak_norm: float = 0.95) -> np.ndarray:
    """Normalizes mono float signal and converts to 16-bit signed stereo PCM."""
    max_val = np.max(np.abs(mono_signal))
    if max_val > 1e-6:
        mono_signal = mono_signal * (peak_norm / max_val)
    else:
        mono_signal = np.zeros_like(mono_signal)
    
    pcm = np.int16(np.clip(mono_signal, -1.0, 1.0) * 32767)
    return np.column_stack((pcm, pcm))


def synthesize_freeze_whoosh(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Sub-bass cinematic drop (140Hz -> 38Hz exponential sine sweep over 0.32s)
    layered with filtered white-noise air whoosh.
    """
    duration = 0.32
    n_samples = int(duration * sr)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    # 1. Exponential sine sweep from 140Hz to 38Hz
    f_start = 140.0
    f_end = 38.0
    decay_rate = np.log(f_end / f_start) / duration
    phase = 2.0 * np.pi * f_start * (np.exp(decay_rate * t) - 1.0) / decay_rate
    sub_bass = np.sin(phase)

    # Envelope for sub-bass: fast attack, smooth decay
    env_sub = np.minimum(t / 0.03, 1.0) * np.exp(-t / 0.16)
    sub_bass = sub_bass * env_sub

    # 2. Filtered white-noise air whoosh
    noise = np.random.normal(0.0, 1.0, n_samples)
    b, a = butter(2, [350.0 / (sr / 2.0), 2800.0 / (sr / 2.0)], btype="band")
    filtered_noise = lfilter(b, a, noise)
    env_noise = np.sin(np.pi * (t / duration)) ** 1.5
    whoosh = filtered_noise * env_noise

    # Layer: 70% sub-bass + 35% air whoosh normalized smoothly for gentle background whoosh
    mix = 0.70 * sub_bass + 0.35 * whoosh
    return _to_stereo_pcm16(mix, peak_norm=0.45)


def synthesize_tactical_tick(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Crisp high-tech UI tick (1800Hz damped sine burst, 0.04s).
    """
    duration = 0.04
    n_samples = int(duration * sr)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    sine_burst = np.sin(2.0 * np.pi * 1800.0 * t) * np.exp(-t / 0.007)
    # Click transient spike in first 2ms
    click = np.sin(2.0 * np.pi * 3200.0 * t) * np.exp(-t / 0.002)

    mix = 0.75 * sine_burst + 0.40 * click
    return _to_stereo_pcm16(mix)


def synthesize_ball_kick(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Punchy low-mid transient thud (110Hz -> 45Hz fast pitch-drop + 8ms click, 0.14s).
    """
    duration = 0.14
    n_samples = int(duration * sr)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    # 1. Pitch drop from 110Hz to 45Hz
    k = 35.0  # decay rate
    f_base = 45.0
    f_delta = 65.0
    phase = 2.0 * np.pi * (f_base * t + (f_delta / k) * (1.0 - np.exp(-k * t)))
    thud = np.sin(phase) * np.exp(-t / 0.045)

    # 2. 8ms click transient
    click = np.sin(2.0 * np.pi * 1400.0 * t) * np.exp(-t / 0.003) * (t < 0.008)

    mix = 0.85 * thud + 0.35 * click
    return _to_stereo_pcm16(mix)


def synthesize_goal_impact(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Deep stadium bass boom + shimmering harmonic chord swell (0.9s).
    """
    duration = 0.90
    n_samples = int(duration * sr)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    # 1. Sub-bass impact boom (55Hz -> 28Hz)
    boom_phase = 2.0 * np.pi * (55.0 * t - 15.0 * (t ** 1.3))
    boom = np.sin(boom_phase) * np.exp(-t / 0.28)

    # 2. Shimmering harmonic chord swell (F minor chord: F3, Ab3, C4, Eb4)
    harmonics = [174.61, 207.65, 261.63, 311.13, 523.25]
    chord = np.zeros_like(t)
    for idx, freq in enumerate(harmonics):
        amp = 1.0 / (idx + 1.2)
        detune = 1.0 + 0.003 * np.sin(2.0 * np.pi * (2.0 + idx) * t)
        chord += amp * np.sin(2.0 * np.pi * freq * detune * t)

    env_chord = (np.sin(np.pi * (t / duration)) ** 1.2) * np.exp(-t / 0.45)
    chord_swell = chord * env_chord

    mix = 0.75 * boom + 0.35 * chord_swell
    return _to_stereo_pcm16(mix)


class ProceduralSFXGenerator:
    """Generates and caches all broadcast procedural sound effects."""

    @staticmethod
    def generate_all_sfx(output_dir: Union[str, Path] = "renders/assets/audio/sfx") -> Dict[str, str]:
        out_p = Path(output_dir).resolve()
        out_p.mkdir(parents=True, exist_ok=True)

        sfx_files = {
            "sfx_freeze_whoosh": out_p / "sfx_freeze_whoosh.wav",
            "sfx_tactical_tick": out_p / "sfx_tactical_tick.wav",
            "sfx_ball_kick": out_p / "sfx_ball_kick.wav",
            "sfx_goal_impact": out_p / "sfx_goal_impact.wav",
        }

        # Synthesize each sound
        whoosh_data = synthesize_freeze_whoosh()
        wavfile.write(str(sfx_files["sfx_freeze_whoosh"]), SAMPLE_RATE, whoosh_data)

        tick_data = synthesize_tactical_tick()
        wavfile.write(str(sfx_files["sfx_tactical_tick"]), SAMPLE_RATE, tick_data)

        kick_data = synthesize_ball_kick()
        wavfile.write(str(sfx_files["sfx_ball_kick"]), SAMPLE_RATE, kick_data)

        impact_data = synthesize_goal_impact()
        wavfile.write(str(sfx_files["sfx_goal_impact"]), SAMPLE_RATE, impact_data)

        logger.info(f"Synthesized 4 broadcast SFX files to: {out_p}")
        return {k: str(v) for k, v in sfx_files.items()}
