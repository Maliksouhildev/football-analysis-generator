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


def synthesize_slow_focus_swell(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Subtle sub-bass cinematic inhale/time-slow audio swell (36Hz -> 84Hz sweep over 0.65s)
    layered with bandpass filtered air texture for dynamic slow-focus deceleration moments.
    """
    duration = 0.65
    n_samples = int(duration * sr)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    # 1. Low frequency exponential glide (36Hz -> 84Hz)
    f_start = 36.0
    f_end = 84.0
    phase = 2.0 * np.pi * (f_start * t + ((f_end - f_start) / (2.0 * duration)) * (t ** 2))
    sub_bass = np.sin(phase)

    # Inhale crescendo envelope: smooth buildup peaking near 0.52s, gentle release
    peak_t = 0.52
    env_sub = np.where(
        t <= peak_t,
        (t / peak_t) ** 1.7,
        np.exp(-(t - peak_t) / 0.05),
    )
    sub_bass = sub_bass * env_sub

    # 2. Filtered air inhale texture (bandpass 180Hz - 700Hz)
    noise = np.random.normal(0.0, 1.0, n_samples)
    b, a = butter(2, [180.0 / (sr / 2.0), 700.0 / (sr / 2.0)], btype="band")
    filtered_noise = lfilter(b, a, noise)
    env_noise = np.where(
        t <= peak_t,
        (t / peak_t) ** 2.0,
        np.exp(-(t - peak_t) / 0.04),
    )
    air_texture = filtered_noise * env_noise

    # Layer: 75% sub-bass + 25% subtle air inhale
    mix = 0.75 * sub_bass + 0.25 * air_texture
    return _to_stereo_pcm16(mix, peak_norm=0.50)


def synthesize_run_chevrons(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Crisp high-velocity aerodynamic whoosh (140ms) layered with 4 rapid micro-ticks
    synced to sequential chevron pops along player run paths.
    """
    duration = 0.20
    n_samples = int(duration * sr)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    # 1. Rising aerodynamic whoosh (bandpass sweep 380Hz -> 1600Hz)
    noise = np.random.normal(0.0, 1.0, n_samples)
    b, a = butter(2, [380.0 / (sr / 2.0), 1600.0 / (sr / 2.0)], btype="band")
    filtered_noise = lfilter(b, a, noise)
    env_whoosh = (np.sin(np.pi * (t / duration)) ** 1.8) * np.exp(-t / 0.12)
    whoosh = filtered_noise * env_whoosh

    # 2. Sequential micro-ticks (4 ticks spaced by 0.035s)
    ticks = np.zeros_like(t)
    tick_times = [0.015, 0.050, 0.085, 0.120]
    for idx, t_tick in enumerate(tick_times):
        dt = t - t_tick
        mask = dt >= 0
        freq = 1600.0 + idx * 250.0
        decay = 0.005
        tick_wave = np.sin(2.0 * np.pi * freq * dt) * np.exp(-dt / decay) * mask
        ticks += 0.35 * tick_wave

    mix = 0.65 * whoosh + 0.50 * ticks
    max_val = np.max(np.abs(mix))
    if max_val > 1e-6:
        mix = mix * (0.65 / max_val)

    pcm = np.int16(np.clip(mix, -1.0, 1.0) * 32767)
    return np.column_stack((pcm, pcm))


def synthesize_error_alarm(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Punchy tactical error/alert sound effect (0.28s):
    - Sub-bass thud (115Hz -> 48Hz exponential drop)
    - Dissonant high-tech alert ping (dissonant tritone 440Hz + 466Hz damped chord)
    - Rapid filtered warning noise burst (800Hz - 3200Hz).
    """
    duration = 0.28
    n_samples = int(duration * sr)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    # 1. Punchy low-end impact (115Hz -> 48Hz pitch drop)
    k_pitch = 28.0
    f_base = 48.0
    f_delta = 67.0
    phase_sub = 2.0 * np.pi * (f_base * t + (f_delta / k_pitch) * (1.0 - np.exp(-k_pitch * t)))
    sub_thud = np.sin(phase_sub) * np.exp(-t / 0.08)

    # 2. Dissonant broadcast warning alert transient (440Hz + 466Hz minor 2nd clash)
    alert_wave = (
        0.55 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.45 * np.sin(2.0 * np.pi * 466.16 * t)
        + 0.25 * np.sin(2.0 * np.pi * 880.0 * t)
    )
    env_alert = np.minimum(t / 0.005, 1.0) * np.exp(-t / 0.075)
    alert_sound = alert_wave * env_alert

    # 3. Filtered air warning transient burst (800Hz - 3200Hz)
    noise = np.random.normal(0.0, 1.0, n_samples)
    b, a = butter(2, [800.0 / (sr / 2.0), 3200.0 / (sr / 2.0)], btype="band")
    filtered_noise = lfilter(b, a, noise)
    sine_half = np.maximum(0.0, np.sin(np.pi * (t / 0.08)))
    env_noise = (sine_half ** 1.5) * (t < 0.08)
    noise_burst = filtered_noise * env_noise

    # Layer: 55% sub thud + 45% alert ping + 25% noise burst
    mix = 0.55 * sub_thud + 0.45 * alert_sound + 0.25 * noise_burst
    return _to_stereo_pcm16(mix, peak_norm=0.72)


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
            "slow_focus_swell": out_p / "slow_focus_swell.wav",
            "sfx_slow_focus_swell": out_p / "sfx_slow_focus_swell.wav",
            "sfx_run_chevrons": out_p / "sfx_run_chevrons.wav",
            "sfx_error_alarm": out_p / "sfx_error_alarm.wav",
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

        swell_data = synthesize_slow_focus_swell()
        wavfile.write(str(sfx_files["slow_focus_swell"]), SAMPLE_RATE, swell_data)
        wavfile.write(str(sfx_files["sfx_slow_focus_swell"]), SAMPLE_RATE, swell_data)

        chevrons_data = synthesize_run_chevrons()
        wavfile.write(str(sfx_files["sfx_run_chevrons"]), SAMPLE_RATE, chevrons_data)

        alarm_data = synthesize_error_alarm()
        wavfile.write(str(sfx_files["sfx_error_alarm"]), SAMPLE_RATE, alarm_data)

        logger.info(f"Synthesized 7 broadcast SFX files to: {out_p}")
        return {k: str(v) for k, v in sfx_files.items()}
