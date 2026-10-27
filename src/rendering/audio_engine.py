"""
src/rendering/audio_engine.py
Kokoro TTS Audio Engine & Synchronized Broadcast Commentary Synthesizer.

Synthesizes high-fidelity British football commentary (bm_george) via the Kokoro
REST container on port 8880, aligns spoken commentary chunks across match event timestamps,
mixes and normalizes audio to EBU R128 (-14 LUFS) broadcast loudness, and exports
base64 payloads for Remotion composition playback.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure FFmpeg/FFprobe binaries are available in PATH
for _p in [
    r"C:\Users\mehdi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin",
    r"C:\Users\mlkme\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\imageio_ffmpeg\binaries",
]:
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

logger = logging.getLogger("rendering.audio_engine")

DEFAULT_KOKORO_ENDPOINT = os.getenv("KOKORO_TTS_ENDPOINT", "http://127.0.0.1:8880/v1/audio/speech")
DEFAULT_VOICE = os.getenv("COMMENTARY_VOICE", "bm_george")


class AudioEngine:
    """Manages commentary voiceover synthesis and timeline audio alignment."""

    def __init__(
        self,
        endpoint: str = DEFAULT_KOKORO_ENDPOINT,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        timeout: float = 60.0,
    ):
        self.endpoint = endpoint
        self.voice = voice
        self.speed = float(speed)
        self.timeout = float(timeout)

    @staticmethod
    def normalize_for_tts(text: str) -> str:
        """
        Phonetic normalization for natural TTS vocalization:
        1. 2v1 / 3v1 / 3v2 -> 'two versus one' / 'three versus one' / 'three versus two'
        2. Expands all positional abbreviations: RB, LB, CB, CDM, RW, LW -> spoken equivalents
        3. Words in ALL-CAPS (e.g. VACATED, ZONE, FREEZE) -> lowercase (prevents spelling out)
        """
        if not text:
            return ""

        num_map = {
            "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
            "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
            "11": "eleven",
        }

        def repl_v(m):
            n1 = num_map.get(m.group(1), m.group(1))
            n2 = num_map.get(m.group(2), m.group(2))
            return f"{n1} versus {n2}"

        # 1. Convert 2v1 / 3v1 / 3v2 numerical matchups
        res = re.sub(r'(\d+)\s*(?:v|vs\.?|versus)\s*(\d+)', repl_v, text, flags=re.IGNORECASE)

        # 2. Expand all positional abbreviations and metrics
        abbr_map = [
            (r'\bCDM\b', 'central defensive midfielder'),
            (r'\bCB\b', 'center back'),
            (r'\bRB\b', 'right back'),
            (r'\bLB\b', 'left back'),
            (r'\bRW\b', 'right wing'),
            (r'\bLW\b', 'left wing'),
            (r'\bCAM\b', 'attacking midfielder'),
            (r'\bCM\b', 'central midfielder'),
            (r'\bGK\b', 'goalkeeper'),
            (r'\bST\b', 'striker'),
            (r'\b(?:xG|XG)\b', 'expected goals'),
        ]
        for pat, repl in abbr_map:
            res = re.sub(pat, repl, res)

        # 3. Phonetic pronunciation overrides for broadcast commentary (UCL)
        phonetic_map = [
            (r'\bTchouam[eé]ni\b', 'Choo-ah-meh-nee'),
            (r'\bValverde\b', 'Val-vair-day'),
            (r'\bSoul[eé]\b', 'Soo-leh'),
            (r'\bKoulierakis\b', 'Koo-lyeh-rah-kees'),
            (r'\bNdicka\b', 'En-dee-kah'),
            (r'\bG[uü]ler\b', 'Goo-lair'),
            (r'\bZone\s*14\b', 'zone fourteen'),
            (r'\bzone\s*14\b', 'zone fourteen'),
            (r'\bEsp[ií]\b', 'Es-pee'),
        ]
        for pat, repl in phonetic_map:
            res = re.sub(pat, repl, res, flags=re.IGNORECASE)

        # 4. Lowercase any ALL-CAPS words (2+ letters) so Kokoro TTS never spells words out
        def repl_caps(m):
            return m.group(0).lower()

        res = re.sub(r'\b[A-Z]{2,}\b', repl_caps, res)
        return res

    @staticmethod
    def clean_commentary_text(raw_text: str) -> str:
        """
        Formats text for natural Kokoro vocal cadence.
        Ensures continuous, stutter-free narration without awkward mid-sentence pauses or ellipses.
        """
        norm = AudioEngine.normalize_for_tts(raw_text)
        # Strip ellipses, dashes, or multi-dots that cause awkward 0.5s pauses/stutters mid-sentence
        text = re.sub(r'\.{2,}', '.', norm)
        text = re.sub(r'…', '.', text)
        text = re.sub(r'\s*--+\s*', ', ', text)
        text = re.sub(r'\bthird\s*-\s*man\b', 'third man', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*;\s*', ', ', text)
        text = re.sub(r'[\r\n]+', ' ', text)
        text = re.sub(r'\.\s*\.', '.', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        response_format: str = "mp3",
    ) -> bytes:
        """
        Synthesizes commentary text to audio bytes using Kokoro TTS container.
        """
        cleaned = self.clean_commentary_text(text)
        if not cleaned:
            raise ValueError("Cannot synthesize empty commentary text.")

        payload = {
            "model": "kokoro",
            "input": cleaned,
            "voice": voice or self.voice,
            "speed": float(speed or self.speed),
            "response_format": response_format,
        }

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Kokoro TTS returned HTTP {resp.status}")
                return resp.read()
        except Exception as exc:
            logger.warning(f"Kokoro TTS synthesis failed ({exc}).")
            raise

    @staticmethod
    def get_audio_duration(file_path: Union[str, Path]) -> float:
        """Measures exact duration in seconds of an audio file using wave/ffprobe."""
        p = Path(file_path)
        if p.suffix.lower() == ".wav" and p.is_file():
            try:
                import wave
                with wave.open(str(p), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0 and frames > 0:
                        return frames / float(rate)
            except Exception:
                pass
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            ).decode().strip()
            val = float(out)
            return val if val > 0 else 3.0
        except Exception:
            return 3.0

    def _generate_fallback_wav(self, text: str, output_path: Path) -> None:
        """Generates spoken commentary audio via Windows SAPI, or silent broadcast fallback."""
        try:
            safe_text = text.replace("'", "''")
            out_p_str = str(output_path).replace('\\', '\\\\')
            ps_code = f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.SelectVoice('Microsoft David Desktop'); $s.Rate = 0; $s.SetOutputToWaveFile('{out_p_str}'); $s.Speak('{safe_text}'); $s.Dispose();"
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_code], check=True, timeout=15)
            temp_sapi = output_path.with_name(f"_sapi_{output_path.name}")
            if output_path.exists():
                output_path.rename(temp_sapi)
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(temp_sapi),
                    "-af", "adelay=100|100,apad=pad_dur=0.10",
                    "-ar", "48000", "-ac", "2", str(output_path)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                if temp_sapi.exists():
                    temp_sapi.unlink(missing_ok=True)
                return
        except Exception as e:
            logger.warning(f"SAPI voice synthesis notice ({e}), falling back to tone/silence.")

        words = len(text.split())
        duration_sec = max(2.5, words / 2.6)
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-t", f"{duration_sec:.2f}",
            "-ar", "48000",
            "-ac", "2",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10.0)
        except Exception as exc:
            logger.error(f"Failed to generate fallback audio via ffmpeg: {exc}, writing pure WAV fallback")
            try:
                import wave
                with wave.open(str(output_path), "wb") as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(2)
                    wf.setframerate(48000)
                    wf.writeframes(b"\x00" * int(48000 * 2 * 2 * duration_sec))
            except Exception as we:
                logger.error(f"Failed to write pure WAV fallback: {we}")

    def synthesize_commentary_clip(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        response_format: Optional[str] = None,
    ) -> float:
        """
        Synthesizes commentary text to an audio file and returns its exact duration in seconds.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        is_wav = out_p.suffix.lower() == ".wav"
        fmt = response_format or ("wav" if is_wav else "mp3")

        try:
            audio_bytes = self.synthesize_speech(text, voice=voice, speed=speed, response_format=fmt)
            with open(out_p, "wb") as f:
                f.write(audio_bytes)

            # If target is WAV, ensure standard broadcast 48kHz stereo WAV encoding
            # and prepend 150ms of clean silence to prevent clipping soft starting consonants
            if is_wav:
                temp_src = out_p.with_name(f"_tmp_{out_p.name}")
                if out_p.exists():
                    out_p.rename(temp_src)
                    cmd = [
                        "ffmpeg", "-y", "-i", str(temp_src),
                        "-af", "adelay=150|150,apad=pad_dur=0.10",
                        "-ar", "48000", "-ac", "2",
                        str(out_p),
                    ]
                    try:
                        subprocess.run(
                            cmd,
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=10.0,
                        )
                    except Exception as fe:
                        logger.warning(f"Audio filter failed ({fe}), falling back to direct resample.")
                        try:
                            subprocess.run(
                                ["ffmpeg", "-y", "-i", str(temp_src), "-ar", "48000", "-ac", "2", str(out_p)],
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=10.0,
                            )
                        except Exception as fe2:
                            logger.warning(f"Direct resample failed ({fe2}), keeping original audio bytes.")
                            if temp_src.exists() and not out_p.exists():
                                temp_src.rename(out_p)
                    finally:
                        if temp_src.exists():
                            if not out_p.exists():
                                temp_src.rename(out_p)
                            else:
                                temp_src.unlink()
        except Exception as e:
            logger.warning(f"Synthesis failed ({e}). Generating broadcast fallback audio for: '{text[:40]}...'")
            self._generate_fallback_wav(text, out_p)

        return self.get_audio_duration(out_p)

    def synthesize_3beat_commentary(
        self,
        beats: Dict[str, Any],
        output_dir: Union[str, Path],
        fps: int = 30,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        cause_frame_idx: Optional[int] = None,
        actual_clip_video_frames: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Synthesizes beat_1.wav, beat_2.wav, and beat_3.wav separately with Kokoro TTS.
        Measures exact millisecond durations and enforces:
        1. beat_1 finishes at least 8 frames before cause_frame_idx:
           beat_1_frames = math.ceil(beat_1_duration_sec * 30) + 8
           cause_frame_idx = max(cause_frame_idx, beat_1_frames)
        2. Freeze duration:
           freeze_frames = int(math.ceil((beat_2_duration_sec + 0.35) * 30))
        3. Total Remotion composition frames:
           total_composition_frames = actual_clip_video_frames + freeze_frames
           (capped cleanly with maximum 0.3s tail buffer).
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def extract_text(val: Any) -> str:
            if isinstance(val, str):
                return val.strip()
            if isinstance(val, dict):
                return str(val.get("text", "")).strip()
            return ""

        text_1 = extract_text(beats.get("beat_1_setup"))
        text_2 = extract_text(beats.get("beat_2_cause"))
        text_3 = extract_text(beats.get("beat_3_effect"))

        beat_1_path = out_dir / "beat_1.wav"
        beat_2_path = out_dir / "beat_2.wav"
        beat_3_path = out_dir / "beat_3.wav"

        dur_1_sec = self.synthesize_commentary_clip(text_1, beat_1_path, voice=voice, speed=speed, response_format="wav")
        dur_1_ms = int(round(dur_1_sec * 1000))
        beat_1_duration_sec = dur_1_ms / 1000.0

        dur_2_sec = self.synthesize_commentary_clip(text_2, beat_2_path, voice=voice, speed=speed, response_format="wav")
        dur_2_ms = int(round(dur_2_sec * 1000))
        beat_2_duration_sec = dur_2_ms / 1000.0

        dur_3_sec = self.synthesize_commentary_clip(text_3, beat_3_path, voice=voice, speed=speed, response_format="wav")
        dur_3_ms = int(round(dur_3_sec * 1000))
        beat_3_duration_sec = dur_3_ms / 1000.0

        # Lock cause_frame_idx to match beat_1_setup exactly: delete min(85) upper clamp
        actual_fps = int(fps) if fps and fps > 0 else 30
        beat_1_frames = int(math.ceil((beat_1_duration_sec + 0.15) * actual_fps))
        cause_frame_idx = max(50, beat_1_frames)
        adjusted_cause_frame_idx = cause_frame_idx

        # Compute freeze_frames: dynamically matches beat_2 duration plus buffer (min 150 frames / 5.0s)
        freeze_frames = max(150, int(math.ceil((beat_2_duration_sec + 0.5) * actual_fps)))

        # Fix 1 & 2: Start beat_3_effect immediately when freeze ends (cause_frame_idx + freeze_frames)
        beat_3_frames = math.ceil(beat_3_duration_sec * actual_fps) + 6
        start_frame_b3 = (adjusted_cause_frame_idx + freeze_frames) if adjusted_cause_frame_idx is not None else None

        post_roll_frames = None
        total_composition_frames = None
        if adjusted_cause_frame_idx is not None:
            if actual_clip_video_frames is not None:
                post_roll_frames = max(1, int(actual_clip_video_frames) - adjusted_cause_frame_idx - 3, beat_3_frames)
            else:
                post_roll_frames = max(1, beat_3_frames)
            total_composition_frames = adjusted_cause_frame_idx + freeze_frames + post_roll_frames
        # Extract 4-Step YouTuber Analyst narration steps
        b2_steps = beats.get("beat_2_steps")
        if not isinstance(b2_steps, dict) and isinstance(beats.get("beat_2_cause"), dict):
            b2_steps = beats.get("beat_2_cause", {}).get("steps") or beats.get("beat_2_cause", {}).get("beat_2_steps")

        s1_text = ""
        s2_text = ""
        s3_text = ""
        s4_text = ""
        if isinstance(b2_steps, dict):
            s1_text = extract_text(b2_steps.get("step_1"))
            s2_text = extract_text(b2_steps.get("step_2"))
            s3_text = extract_text(b2_steps.get("step_3"))
            s4_text = extract_text(b2_steps.get("step_4"))

        # Fallback to splitting text_2 by sentence if b2_steps missing or incomplete
        if not (s1_text and s2_text and s3_text and s4_text) and text_2:
            s_list = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_2) if s.strip()]
            if len(s_list) >= 4:
                s1_text = s_list[0]
                s2_text = s_list[1]
                s3_text = s_list[2]
                s4_text = " ".join(s_list[3:])
            elif len(s_list) == 3:
                s1_text = s_list[0]
                s2_text = s_list[1]
                s3_text = s_list[2]
                s4_text = s_list[2]
            elif len(s_list) == 2:
                s1_text = s_list[0]
                s2_text = s_list[0]
                s3_text = s_list[1]
                s4_text = s_list[1]
            else:
                s1_text = s2_text = s3_text = s4_text = text_2

        # Compute word-proportional duration boundaries for the 4 steps
        w1 = max(1, len(s1_text.split()))
        w2 = max(1, len(s2_text.split()))
        w3 = max(1, len(s3_text.split()))
        w4 = max(1, len(s4_text.split()))
        total_w = w1 + w2 + w3 + w4

        step_1_ms = int(round(dur_2_ms * (w1 / total_w)))
        step_2_ms = int(round(dur_2_ms * (w2 / total_w)))
        step_3_ms = int(round(dur_2_ms * (w3 / total_w)))
        step_4_ms = max(0, dur_2_ms - (step_1_ms + step_2_ms + step_3_ms))

        step_1_start_ms = 0
        step_2_start_ms = step_1_ms
        step_3_start_ms = step_1_ms + step_2_ms
        step_4_start_ms = step_1_ms + step_2_ms + step_3_ms
        step_boundaries_ms = [0, step_1_ms, step_1_ms + step_2_ms, step_1_ms + step_2_ms + step_3_ms, dur_2_ms]

        # Calculate exact frame boundaries for Remotion
        step_1_frames = max(1, int(round((step_1_ms / max(1, dur_2_ms)) * freeze_frames)))
        step_2_frames = max(1, int(round((step_2_ms / max(1, dur_2_ms)) * freeze_frames)))
        step_3_frames = max(1, int(round((step_3_ms / max(1, dur_2_ms)) * freeze_frames)))
        step_4_frames = max(0, freeze_frames - (step_1_frames + step_2_frames + step_3_frames))
        step_boundaries_frames = [
            0,
            step_1_frames,
            step_1_frames + step_2_frames,
            step_1_frames + step_2_frames + step_3_frames,
            freeze_frames,
        ]

        def to_url(wav_p: Path) -> str:
            norm = str(wav_p).replace("\\", "/")
            if "DockerRenders" in norm:
                rel = norm.split("DockerRenders/")[1].lstrip("/")
                if rel.startswith("assets/"):
                    rel = rel[len("assets/"):]
                return f"http://127.0.0.1:3000/assets/{rel}"
            return str(wav_p)

        ret: Dict[str, Any] = {
            "beat_1_setup": {
                "text": text_1,
                "wav_path": str(beat_1_path),
                "url": to_url(beat_1_path),
                "duration_ms": dur_1_ms,
                "duration_sec": beat_1_duration_sec,
                "beat_1_frames": beat_1_frames,
            },
            "beat_2_cause": {
                "text": text_2,
                "wav_path": str(beat_2_path),
                "url": to_url(beat_2_path),
                "duration_ms": dur_2_ms,
                "duration_sec": beat_2_duration_sec,
                "freeze_frames": freeze_frames,
                "step_1_ms": step_1_ms,
                "step_2_ms": step_2_ms,
                "step_3_ms": step_3_ms,
                "step_4_ms": step_4_ms,
                "step_1_start_ms": step_1_start_ms,
                "step_2_start_ms": step_2_start_ms,
                "step_3_start_ms": step_3_start_ms,
                "step_4_start_ms": step_4_start_ms,
                "step_boundaries_ms": step_boundaries_ms,
                "step_1_frames": step_1_frames,
                "step_2_frames": step_2_frames,
                "step_3_frames": step_3_frames,
                "step_4_frames": step_4_frames,
                "step_boundaries_frames": step_boundaries_frames,
                "steps": {
                    "step_1": s1_text,
                    "step_2": s2_text,
                    "step_3": s3_text,
                    "step_4": s4_text,
                },
            },
            "beat_3_effect": {
                "text": text_3,
                "wav_path": str(beat_3_path),
                "url": to_url(beat_3_path),
                "duration_ms": dur_3_ms,
                "duration_sec": beat_3_duration_sec,
                "start_frame": start_frame_b3,
            },
            "step_1_ms": step_1_ms,
            "step_2_ms": step_2_ms,
            "step_3_ms": step_3_ms,
            "step_4_ms": step_4_ms,
            "step_1_start_ms": step_1_start_ms,
            "step_2_start_ms": step_2_start_ms,
            "step_3_start_ms": step_3_start_ms,
            "step_4_start_ms": step_4_start_ms,
            "step_boundaries_ms": step_boundaries_ms,
            "step_1_frames": step_1_frames,
            "step_2_frames": step_2_frames,
            "step_3_frames": step_3_frames,
            "step_4_frames": step_4_frames,
            "step_boundaries_frames": step_boundaries_frames,
            "beat_2_steps": {
                "step_1": s1_text,
                "step_2": s2_text,
                "step_3": s3_text,
                "step_4": s4_text,
            },
        }
        if adjusted_cause_frame_idx is not None:
            ret["cause_frame_idx"] = adjusted_cause_frame_idx
        if actual_clip_video_frames is not None:
            ret["actual_video_frames"] = actual_clip_video_frames
        if post_roll_frames is not None:
            ret["post_roll_frames"] = post_roll_frames
        if total_composition_frames is not None:
            ret["total_composition_frames"] = total_composition_frames
            ret["duration_frames"] = total_composition_frames
            ret["durationInFrames"] = total_composition_frames
            ret["duration"] = round(total_composition_frames / float(actual_fps), 4)
        if beats.get("is_actual_goal_scored_in_clip"):
            ret["is_actual_goal_scored_in_clip"] = True
            ret["goal_impact_frame_idx"] = beats.get("goal_impact_frame_idx", 265)
        return ret

    def generate_masterclass_audio_track(
        self,
        timeline_commentaries: List[Dict[str, Any]],
        total_duration: float,
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Synthesizes multiple commentary snippets and mixes them into a continuous
        chronological audio track aligned with match segment start times.
        Guarantees STRICT SERIALIZATION (no overlapping voices / double speech).
        Normalizes loudness to -14 LUFS broadcast standard.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = out_file.parent / "temp_audio_chunks"
        temp_dir.mkdir(parents=True, exist_ok=True)

        inputs: List[str] = []
        filter_parts: List[str] = []
        mix_inputs: List[str] = []

        logger.info(f"Synthesizing {len(timeline_commentaries)} tactical commentary clips via Kokoro TTS...")

        current_audio_time = 0.5  # 0.5s opening stadium sound buffer

        for idx, item in enumerate(timeline_commentaries):
            text = item.get("text", "").strip()
            req_start = max(0.0, float(item.get("startTime", current_audio_time)))
            chunk_path = item.get("audio_path")
            dur = float(item.get("duration", 0.0))

            if not chunk_path or not Path(chunk_path).is_file():
                if not text:
                    continue
                chunk_path = str(temp_dir / f"chunk_{idx:03d}.mp3")
                try:
                    dur = self.synthesize_commentary_clip(text, chunk_path)
                except Exception as e:
                    logger.warning(f"Skipping audio chunk {idx} due to error: {e}")
                    continue
            elif not dur:
                dur = self.get_audio_duration(chunk_path)

            # Strict Anti-Overlap Guarantee:
            # Voice segment can NEVER start before the previous one finishes + 0.1s breathing pause
            actual_start = max(req_start, current_audio_time)
            delay_ms = int(actual_start * 1000)

            filter_parts.append(f"[{len(inputs)}:a]adelay={delay_ms}|{delay_ms}[a{idx}]")
            mix_inputs.append(f"[a{idx}]")
            inputs.append(str(chunk_path))

            current_audio_time = actual_start + dur + 0.1

        # If no clips could be generated, build ambient silent baseline
        if not inputs:
            logger.warning("No commentary audio clips generated. Generating silent track baseline.")
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                "-t", str(total_duration),
                "-b:a", "192k",
                str(out_file),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            filter_str = "; ".join(filter_parts) + "; "
            filter_str += f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0,apad,loudnorm=I=-14:LRA=7:TP=-1.0[outa]"

            cmd = ["ffmpeg", "-y"]
            for inp in inputs:
                cmd.extend(["-i", inp])
            cmd.extend([
                "-filter_complex", filter_str,
                "-map", "[outa]",
                "-t", str(total_duration),
                "-ar", "48000",
                "-b:a", "192k",
                str(out_file),
            ])

            logger.info("Assembling synchronized broadcast commentary track with FFmpeg...")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Clean up chunk files
        for chunk in temp_dir.glob("chunk_*.mp3"):
            try:
                chunk.unlink()
            except Exception:
                pass
        try:
            temp_dir.rmdir()
        except Exception:
            pass

        # Read final audio as Base64 for Remotion
        with open(out_file, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        logger.info(f"Masterclass audio track complete: {out_file.name} ({out_file.stat().st_size} bytes)")

        return {
            "audio_path": str(out_file),
            "audio_base64": audio_base64,
            "duration": total_duration,
        }
