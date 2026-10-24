"""
src/ingestion/stream_buffer.py
Stream Ingestion & Circular Buffering Module.

Continuously captures live RTMP/HLS broadcast streams or loops local video files
into a rolling circular buffer of 2.0-second MPEG-TS segments (120-minute sliding window).
Manages live manifest.m3u8, automatic disk retention pruning with Windows file-lock safety,
and thread-safe segment queries.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ingestion.stream_buffer")


@dataclass(frozen=True)
class SegmentMetadata:
    """Metadata representing an ingested 2.0-second transport stream segment."""
    sequence: int
    filename: str
    filepath: Path
    duration: float
    start_pts: float
    end_pts: float
    created_at: float
    file_size_bytes: int
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.sequence,
            "filename": self.filename,
            "filepath": str(self.filepath),
            "duration": self.duration,
            "start_pts": self.start_pts,
            "end_pts": self.end_pts,
            "size_bytes": self.file_size_bytes,
            "is_valid": self.is_valid,
        }


@dataclass
class StreamBufferConfig:
    """Configuration settings for StreamBufferManager."""
    buffer_dir: Path = Path("data/buffer")
    segment_duration: float = 2.0
    retention_minutes: float = 120.0
    ffmpeg_binary: str = "ffmpeg"
    probe_binary: str = "ffprobe"
    prune_interval_sec: float = 5.0
    reconnect_delay_max: int = 5
    socket_timeout_sec: int = 15
    startup_timeout_sec: float = 6.0


class StreamBufferManager:
    """
    Manages continuous FFmpeg ingestion into an indexed circular buffer of 2.0s .ts chunks.
    Maintains live manifest.m3u8 and provides thread-safe access to media segments.
    """

    def __init__(
        self,
        config: Optional[StreamBufferConfig] = None,
        max_segments: Optional[int] = None,
        segment_duration: Optional[float] = None,
    ):
        self.config = config or StreamBufferConfig()
        if segment_duration is not None:
            self.config.segment_duration = float(segment_duration)
        if max_segments is not None:
            self.config.retention_minutes = (max_segments * self.config.segment_duration) / 60.0

        self.buffer_dir = Path(self.config.buffer_dir).resolve()
        self.manifest_path = self.buffer_dir / "manifest.m3u8"

        self._lock = threading.RLock()
        self._segments: Dict[int, SegmentMetadata] = {}
        self._accumulated_pts = 0.0
        self._next_synthetic_seq = 0

        self._process: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._prune_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._source: Optional[str] = None
        self._is_ingesting = False

        # Register exit hook to prevent orphaned processes
        atexit.register(self._emergency_cleanup)

    @property
    def max_segments(self) -> int:
        """Maximum number of segments in the rolling window."""
        return max(1, int(self.config.retention_minutes * 60.0 / self.config.segment_duration))

    @property
    def segment_duration(self) -> float:
        return self.config.segment_duration

    @property
    def segments(self) -> List[Dict[str, Any]]:
        """List representation of active segments for test harness compatibility."""
        with self._lock:
            return [s.to_dict() for s in sorted(self._segments.values(), key=lambda x: x.sequence)]

    def add_segment(
        self,
        start_pts: float,
        size_bytes: int = 1500000,
        duration: Optional[float] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Thread-safe programmatic segment injection (useful for simulation & testing).
        Updates internal registry and executes circular buffer pruning.
        """
        with self._lock:
            dur = duration if duration is not None else self.config.segment_duration
            seq_id = self._next_synthetic_seq
            self._next_synthetic_seq += 1

            fname = filename or f"chunk_{seq_id:04d}.ts"
            file_path = self.buffer_dir / fname

            end_pts = start_pts + dur
            metadata = SegmentMetadata(
                sequence=seq_id,
                filename=fname,
                filepath=file_path,
                duration=dur,
                start_pts=start_pts,
                end_pts=end_pts,
                created_at=time.time(),
                file_size_bytes=size_bytes,
                is_valid=True,
            )
            self._segments[seq_id] = metadata
            self._accumulated_pts = max(self._accumulated_pts, end_pts)
            self._prune_segments()
            return metadata.to_dict()

    def start_ingestion(
        self,
        source: str,
        is_loop: bool = False,
        realtime: bool = True,
        force_transcode: bool = False,
    ) -> None:
        """
        Starts stream capture in a managed background subprocess and launches
        segment monitoring and pruning worker threads.
        """
        with self._lock:
            if self._is_ingesting:
                logger.warning("Stream ingestion is already active.")
                return

            self._source = source
            self.buffer_dir.mkdir(parents=True, exist_ok=True)
            self._stop_event.clear()
            self._segments.clear()
            self._accumulated_pts = 0.0

            cmd = self._build_ffmpeg_command(source, is_loop, realtime, force_transcode)
            logger.info("Starting FFmpeg ingestion: %s", " ".join(cmd))

            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    creationflags=creation_flags,
                )
            except Exception as e:
                logger.error("Failed to spawn FFmpeg process: %s", e)
                raise RuntimeError(f"FFmpeg process launch failed: {e}") from e

            self._start_time = time.time()
            self._is_ingesting = True

            # Launch background worker threads
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, name="StreamBufferMonitor", daemon=True
            )
            self._prune_thread = threading.Thread(
                target=self._prune_loop, name="StreamBufferPruner", daemon=True
            )
            self._monitor_thread.start()
            self._prune_thread.start()

        # Await startup verification
        self._verify_startup()

    def stop_ingestion(self, timeout: float = 5.0) -> None:
        """
        Gracefully terminates FFmpeg ingestion and shuts down worker threads.
        """
        with self._lock:
            if not self._is_ingesting and not self._process:
                return

            logger.info("Stopping stream ingestion...")
            self._stop_event.set()
            self._is_ingesting = False

            if self._process and self._process.poll() is None:
                try:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        logger.warning("FFmpeg did not exit within %ss; killing.", timeout)
                        self._process.kill()
                        self._process.wait(timeout=2.0)
                except Exception as e:
                    logger.error("Error terminating FFmpeg: %s", e)
                finally:
                    self._process = None

        # Join worker threads
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        if self._prune_thread and self._prune_thread.is_alive():
            self._prune_thread.join(timeout=2.0)

        # Final sweep of manifest to register any trailing segments
        self._poll_manifest()
        logger.info("Stream ingestion stopped. Active segments: %d", len(self._segments))

    def get_segment_list(self, last_n_minutes: Optional[float] = None) -> List[SegmentMetadata]:
        """
        Returns an ordered list of active segments in the circular buffer.
        """
        with self._lock:
            segments = sorted(self._segments.values(), key=lambda s: s.sequence)
            if not segments or last_n_minutes is None:
                return segments

            cutoff_sec = last_n_minutes * 60.0
            latest_end_pts = segments[-1].end_pts
            min_pts = max(0.0, latest_end_pts - cutoff_sec)
            return [s for s in segments if s.end_pts >= min_pts]

    def get_active_buffer_duration(self) -> float:
        """
        Calculates the total continuous duration (seconds) currently buffered.
        """
        with self._lock:
            if not self._segments:
                return 0.0
            segments = sorted(self._segments.values(), key=lambda s: s.sequence)
            return max(0.0, segments[-1].end_pts - segments[0].start_pts)

    def get_segment_for_timestamp(self, target_pts: float) -> Optional[SegmentMetadata]:
        """
        Finds the segment containing the requested presentation timestamp (PTS).
        """
        with self._lock:
            for s in self._segments.values():
                if s.start_pts <= target_pts <= s.end_pts:
                    return s
            return None

    def get_manifest_path(self) -> Path:
        """Returns the path to manifest.m3u8."""
        return self.manifest_path

    def get_manifest(self) -> str:
        """
        Generates or reads the live HLS m3u8 manifest.
        """
        with self._lock:
            if self.manifest_path.exists():
                try:
                    return self.manifest_path.read_text(encoding="utf-8")
                except Exception:
                    pass

            # Generate dynamically from active segments
            segments = sorted(self._segments.values(), key=lambda s: s.sequence)
            seq_base = segments[0].sequence if segments else 0
            lines = [
                "#EXTM3U",
                "#EXT-X-VERSION:3",
                f"#EXT-X-TARGETDURATION:{int(self.config.segment_duration)}",
                f"#EXT-X-MEDIA-SEQUENCE:{seq_base}"
            ]
            for seg in segments[-10:]:
                lines.append(f"#EXTINF:{seg.duration:.3f},")
                lines.append(seg.filename)
            return "\n".join(lines) + "\n"

    def is_ingesting(self) -> bool:
        """Returns True if the ingestion process is currently running."""
        return self._is_ingesting and (self._process is not None) and (self._process.poll() is None)

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic health metrics."""
        with self._lock:
            running = self.is_ingesting()
            duration = self.get_active_buffer_duration()
            count = len(self._segments)
            total_bytes = sum(s.file_size_bytes for s in self._segments.values())
            uptime = (time.time() - self._start_time) if self._start_time else 0.0
            pid = self._process.pid if self._process and self._process.poll() is None else None

            return {
                "running": running,
                "pid": pid,
                "source": self._source,
                "segment_count": count,
                "max_segments": self.max_segments,
                "buffer_duration_sec": round(duration, 2),
                "total_size_mb": round(total_bytes / (1024 * 1024), 2),
                "uptime_sec": round(uptime, 2),
                "manifest_exists": self.manifest_path.exists(),
            }

    def prune(self) -> int:
        """Alias for _prune_segments for test harness compatibility."""
        return self._prune_segments()

    def cleanup(self, remove_all: bool = False) -> int:
        """
        Removes buffer files. If remove_all=True, deletes all chunk files and manifest.
        """
        removed = 0
        with self._lock:
            if remove_all:
                if self.buffer_dir.exists():
                    for f in self.buffer_dir.glob("chunk_*.ts"):
                        try:
                            f.unlink(missing_ok=True)
                            removed += 1
                        except PermissionError:
                            pass
                    if self.manifest_path.exists():
                        try:
                            self.manifest_path.unlink()
                        except PermissionError:
                            pass
                self._segments.clear()
            else:
                removed = self._prune_segments()
        return removed

    # -------------------------------------------------------------------------
    # Internal Helpers & Worker Loops
    # -------------------------------------------------------------------------

    def _build_ffmpeg_command(
        self, source: str, is_loop: bool, realtime: bool, force_transcode: bool
    ) -> List[str]:
        cmd = [self.config.ffmpeg_binary, "-y"]

        is_network = any(
            source.startswith(proto)
            for proto in ("http://", "https://", "rtmp://", "rtmps://", "rtsp://", "udp://", "srt://")
        )
        is_synthetic = source.startswith("synthetic") or source.startswith("testsrc")

        if is_network:
            cmd.extend([
                "-reconnect", "1",
                "-reconnect_at_eof", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", str(self.config.reconnect_delay_max),
                "-rw_timeout", str(self.config.socket_timeout_sec * 1000000),
                "-fflags", "+genpts+nobuffer+discardcorrupt",
                "-flags", "low_delay",
                "-i", source,
            ])
        elif is_synthetic:
            # Synthetic source via lavfi
            src_filter = "testsrc=size=1280x720:rate=25" if source in ("synthetic", "testsrc") else source
            if realtime:
                cmd.extend(["-re"])
            cmd.extend(["-f", "lavfi", "-i", src_filter])
        else:
            # Local video file
            file_path = Path(source).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"Source video file not found: {file_path}")

            if is_loop:
                cmd.extend(["-stream_loop", "-1"])
            if realtime:
                cmd.extend(["-re"])
            cmd.extend(["-i", str(file_path)])

        # Codec configuration
        if force_transcode or is_synthetic:
            gop = int(self.config.segment_duration * 25)
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-g", str(gop),
                "-keyint_min", str(gop),
                "-sc_threshold", "0",
                "-c:a", "aac",
                "-b:a", "128k",
            ])
        else:
            # Passthrough stream copy (lowest CPU overhead)
            cmd.extend(["-c", "copy"])

        # Segment muxer configuration
        chunk_pattern = str(self.buffer_dir / "chunk_%06d.ts")
        manifest_str = str(self.manifest_path)

        cmd.extend([
            "-f", "segment",
            "-segment_time", str(self.config.segment_duration),
            "-segment_time_delta", "0.05",
            "-segment_format", "mpegts",
            "-segment_list_flags", "live",
            "-segment_list_size", str(self.max_segments),
            "-segment_list", manifest_str,
            "-reset_timestamps", "0",
            "-avoid_negative_ts", "make_zero",
            chunk_pattern,
        ])
        return cmd

    def _verify_startup(self) -> None:
        """Polls for initial segment/manifest creation within startup_timeout_sec."""
        deadline = time.time() + self.config.startup_timeout_sec
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode("utf-8", errors="replace")
                self._is_ingesting = False
                raise RuntimeError(
                    f"FFmpeg process exited immediately (code {self._process.returncode}):\n{stderr[-1000:]}"
                )

            if self.manifest_path.exists() and len(self._segments) > 0:
                logger.info("Startup verified: first segment registered.")
                return
            self._poll_manifest()
            time.sleep(0.2)

        if not self.manifest_path.exists():
            logger.warning("Startup timeout reached before manifest.m3u8 appeared.")

    def _monitor_loop(self) -> None:
        """Background worker polling manifest.m3u8 and updating segment registry."""
        while not self._stop_event.is_set():
            self._poll_manifest()
            time.sleep(0.5)

    def _poll_manifest(self) -> None:
        """Parses manifest.m3u8 and registers newly completed chunks."""
        if not self.manifest_path.exists():
            return

        try:
            content = self.manifest_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        lines = content.splitlines()
        seq_match = re.search(r"#EXT-X-MEDIA-SEQUENCE:(\d+)", content)
        seq_base = int(seq_match.group(1)) if seq_match else 0

        entries = []
        cur_duration = None
        for line in lines:
            line = line.strip()
            if line.startswith("#EXTINF:"):
                try:
                    cur_duration = float(line.split(":")[1].split(",")[0])
                except (ValueError, IndexError):
                    cur_duration = self.config.segment_duration
            elif line.endswith(".ts") and cur_duration is not None:
                entries.append((line, cur_duration))
                cur_duration = None

        with self._lock:
            for idx, (filename, duration) in enumerate(entries):
                seq_id = self._extract_seq_from_filename(filename, default=seq_base + idx)
                if seq_id not in self._segments:
                    file_path = self.buffer_dir / filename
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                    start_pts = self._accumulated_pts
                    end_pts = start_pts + duration
                    self._accumulated_pts = end_pts

                    metadata = SegmentMetadata(
                        sequence=seq_id,
                        filename=filename,
                        filepath=file_path,
                        duration=duration,
                        start_pts=start_pts,
                        end_pts=end_pts,
                        created_at=time.time(),
                        file_size_bytes=file_size,
                        is_valid=file_path.exists() and file_size > 0,
                    )
                    self._segments[seq_id] = metadata

    def _prune_loop(self) -> None:
        """Background worker periodically trimming segments exceeding sliding window."""
        while not self._stop_event.is_set():
            time.sleep(self.config.prune_interval_sec)
            self._prune_segments()

    def _prune_segments(self) -> int:
        """Deletes expired segments outside the retention window with Windows lock tolerance."""
        with self._lock:
            if len(self._segments) <= self.max_segments:
                return 0

            sorted_seqs = sorted(self._segments.keys())
            overflow = len(sorted_seqs) - self.max_segments
            candidates = sorted_seqs[:overflow]

        pruned = 0
        for seq in candidates:
            with self._lock:
                meta = self._segments.get(seq)
            if not meta:
                continue

            try:
                if meta.filepath.exists():
                    meta.filepath.unlink(missing_ok=True)
                with self._lock:
                    self._segments.pop(seq, None)
                pruned += 1
            except PermissionError:
                # Locked on Windows by an active reader; defer to next cycle
                logger.debug("Segment %s locked; deferring deletion", meta.filename)
            except Exception as e:
                logger.warning("Error deleting segment %s: %s", meta.filename, e)

        return pruned

    def _extract_seq_from_filename(self, filename: str, default: int) -> int:
        """Extracts integer sequence number from chunk_######.ts."""
        m = re.search(r"chunk_(\d+)\.ts", filename)
        if m:
            return int(m.group(1))
        return default

    def _emergency_cleanup(self) -> None:
        """Hook called at interpreter shutdown to kill orphaned FFmpeg."""
        if self._process and self._process.poll() is None:
            try:
                self._process.kill()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_ingestion()


# Backward-compatible alias
CircularBufferManager = StreamBufferManager
