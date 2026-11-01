import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Easing,
  interpolate,
  Sequence,
  useCurrentFrame,
} from 'remotion';

export interface SpeedSegment {
  segment_type: 'FAST_TRANSIT' | 'NORMAL_FLOW' | 'SLOW_FOCUS' | 'FREEZE_HOLD';
  rate: number;
  source_start_sec?: number;
  source_end_sec?: number;
  duration_sec?: number;
  start_frame?: number;
  end_frame?: number;
  startFrame?: number;
  endFrame?: number;
  reason?: string;
}

export interface SplitCanvasCameraRigProps {
  children: React.ReactNode;
  speedSegments?: SpeedSegment[];
  speed_segments?: SpeedSegment[];
  focusX?: number; // 0 to 100 percentage
  focusY?: number; // 0 to 100 percentage
  baseScale?: number;
  shakeX?: number; // Micro-camera tremor on tactical impact / mistake alarm
  shakeY?: number;
}

/**
 * SplitCanvasCameraRig
 * 
 * Provides unified camera framing and subtle visual & audio cues during "SLOW_FOCUS" moments:
 * - When entering a "SLOW_FOCUS" (0.60x-0.75x) window:
 *   * Applies a subtle micro-push zoom (scale: 1.0 -> 1.12 with Easing.inOut(Easing.sin))
 *   * Adds gentle vignette edge darkening (opacity: 0.25) to focus viewer attention on the micro-technique
 *   * Triggers a low sub-bass time-slow audio swell (slow_focus_swell.wav) at volume 0.22 right as video decelerates
 */
export const SplitCanvasCameraRig: React.FC<SplitCanvasCameraRigProps> = ({
  children,
  speedSegments,
  speed_segments,
  focusX = 50,
  focusY = 50,
  baseScale = 1.0,
  shakeX = 0,
  shakeY = 0,
}) => {
  const frame = useCurrentFrame();
  const segments = speedSegments || speed_segments || [];

  // Find active SLOW_FOCUS segment for current frame
  let activeSlowFocusSeg: SpeedSegment | null = null;
  const slowFocusSegments: SpeedSegment[] = [];

  for (const seg of segments) {
    if (seg.segment_type === 'SLOW_FOCUS') {
      slowFocusSegments.push(seg);
      const sFrame = seg.start_frame ?? seg.startFrame ?? 0;
      const eFrame = seg.end_frame ?? seg.endFrame ?? 0;
      if (frame >= sFrame && frame < eFrame) {
        activeSlowFocusSeg = seg;
      }
    }
  }

  // Subtle slow-focus vignette & camera steadiness (suppress jittery micro-zoom bounce)
  let slowFocusScale = 1.0;
  let vignetteOpacity = 0.0;

  if (activeSlowFocusSeg && baseScale <= 1.02) {
    const sFrame = activeSlowFocusSeg.start_frame ?? activeSlowFocusSeg.startFrame ?? 0;
    const eFrame = activeSlowFocusSeg.end_frame ?? activeSlowFocusSeg.endFrame ?? 0;
    const dur = Math.max(1, eFrame - sFrame);
    const relFrame = frame - sFrame;

    // Only apply subtle drift if segment is at least 24 frames (~0.8s) to prevent jittery micro-zoom pulse
    const maxSlowScale = dur >= 24 ? 1.025 : 1.0;
    const rampIn = Math.min(12, Math.max(6, Math.floor(dur * 0.25)));
    const rampOut = Math.min(12, Math.max(6, Math.floor(dur * 0.25)));
    const midStart = rampIn;
    const midEnd = Math.max(midStart, dur - rampOut);

    if (relFrame < midStart) {
      const p = relFrame / Math.max(1, midStart);
      const eased = 0.5 - 0.5 * Math.cos(p * Math.PI);
      slowFocusScale = 1.0 + (maxSlowScale - 1.0) * eased;
      vignetteOpacity = 0.18 * eased;
    } else if (relFrame <= midEnd) {
      slowFocusScale = maxSlowScale;
      vignetteOpacity = 0.18;
    } else {
      const p = (relFrame - midEnd) / Math.max(1, dur - midEnd);
      const eased = 0.5 - 0.5 * Math.cos(p * Math.PI);
      slowFocusScale = maxSlowScale - (maxSlowScale - 1.0) * eased;
      vignetteOpacity = 0.18 * (1.0 - eased);
    }
  }

  // When player focus or freeze zoom is active (baseScale > 1.001), lock strictly to baseScale with zero scale popping
  const effectiveSlowScale = baseScale > 1.001 ? 1.0 : slowFocusScale;
  const effectiveScale = baseScale * effectiveSlowScale;
  const safeX = Math.max(10, Math.min(90, focusX));
  const safeY = Math.max(10, Math.min(90, focusY));

  // Compute smooth camera translation to dynamically center the focused player under zoom
  // With transformOrigin: '50% 50%', full centering shifts the targeted player directly to (50% 50%)
  // Clamped to [-txMax, txMax] and [-tyMax, tyMax] to strictly prevent revealing any black canvas boundaries
  const txMax = (1920 * Math.max(0, effectiveScale - 1.0)) / 2.0;
  const tyMax = (1080 * Math.max(0, effectiveScale - 1.0)) / 2.0;

  const dx = (safeX - 50.0) * 19.2;
  const dy = (safeY - 50.0) * 10.8;

  // Full centering translation smoothly scaled by available zoom margin ratio,
  // decoupled from micro-scale breathing once effectiveScale >= 1.42 so X/Y pan never recoils during inter-player glides
  const scaleMargin = Math.max(0, effectiveScale - 1.0);
  const panWeight = Math.min(1.0, scaleMargin / 0.42);
  const rawTransX = panWeight > 0 ? -dx * panWeight : 0;
  const rawTransY = panWeight > 0 ? -dy * panWeight : 0;

  const transX = Math.max(-txMax, Math.min(txMax, rawTransX));
  const transY = Math.max(-tyMax, Math.min(tyMax, rawTransY));

  // Broadcast steadiness: Zero artificial camera tremor or vibration
  const finalTransX = transX;
  const finalTransY = transY;

  return (
    <AbsoluteFill style={{ overflow: 'hidden' }}>
      {/* Audio swell on each SLOW_FOCUS segment entry */}
      {slowFocusSegments.map((seg, idx) => {
        const sFrame = seg.start_frame ?? seg.startFrame ?? 0;
        const eFrame = seg.end_frame ?? seg.endFrame ?? 0;
        const dur = Math.max(1, eFrame - sFrame);
        return (
          <Sequence key={`slow-focus-swell-${idx}`} from={sFrame} durationInFrames={Math.min(30, dur)}>
            <Audio src="http://127.0.0.1:3000/assets/audio/sfx/slow_focus_swell.wav" volume={0.22} />
          </Sequence>
        );
      })}

      {/* Camera-transformed Canvas Container with fluid subpixel interpolation */}
      <AbsoluteFill
        style={{
          transform: `translate(${finalTransX.toFixed(3)}px, ${finalTransY.toFixed(3)}px) scale(${effectiveScale.toFixed(5)})`,
          transformOrigin: '50% 50%',
          willChange: 'transform',
        }}
      >
        {children}
      </AbsoluteFill>

      {/* Gentle Vignette Edge Darkening (0.25 opacity) during SLOW_FOCUS moments */}
      {vignetteOpacity > 0 && (
        <AbsoluteFill
          style={{
            pointerEvents: 'none',
            opacity: vignetteOpacity,
            background: 'radial-gradient(circle at center, transparent 40%, rgba(0, 0, 0, 0.9) 100%)',
            mixBlendMode: 'multiply',
          }}
        />
      )}
    </AbsoluteFill>
  );
};
