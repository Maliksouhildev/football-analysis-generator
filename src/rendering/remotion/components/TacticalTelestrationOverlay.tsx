import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Easing,
  Freeze,
  Img,
  interpolate,
  OffthreadVideo,
  Sequence,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  Video,
} from 'remotion';
import { PlayerSofaScoreCard, PlayerStatsData } from './PlayerSofaScoreCard';
import { PlayerHeatmapModal, PlayerHeatmapProfile } from './PlayerHeatmapModal';
import { SplitCanvasCameraRig, SpeedSegment } from './SplitCanvasCameraRig';
export type { SpeedSegment };

export interface TelestrationPlayer {
  id: string;
  name?: string;
  label?: string;
  number?: string | number;
  x: number; // 0 to 1920 (or percentage 0 to 100)
  y: number; // 0 to 1080 (or percentage 0 to 100)
  color?: string; // e.g. '#ffff00', '#a855f7', '#ef4444', '#ffffff'
  showRing?: boolean;
  showBeam?: boolean;
  showBadge?: boolean;
  role?: string; // e.g. 'DM', 'CB', 'LW'
  team?: 'home' | 'away' | string;
}

export interface TelestrationLine {
  id: string;
  playerIds: string[]; // connecting player ids in sequence
  color?: string;
  width?: number;
  dashed?: boolean;
  style?: 'solid' | 'dashed' | 'dotted' | 'bezier';
  lineType?: 'tactical_structure' | 'ball_pass' | 'player_run' | string;
  line_type?: 'tactical_structure' | 'ball_pass' | 'player_run' | string;
  haloPaddingPx?: number;
  halo_padding_px?: number;
  bezierControl?: { x: number; y: number } | [number, number];
  bezier_control?: { x: number; y: number } | [number, number];
}

export interface TelestrationPolygon {
  id: string;
  playerIds?: string[]; // 3 or more player ids forming a shape
  points?: { x: number; y: number }[]; // 3 or more arbitrary coordinates for empty zones/spaces
  fillColor?: string;
  borderColor?: string;
  pattern?: 'solid' | 'stripes' | 'dots';
  label?: string;
  isVacatedZone?: boolean;
}

export interface TrackItem {
  track_id: number | string;
  team?: 'home' | 'away' | string;
  x_pct: number;
  y_pct: number;
  bbox?: number[];
  name?: string;
  label?: string;
  number?: string | number;
  role?: string;
  name_pos?: string;
  color?: string;
}

export interface TacticalCatalystPlayer {
  track_id: number | string;
  name_pos?: string;
  name?: string;
  role?: string;
  reason?: string;
  team?: 'home' | 'away' | string;
}

export interface TacticalStructuralNetwork {
  type?: 'DEFENSIVE_LINE' | 'BUILDUP_POLYGON' | 'PRESSING_TRAP' | string;
  track_ids: (number | string)[];
  closed_polygon?: boolean;
  unit_label?: string;
  vertices_pct?: ([number, number] | { x: number; y: number })[];
}

export interface TacticalExploitZone {
  label?: string;
  vertices_pct: ([number, number] | { x: number; y: number })[];
}

export interface TacticalVectorItem {
  type: 'PLAYER_RUN_DASHED' | 'GROUND_PASS_SOLID' | 'AERIAL_BALL_ARC' | string;
  from_track_id?: number | string;
  name_pos?: string;
  to_pct?: [number, number] | { x: number; y: number };
  from_pct?: [number, number] | { x: number; y: number };
  landing_target_pct?: [number, number] | { x: number; y: number };
  arc_height_pct?: number;
  landing_label?: string;
  label?: string;
  color?: string;
  was_taken?: boolean;
  [key: string]: any;
}

export interface CauseTelestrationData {
  clip_id?: string;
  is_set_piece_or_long_ball?: boolean;
  is_set_piece?: boolean;
  sequence_type?: string;
  focal_track_ids?: (number | string)[];
  backline_track_ids?: (number | string)[];
  midfield_track_ids?: (number | string)[];
  triangle_track_ids?: (number | string)[];
  pressing_track_ids?: (number | string)[];
  attacking_team?: 'home' | 'away' | string;
  defending_team?: 'home' | 'away' | string;
  attacking_jersey?: string;
  defending_jersey?: string;
  vacated_zone_polygon?: ([number, number] | { x: number; y: number })[];
  vacated_zone_ellipse?: {
    cx: number;
    cy: number;
    rx: number;
    ry: number;
    angle?: number;
    color?: string;
  };
  vacated_label?: string;
  ball_handler_track_id?: number | string;
  pass_vector?: {
    from_track_id?: number | string;
    from_x: number;
    from_y: number;
    to_x: number;
    to_y: number;
  };
  run_vectors?: {
    track_id: number | string;
    from_x: number;
    from_y: number;
    to_x: number;
    to_y: number;
  }[];
  overload_callout?: {
    track_id: number | string;
    text: string;
  };

  // V3 Analyst Engine Fields
  video_freeze_frame?: number;
  videoFreezeFrame?: number;
  pre_roll_duration_frames?: number;
  preRollDurationFrames?: number;
  play_type?: string;
  catalyst_player?: TacticalCatalystPlayer;
  runner_player?: TacticalCatalystPlayer;
  runner_track_id?: number | string;
  structural_network?: TacticalStructuralNetwork;
  exploit_zone?: TacticalExploitZone;
  vectors?: TacticalVectorItem[];
  actual_decision_note?: string;
  is_actual_goal_scored_in_clip?: boolean;
  goal_impact_frame_idx?: number | null;
  ball_carrier_per_frame?: Record<string | number, number>;
  players?: any[];
  is_tactical?: boolean;
  isTactical?: boolean;
  is_close_up?: boolean;
  isCloseUp?: boolean;
  shot_type?: string;
  [key: string]: any;
}

export interface BeatAudioData {
  text?: string;
  wav_path?: string;
  url?: string;
  duration_ms?: number;
  duration_sec?: number;
  duration?: number;
  freeze_frames?: number;
}

export interface VisualCueTag {
  type: 'candidate_passes' | 'pass_options' | 'run_vectors' | 'vectors' | 'highlight' | 'stat_card' | 'sofascore' | 'player_stats' | 'pause' | 'freeze' | string;
  target?: string;
  start_frame?: number;
  duration_frames?: number;
  end_frame?: number;
  word_index?: number;
  offset_sec?: number;
  player_stats?: PlayerStatsData;
  player_name?: string;
  player_number?: number | string;
  player_role?: string;
  raw?: string;
  beat?: string;
}

export interface BeatsData {
  beat_1_setup?: BeatAudioData;
  beat_2_cause?: BeatAudioData;
  beat_3_effect?: BeatAudioData;
  visual_cue_tags?: VisualCueTag[];
}

export interface TacticalTelestrationProps {
  players?: TelestrationPlayer[];
  lines?: TelestrationLine[];
  polygons?: TelestrationPolygon[];
  title?: string;
  subtitle?: string;
  delay?: number;
  renderTurfOnly?: boolean;
  renderBadgesOnly?: boolean;

  // Visual Cue Tags & Interactive Commentary HUD
  visual_cue_tags?: VisualCueTag[];
  visualCueTags?: VisualCueTag[];

  // 3-Beat & 3-Phase Choreography Properties
  video_freeze_frame?: number;
  videoFreezeFrame?: number;
  pre_roll_duration_frames?: number;
  preRollDurationFrames?: number;
  cause_frame_idx?: number;
  causeFrameIdx?: number;
  freeze_frames?: number;
  freezeFrames?: number;
  effect_frame_idx?: number;
  effectFrameIdx?: number;
  actual_video_frames?: number;
  actualVideoFrames?: number;
  post_roll_frames?: number;
  postRollFrames?: number;
  tracks_per_frame?: Record<string | number, TrackItem[]>;
  tracksPerFrame?: Record<string | number, TrackItem[]>;
  cause_telestration?: CauseTelestrationData;
  causeTelestration?: CauseTelestrationData;
  ball_carrier_per_frame?: Record<string | number, number>;
  ballCarrierPerFrame?: Record<string | number, number>;
  team_colors?: { home_hex?: string; away_hex?: string };
  teamColors?: { home_hex?: string; away_hex?: string };
  beats?: BeatsData;
  videoUrl?: string;
  video_end_frame?: number;
  videoEndFrame?: number;
  actual_clip_video_frames?: number;
  actualClipVideoFrames?: number;
  total_video_frames?: number;
  totalVideoFrames?: number;
  freezeFrameUrl?: string;
  effectFrameUrl?: string;
  effect_frame_url?: string;
  fgMaskUrl?: string;
  fg_mask_url?: string;
  zoomIntensity?: number;

  // V3 Analyst Engine Props
  is_actual_goal_scored_in_clip?: boolean;
  isActualGoalScoredInClip?: boolean;
  goal_impact_frame_idx?: number | null;
  goalImpactFrameIdx?: number | null;
  catalyst_player?: TacticalCatalystPlayer;
  structural_network?: TacticalStructuralNetwork;
  exploit_zone?: TacticalExploitZone;
  vectors?: TacticalVectorItem[];
  actual_decision_note?: string;
  speed_segments?: SpeedSegment[];
  speedSegments?: SpeedSegment[];
  is_warped?: boolean;
  isWarped?: boolean;
  warped_video_url?: string;
  warpedVideoUrl?: string;
  is_tactical?: boolean;
  isTactical?: boolean;
  is_close_up?: boolean;
  isCloseUp?: boolean;
  shot_type?: string;
}

const resolveAudioSrc = (beat?: BeatAudioData): string | undefined => {
  if (!beat) return undefined;
  if (beat.url) return beat.url;
  if (beat.wav_path) {
    if (beat.wav_path.startsWith('http://') || beat.wav_path.startsWith('https://')) {
      return beat.wav_path;
    }
    // Format local assets for Remotion server static host
    const normalized = beat.wav_path.replace(/\\/g, '/');
    if (normalized.includes('/assets/')) {
      const sub = normalized.split('/assets/')[1];
      return `http://127.0.0.1:3000/assets/${sub}`;
    }
    if (normalized.includes('DockerRenders/')) {
      const sub = normalized.split('DockerRenders/')[1].replace(/^assets\//, '');
      return `http://127.0.0.1:3000/assets/${sub}`;
    }
    if (normalized.includes('/audio/')) {
      const sub = normalized.split('/audio/')[1];
      return `http://127.0.0.1:3000/assets/audio/${sub}`;
    }
    return beat.wav_path;
  }
  return undefined;
};

const findTracksForFrame = (
  tracksRecord?: Record<string | number, TrackItem[]>,
  targetFrame: number = 0,
  maxAllowedDiff: number = 6,
  strictInternalOnly: boolean = false
): TrackItem[] => {
  if (!tracksRecord) return [];
  if (tracksRecord[targetFrame]) return tracksRecord[targetFrame];
  if (tracksRecord[String(targetFrame)]) return tracksRecord[String(targetFrame)];

  const keys = Object.keys(tracksRecord)
    .map(Number)
    .filter((n) => !isNaN(n))
    .sort((a, b) => a - b);

  if (keys.length === 0) return [];

  // Find previous and next available tracked frames for linear interpolation
  let prevKey: number | null = null;
  let nextKey: number | null = null;

  for (const k of keys) {
    if (k <= targetFrame) {
      prevKey = k;
    } else if (k > targetFrame && nextKey === null) {
      nextKey = k;
      break;
    }
  }

  if (prevKey !== null && nextKey !== null) {
    const diffPrev = targetFrame - prevKey;
    const diffNext = nextKey - targetFrame;
    if (diffPrev <= maxAllowedDiff && diffNext <= maxAllowedDiff) {
      const prevTracks = tracksRecord[prevKey] || tracksRecord[String(prevKey)] || [];
      const nextTracks = tracksRecord[nextKey] || tracksRecord[String(nextKey)] || [];
      const span = nextKey - prevKey;
      const ratio = span > 0 ? (targetFrame - prevKey) / span : 0;

      const nextMap = new Map<string, TrackItem>();
      nextTracks.forEach((t) => nextMap.set(String(t.track_id), t));

      const interpolated: TrackItem[] = [];
      prevTracks.forEach((pt) => {
        const tid = String(pt.track_id);
        const nt = nextMap.get(tid);
        if (nt) {
          const interpX = pt.x_pct + ratio * (nt.x_pct - pt.x_pct);
          const interpY = pt.y_pct + ratio * (nt.y_pct - pt.y_pct);
          interpolated.push({
            ...pt,
            x_pct: Math.round(interpX * 10) / 10,
            y_pct: Math.round(interpY * 10) / 10,
          });
          nextMap.delete(tid);
        } else if (!strictInternalOnly) {
          interpolated.push(pt);
        }
      });
      if (!strictInternalOnly) {
        nextMap.forEach((nt) => interpolated.push(nt));
      }
      return interpolated;
    }
  }

  if (strictInternalOnly) {
    return [];
  }

  let closest = keys[0];
  let minDiff = Math.abs(targetFrame - closest);
  for (const k of keys) {
    const diff = Math.abs(targetFrame - k);
    if (diff < minDiff) {
      minDiff = diff;
      closest = k;
    }
  }

  // Reject fallback if closest tracked frame exceeds max allowed distance
  if (minDiff > maxAllowedDiff) {
    return [];
  }

  // Allow closest tracked frame fallback so adjusted cause frames retain player tags
  return tracksRecord[closest] || tracksRecord[String(closest)] || [];
};

const PLAYER_SLUG_ALIASES: Record<string, string> = {
  mbappe: 'kylian_mbappe',
  kylian_mbappe: 'kylian_mbappe',
  rudiger: 'antonio_rudiger',
  antonio_rudiger: 'antonio_rudiger',
  courtois: 'thibaut_courtois',
  thibaut_courtois: 'thibaut_courtois',
  valverde: 'federico_valverde',
  // Real Madrid
  federico_valverde: 'federico_valverde',
  valverde: 'federico_valverde',
  bellingham: 'jude_bellingham',
  jude_bellingham: 'jude_bellingham',
  tchouameni: 'aurelien_tchouameni',
  aurelien_tchouameni: 'aurelien_tchouameni',
  vinicius: 'vinicius_junior',
  vinicius_jr: 'vinicius_junior',
  vinicius_junior: 'vinicius_junior',
  guler: 'arda_guler',
  arda_guler: 'arda_guler',
  endrick: 'endrick',
  rudiger: 'antonio_rudiger',
  antonio_rudiger: 'antonio_rudiger',
  militao: 'eder_militao',
  eder_militao: 'eder_militao',
  vazquez: 'lucas_vazquez',
  lucas_vazquez: 'lucas_vazquez',
  mendy: 'ferland_mendy',
  ferland_mendy: 'ferland_mendy',
  courtois: 'thibaut_courtois',
  thibaut_courtois: 'thibaut_courtois',
  modric: 'luka_modric',
  luka_modric: 'luka_modric',
  camavinga: 'eduardo_camavinga',
  eduardo_camavinga: 'eduardo_camavinga',
  rodrygo: 'rodrygo',
  mbappe: 'kylian_mbappe',
  kylian_mbappe: 'kylian_mbappe',
  brahim: 'brahim_diaz',
  brahim_diaz: 'brahim_diaz',
  ceballos: 'dani_ceballos',
  dani_ceballos: 'dani_ceballos',
  fran_garcia: 'fran_garcia',
  asencio: 'raul_asencio',
  raul_asencio: 'raul_asencio',

  // FC Barcelona
  yamal: 'lamine_yamal',
  lamine_yamal: 'lamine_yamal',
  lewandowski: 'robert_lewandowski',
  robert_lewandowski: 'robert_lewandowski',
  pedri: 'pedri',
  raphinha: 'raphinha',
  olmo: 'dani_olmo',
  dani_olmo: 'dani_olmo',
  casado: 'marc_casado',
  marc_casado: 'marc_casado',
  cubarsi: 'pau_cubarsi',
  pau_cubarsi: 'pau_cubarsi',
  martinez: 'inigo_martinez',
  inigo_martinez: 'inigo_martinez',
  balde: 'alejandro_balde',
  alejandro_balde: 'alejandro_balde',
  kounde: 'jules_kounde',
  jules_kounde: 'jules_kounde',
  ter_stegen: 'marc_andre_ter_stegen',
  marc_andre_ter_stegen: 'marc_andre_ter_stegen',
  pena: 'inaki_pena',
  inaki_pena: 'inaki_pena',
  gavi: 'gavi',
  de_jong: 'frenkie_de_jong',
  frenkie_de_jong: 'frenkie_de_jong',
  fermin: 'fermin_lopez',
  fermin_lopez: 'fermin_lopez',
  ferran_torres: 'ferran_torres',
  torres: 'ferran_torres',
  pau_victor: 'pau_victor',
  victor: 'pau_victor',
};

const FORBIDDEN_PLAYERS = new Set([
  'de roon', 'marten de roon',
  'mancini', 'gianluca mancini',
  'ndicka', 'evan ndicka',
  'pisilli', 'niccolo pisilli', 'niccolò pisilli',
  'malen', 'donyell malen',
  'dybala', 'paulo dybala',
  'hermoso', 'mario hermoso',
  'mora', 'rodrigo mora',
  'konate', 'konaté', 'ibrahima konate', 'ibrahima konaté',
  'dumfries', 'denzel dumfries',
  'wesley',
  'soule', 'soulé', 'matias soule', 'matías soulé',
  'cucurella', 'marc cucurella',
  'diomande', 'osmane diomande',
  'cristante', 'bryan cristante',
  'pellegrini', 'lorenzo pellegrini',
  'el shaarawy', 'stephan el shaarawy',
  'roma', 'as roma', 'atalanta'
]);

export const isForbiddenPlayer = (name?: string): boolean => {
  if (!name) return false;
  const lower = name.toLowerCase().replace(/[^a-z0-9]/g, ' ').trim();
  for (const f of FORBIDDEN_PLAYERS) {
    if (lower === f || lower.startsWith(f + ' ') || lower.endsWith(' ' + f) || lower.includes(' ' + f + ' ')) {
      return true;
    }
  }
  return false;
};

const getPlayerPortraitSlug = (name: string): string => {
  if (!name || isForbiddenPlayer(name)) return '';
  const clean = name
    .split(' (')[0]
    .replace(/#\d+\s*/g, '')
    .trim()
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');
  return PLAYER_SLUG_ALIASES[clean] || clean;
};

interface SmoothedPosition {
  x: number;
  y: number;
  vx: number;
  vy: number;
  isPresent: boolean;
}

/**
 * Ground-anchored temporal position smoothing for moving players.
 * Eliminates bounding box height jitter (e.g. YOLO clipping at knees/calves during stride)
 * by locking y to the pitch turf baseline and applying a Gaussian moving window.
 */
const getSmoothedTurfPosition = (
  tracksRecord: Record<string | number, TrackItem[]> | undefined,
  trackId: string,
  targetFrame: number,
  canvasWidth: number,
  canvasHeight: number,
  isMoving: boolean
): SmoothedPosition => {
  if (!tracksRecord) {
    return { x: 0, y: 0, vx: 0, vy: 0, isPresent: false };
  }

  // During still freeze-frames, return the exact frame coordinate anchored strictly to the turf/feet baseline
  if (!isMoving) {
    const directTracks = tracksRecord[targetFrame] || tracksRecord[String(targetFrame)] || [];
    let directTrack = directTracks.find((t) => String(t.track_id) === trackId);
    if (!directTrack) {
      // Extended proximity search during freeze if YOLO missed the exact freeze frame
      for (const off of [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -8, 8, -10, 10, -15, 15]) {
        const fNear = targetFrame + off;
        const nearList = tracksRecord[fNear] || tracksRecord[String(fNear)] || [];
        const found = nearList.find((t) => String(t.track_id) === trackId);
        if (found) {
          directTrack = found;
          break;
        }
      }
    }
    if (directTrack) {
      let px = directTrack.x_pct <= 100 ? (directTrack.x_pct / 100) * canvasWidth : directTrack.x_pct;
      let py = directTrack.y_pct <= 100 ? (directTrack.y_pct / 100) * canvasHeight : directTrack.y_pct;
      if (directTrack.bbox && Array.isArray(directTrack.bbox) && directTrack.bbox.length >= 4) {
        const [ymin, xmin, ymax, xmax] = directTrack.bbox;
        if (ymax > ymin && xmax > xmin) {
          px = ((xmin + xmax) / (2 * 1920)) * canvasWidth;
          py = (ymax / 1080) * canvasHeight;
        }
      } else if ((directTrack as any).feet_y_pct !== undefined) {
        py = ((directTrack as any).feet_y_pct / 100) * canvasHeight;
      }
      return { x: px, y: py, vx: 0, vy: 0, isPresent: true };
    }
    return { x: 0, y: 0, vx: 0, vy: 0, isPresent: false };
  }

  // For moving phases: scan a tight, responsive sliding window [targetFrame - 3, targetFrame + 3]
  // Tight window eliminates group delay during sprint runs while filtering bounding-box stride jitter.
  const W = 3;
  const sigma = 1.8;
  let maxBboxHeight = 0;
  const validDetections: { f: number; x: number; y: number; ymin?: number; ymax?: number; h?: number }[] = [];

  for (let offset = -W; offset <= W; offset++) {
    const f = targetFrame + offset;
    const trList = tracksRecord[f] || tracksRecord[String(f)] || [];
    const tr = trList.find((t) => String(t.track_id) === trackId);
    if (tr) {
      let px = tr.x_pct <= 100 ? (tr.x_pct / 100) * canvasWidth : tr.x_pct;
      let py = tr.y_pct <= 100 ? (tr.y_pct / 100) * canvasHeight : tr.y_pct;
      let ymin: number | undefined;
      let ymax: number | undefined;
      let h: number | undefined;

      if (tr.bbox && Array.isArray(tr.bbox) && tr.bbox.length >= 4) {
        ymin = tr.bbox[0];
        const xmin = tr.bbox[1];
        ymax = tr.bbox[2];
        const xmax = tr.bbox[3];
        h = ymax - ymin;
        if (h > maxBboxHeight) maxBboxHeight = h;
        if (ymax > ymin && xmax > xmin) {
          px = ((xmin + xmax) / (2 * 1920)) * canvasWidth;
          py = (ymax / 1080) * canvasHeight;
        }
      } else if ((tr as any).feet_y_pct !== undefined) {
        py = ((tr as any).feet_y_pct / 100) * canvasHeight;
      }
      validDetections.push({ f, x: px, y: py, ymin, ymax, h });
    }
  }

  // If track not found in local window, check if dropped just 1 frame ago during motion
  if (validDetections.length === 0) {
    let lastKnown: { f: number; x: number; y: number } | null = null;
    let prevKnown: { f: number; x: number; y: number } | null = null;
    for (let back = 1; back <= 3; back++) {
      const f = targetFrame - back;
      const trList = tracksRecord[f] || tracksRecord[String(f)] || [];
      const tr = trList.find((t) => String(t.track_id) === trackId);
      if (tr) {
        let px = tr.x_pct <= 100 ? (tr.x_pct / 100) * canvasWidth : tr.x_pct;
        let py = tr.y_pct <= 100 ? (tr.y_pct / 100) * canvasHeight : tr.y_pct;
        if (tr.bbox && Array.isArray(tr.bbox) && tr.bbox.length >= 4) {
          const [ymin, xmin, ymax, xmax] = tr.bbox;
          if (ymax > ymin && xmax > xmin) {
            px = ((xmin + xmax) / (2 * 1920)) * canvasWidth;
            py = (ymax / 1080) * canvasHeight;
          }
        } else if ((tr as any).feet_y_pct !== undefined) {
          py = ((tr as any).feet_y_pct / 100) * canvasHeight;
        }
        if (!lastKnown) {
          lastKnown = { f, x: px, y: py };
        } else if (!prevKnown) {
          prevKnown = { f, x: px, y: py };
          break;
        }
      }
    }
    // Extrapolate up to 3 frames during motion so brief YOLO occlusion doesn't drop the spotlight ring or tag
    if (lastKnown && (targetFrame - lastKnown.f) <= 3) {
      const elapsed = targetFrame - lastKnown.f;
      const dt = prevKnown && lastKnown.f !== prevKnown.f ? (lastKnown.f - prevKnown.f) : 1;
      const rawVx = prevKnown ? (lastKnown.x - prevKnown.x) / dt : 0;
      const rawVy = prevKnown ? (lastKnown.y - prevKnown.y) / dt : 0;
      const clampedVx = Math.max(-35, Math.min(35, rawVx));
      const clampedVy = Math.max(-25, Math.min(25, rawVy));
      const damping = Math.pow(0.85, elapsed);
      return {
        x: lastKnown.x + clampedVx * elapsed * damping,
        y: lastKnown.y + clampedVy * elapsed * damping,
        vx: clampedVx * damping,
        vy: clampedVy * damping,
        isPresent: true,
      };
    }
    return { x: 0, y: 0, vx: 0, vy: 0, isPresent: false };
  }

  // Calculate velocity from window span
  let vx = 0;
  let vy = 0;
  if (validDetections.length >= 2) {
    const first = validDetections[0];
    const last = validDetections[validDetections.length - 1];
    const span = last.f - first.f;
    if (span > 0) {
      vx = (last.x - first.x) / span;
      vy = (last.y - first.y) / span;
    }
  }

  let weightsSum = 0;
  let weightedX = 0;
  let weightedY = 0;
  let timeOffsetSum = 0;

  for (const det of validDetections) {
    const offset = det.f - targetFrame;
    const w = Math.exp(-(offset * offset) / (2 * sigma * sigma));
    weightsSum += w;
    weightedX += det.x * w;
    timeOffsetSum += offset * w;

    let yTurf = det.y;
    // Ground turf anchor: if bbox is available and height is clipped (less than 90% of max window height),
    // anchor yTurf to the turf baseline (det.ymin + maxBboxHeight)
    if (det.ymin !== undefined && det.h !== undefined && maxBboxHeight > 0) {
      if (det.h < 0.90 * maxBboxHeight) {
        const turfPx = det.ymin + maxBboxHeight;
        yTurf = (turfPx / 1080) * canvasHeight;
      }
    }
    weightedY += yTurf * w;
  }

  const smoothX = weightsSum > 0 ? weightedX / weightsSum : validDetections[0].x;
  const smoothY = weightsSum > 0 ? weightedY / weightsSum : validDetections[0].y;
  const meanTau = weightsSum > 0 ? timeOffsetSum / weightsSum : 0;

  // Phase-lead compensation: cancel the lag caused by asymmetric or causal detection windows
  const compX = smoothX - vx * meanTau;
  const compY = smoothY - vy * meanTau;

  // If instantaneous detection exists at targetFrame, blend 70% compensated smooth + 30% direct
  const directDet = validDetections.find((d) => d.f === targetFrame);
  let finalX = compX;
  let finalY = compY;
  if (directDet) {
    finalX = 0.70 * compX + 0.30 * directDet.x;
    let directYTurf = directDet.y;
    if (directDet.ymin !== undefined && directDet.h !== undefined && maxBboxHeight > 0 && directDet.h < 0.90 * maxBboxHeight) {
      directYTurf = ((directDet.ymin + maxBboxHeight) / 1080) * canvasHeight;
    }
    finalY = 0.70 * compY + 0.30 * directYTurf;
  }

  return {
    x: finalX,
    y: finalY,
    vx,
    vy,
    isPresent: true,
  };
};

interface PointCoord {
  x: number;
  y: number;
  [key: string]: any;
}

function getNormDist(
  p1: PointCoord,
  p2: PointCoord,
  width: number = 1920,
  height: number = 1080
): number {
  const x1 = p1.x <= 100 ? p1.x : (p1.x / width) * 100;
  const y1 = p1.y <= 100 ? p1.y : (p1.y / height) * 100;
  const x2 = p2.x <= 100 ? p2.x : (p2.x / width) * 100;
  const y2 = p2.y <= 100 ? p2.y : (p2.y / height) * 100;
  return Math.hypot(x1 - x2, y1 - y2);
}

/**
 * Counter-Clockwise orientation test.
 * True if points A, B, C are in counter-clockwise order.
 */
function ccw(
  A: { x: number; y: number },
  B: { x: number; y: number },
  C: { x: number; y: number }
): boolean {
  return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

/**
 * Line-segment intersection check using CCW orientation.
 * Returns true if segment (p1 -> p2) crosses segment (p3 -> p4).
 * Excludes adjacent segments sharing an endpoint.
 */
function doSegmentsIntersect(
  p1: { x: number; y: number },
  p2: { x: number; y: number },
  p3: { x: number; y: number },
  p4: { x: number; y: number },
  eps: number = 1e-5
): boolean {
  const d13 = Math.hypot(p1.x - p3.x, p1.y - p3.y);
  const d14 = Math.hypot(p1.x - p4.x, p1.y - p4.y);
  const d23 = Math.hypot(p2.x - p3.x, p2.y - p3.y);
  const d24 = Math.hypot(p2.x - p4.x, p2.y - p4.y);
  if (d13 < eps || d14 < eps || d23 < eps || d24 < eps) return false;
  return ccw(p1, p3, p4) !== ccw(p2, p3, p4) && ccw(p1, p2, p3) !== ccw(p1, p2, p4);
}

/**
 * Shortens both ends of a line segment by a radial halo margin (default 22px),
 * anchoring cleanly to player foot rings without overlapping player bodies.
 */
function applyFootHaloPadding(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  paddingPx: number = 22
): [number, number, number, number] {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dist = Math.hypot(dx, dy);
  if (dist < 2 * paddingPx + 2) {
    return [x1, y1, x2, y2];
  }
  const ux = dx / dist;
  const uy = dy / dist;
  return [
    Math.round((x1 + paddingPx * ux) * 100) / 100,
    Math.round((y1 + paddingPx * uy) * 100) / 100,
    Math.round((x2 - paddingPx * ux) * 100) / 100,
    Math.round((y2 - paddingPx * uy) * 100) / 100,
  ];
}

/**
 * Builds an SVG path string for an open chain or closed polygon with 22px foot halo margin.
 * Optionally avoids creating links that slice through an avoidPoint (such as the ball carrier).
 */
function buildPaddedChainPath<T extends { x: number; y: number }>(
  coords: T[],
  paddingPx: number = 22,
  closed: boolean = false,
  avoidPoint?: { x: number; y: number } | null,
  avoidRadiusPx: number = 32
): string {
  if (coords.length < 2) return '';
  const parts: string[] = [];
  const n = coords.length;
  const count = closed ? n : n - 1;
  for (let i = 0; i < count; i++) {
    const pA = coords[i];
    const pB = coords[(i + 1) % n];

    // If an avoidPoint is provided, skip segments that cut straight through it
    if (avoidPoint) {
      const dx = pB.x - pA.x;
      const dy = pB.y - pA.y;
      const l2 = dx * dx + dy * dy;
      if (l2 > 0) {
        const t = Math.max(0, Math.min(1, ((avoidPoint.x - pA.x) * dx + (avoidPoint.y - pA.y) * dy) / l2));
        const projX = pA.x + t * dx;
        const projY = pA.y + t * dy;
        const dist = Math.hypot(avoidPoint.x - projX, avoidPoint.y - projY);
        if (dist < avoidRadiusPx) {
          continue; // Avoid slicing through the ball carrier
        }
      }
    }

    const [sx, sy, ex, ey] = applyFootHaloPadding(pA.x, pA.y, pB.x, pB.y, paddingPx);
    parts.push(`M ${sx} ${sy} L ${ex} ${ey}`);
  }
  return parts.join(' ');
}

/**
 * Computes SVG polygon points for a perspective-aware arrowhead lying flat on the pitch turf.
 * Projects the arrowhead in pitch plane coordinates where vertical dimension is foreshortened
 * by kPitch (~0.35), ensuring arrowheads lie flat on the grass aligned with broadcast perspective.
 */
function getPerspectiveArrowheadPoints(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  headLen: number = 18,
  headWidth: number = 11,
  kPitch: number = 0.35
): { points: string; shadowPoints: string; notchX: number; notchY: number; tipX: number; tipY: number } {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len < 1e-4) {
    return {
      points: `${x2.toFixed(1)},${y2.toFixed(1)}`,
      shadowPoints: `${x2.toFixed(1)},${(y2 + 2.5).toFixed(1)}`,
      notchX: x2,
      notchY: y2,
      tipX: x2,
      tipY: y2,
    };
  }

  const ux = dx / len;
  const uy = dy / len;
  const nx = -uy;
  const ny = ux;

  // Effective length: preserve screen visibility for longitudinal vectors
  const effLen = headLen * (0.80 + 0.20 * Math.abs(ux));

  // Lateral wings lying flat on the pitch: foreshorten Y displacement by kPitch (~0.35)
  const wx = nx * headWidth;
  const wy = ny * headWidth * kPitch;

  const tipX = x2;
  const tipY = y2;

  const leftX = x2 - ux * effLen + wx;
  const leftY = y2 - uy * effLen + wy;

  const rightX = x2 - ux * effLen - wx;
  const rightY = y2 - uy * effLen - wy;

  const notchX = x2 - ux * (effLen * 0.72);
  const notchY = y2 - uy * (effLen * 0.72);

  const pts = `${tipX.toFixed(1)},${tipY.toFixed(1)} ${leftX.toFixed(1)},${leftY.toFixed(1)} ${notchX.toFixed(1)},${notchY.toFixed(1)} ${rightX.toFixed(1)},${rightY.toFixed(1)}`;
  const shadowYOff = 2.5;
  const sPts = `${tipX.toFixed(1)},${(tipY + shadowYOff).toFixed(1)} ${leftX.toFixed(1)},${(leftY + shadowYOff).toFixed(1)} ${notchX.toFixed(1)},${(notchY + shadowYOff).toFixed(1)} ${rightX.toFixed(1)},${(rightY + shadowYOff).toFixed(1)}`;

  return { points: pts, shadowPoints: sPts, notchX, notchY, tipX, tipY };
}

/**
 * Broadcast-grade easing curve for telestration lines and pass vectors.
 * Smooth broadcast S-curve easing (cosine ease-in-out):
 * Provides a gentle, deliberate initial departure, a steady, trackable glide across the pitch turf,
 * and a smooth, satisfying docking deceleration at the receiver/target.
 * Replaces aggressive instant snaps so the television audience can clearly watch long arrows deploy.
 */
function easeVector(p: number): number {
  const t = Math.min(1.0, Math.max(0.0, p));
  return 0.5 - 0.5 * Math.cos(t * Math.PI);
}

/**
 * Renders sequential rounded chevrons (<<<<<) along a run path lying flat on the turf.
 * Matches broadcast style: staggered elastic entrance, 3D pitch perspective projection,
 * tight spacing, thick broadcast strokes, turf drop shadow, and team jersey color matching.
 */
function renderRunChevrons(
  keyPrefix: string,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  progress: number,
  opacity: number,
  color: string = '#ffffff',
  canvasHeight: number = 1080
): React.ReactNode {
  if (opacity <= 0.01 || progress <= 0.01) return null;

  const dx = x2 - x1;
  const dy = y2 - y1;
  const totalDist = Math.hypot(dx, dy);
  if (totalDist < 20) return null;

  const ux = dx / totalDist;
  const uy = dy / totalDist;
  const nx = -uy;
  const ny = ux;

  // Spacing & Count: Compact direction indicator with strictly 3 to 5 chevrons
  // Prevents oversized runways from taking up too much screen space while maintaining crisp direction readability
  const numChevrons = Math.min(5, Math.max(3, Math.round(totalDist / 24.0)));
  const chevrons: React.ReactNode[] = [];

  const isWhite = color === '#ffffff' || color.toLowerCase() === '#fff' || color.startsWith('rgba(255, 255, 255');
  const spacingPx = 11.0;
  const leadStartPx = 12.0;

  for (let i = 0; i < numChevrons; i++) {
    // Start delay staggered smoothly across the animation window
    const startT = (i / numChevrons) * 0.45;
    const durT = 0.35;
    const localT = Math.max(0, Math.min(1, (progress - startT) / durT));
    if (localT <= 0.01) continue;

    // Elastic pop scale: 0 -> ~1.16 -> 1.0
    const elasticScale = Math.sin((localT * Math.PI) / 2) * (1.0 + 0.16 * Math.sin(localT * Math.PI));
    const chevOpacity = Math.min(1, localT * 2.8) * opacity;

    // Compact direction indicator placement: clustered tightly near origin pointing along unit vector
    const distAlongVector = leadStartPx + i * spacingPx;
    const cx = x1 + ux * distAlongVector;
    const cy = y1 + uy * distAlongVector;

    // Dynamic pitch perspective foreshortening factor based on Y depth on pitch (~0.32 to ~0.40)
    const kPitch = Math.max(0.30, Math.min(0.42, 0.33 + 0.09 * (cy / canvasHeight)));
    const depthScale = 0.90 + 0.18 * (cy / canvasHeight);

    // 3D Stair effect: first chevron is at the top, stepping down to turf at the end like stairs
    const stairFraction = 1.0 - (i / Math.max(1, numChevrons - 1));
    const stairLiftY = -2.8 * stairFraction * depthScale;
    const shadowYOff = 2.0 + 2.8 * stairFraction;

    // Lateral wing span and longitudinal length matching broadcast perspective (same math as getPerspectiveArrowheadPoints)
    const effLen = 8.2 * depthScale * elasticScale * (0.82 + 0.22 * Math.abs(ux));
    const wingWidth = 10.5 * depthScale * elasticScale;

    // Lateral wings lying flat on pitch: foreshorten Y displacement by kPitch (~0.35)
    const wx = nx * wingWidth;
    const wy = ny * wingWidth * kPitch;

    // Longitudinal travel vector with depth-aware vertical compression
    const lx = ux * effLen;
    const ly = uy * effLen * (0.58 + 0.42 * kPitch);

    // Tip points forward along travel vector (with stair elevation)
    const tipX = cx + 0.35 * lx;
    const tipY = cy + 0.35 * ly + stairLiftY;

    // Left and right wings flare back and out on the turf plane
    const leftX = cx - 0.65 * lx + wx;
    const leftY = cy - 0.65 * ly + wy + stairLiftY;

    const rightX = cx - 0.65 * lx - wx;
    const rightY = cy - 0.65 * ly - wy + stairLiftY;

    // SVG paths for 3D Turf Drop Shadow (projected flat on turf plane below the elevated chevron)
    const groundLeftY = leftY - stairLiftY;
    const groundTipY = tipY - stairLiftY;
    const groundRightY = rightY - stairLiftY;
    const shadowPath = `M ${leftX.toFixed(1)} ${(groundLeftY + shadowYOff).toFixed(1)} L ${tipX.toFixed(1)} ${(groundTipY + shadowYOff).toFixed(1)} L ${rightX.toFixed(1)} ${(groundRightY + shadowYOff).toFixed(1)}`;
    const mainPath = `M ${leftX.toFixed(1)} ${leftY.toFixed(1)} L ${tipX.toFixed(1)} ${tipY.toFixed(1)} L ${rightX.toFixed(1)} ${rightY.toFixed(1)}`;

    const mainStrokeWidth = (3.8 * depthScale).toFixed(1);
    const shadowStrokeWidth = (5.4 * depthScale).toFixed(1);
    const coreStrokeWidth = (1.3 * depthScale).toFixed(1);

    chevrons.push(
      <g key={`${keyPrefix}-chev-${i}`} style={{ opacity: chevOpacity }}>
        {/* 1. 3D Turf Drop Shadow - flat on grass plane */}
        <path
          d={shadowPath}
          fill="none"
          stroke="rgba(0, 0, 0, 0.52)"
          strokeWidth={shadowStrokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* 2. Bold Team Jersey Chevron */}
        <path
          d={mainPath}
          fill="none"
          stroke={color}
          strokeWidth={mainStrokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* 3. Crisp Luminous Core Highlight */}
        <path
          d={mainPath}
          fill="none"
          stroke={isWhite ? 'rgba(255, 255, 255, 0.95)' : 'rgba(255, 255, 255, 0.82)'}
          strokeWidth={coreStrokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
    );
  }

  // Reversed layer stacking order: first chevron renders at the TOP, last chevron at the BOTTOM like stairs
  return (
    <g key={keyPrefix} style={{ pointerEvents: 'none' }}>
      {chevrons.reverse()}
    </g>
  );
}

/**
 * Detects collinearity and spatial overlap between a secondary vector and higher-priority vectors
 * (e.g., between a pressing vector and a ground pass line).
 * Returns the lateral screen offset (dx, dy) to displace the secondary vector so they never clash.
 */
function getVectorCollinearOffset(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  primaryVectors: { x1: number; y1: number; x2: number; y2: number }[],
  kPitch: number = 0.35,
  minDistPx: number = 24
): { offsetX: number; offsetY: number } {
  let totalOffX = 0;
  let totalOffY = 0;

  for (const pv of primaryVectors) {
    const pdx = pv.x2 - pv.x1;
    const pdy = pv.y2 - pv.y1;
    const pLen = Math.hypot(pdx, pdy);
    if (pLen < 1e-3) continue;

    const vdx = x2 - x1;
    const vdy = y2 - y1;
    const vLen = Math.hypot(vdx, vdy);
    if (vLen < 1e-3) continue;

    // Unit directions
    const pux = pdx / pLen;
    const puy = pdy / pLen;
    const vux = vdx / vLen;
    const vuy = vdy / vLen;

    // Directional alignment (|cos theta|)
    const align = Math.abs(pux * vux + puy * vuy);
    if (align < 0.82) continue; // Not collinear

    // Perpendicular unit normal
    const pnx = -puy;
    const pny = pux;

    // Perpendicular distances of endpoints to primary line
    const d1 = (x1 - pv.x1) * pnx + (y1 - pv.y1) * pny;
    const d2 = (x2 - pv.x1) * pnx + (y2 - pv.y1) * pny;
    const perpDist = (Math.abs(d1) + Math.abs(d2)) / 2;

    if (perpDist >= minDistPx) continue; // Far enough apart

    // Check longitudinal overlap along primary vector
    const proj1 = (x1 - pv.x1) * pux + (y1 - pv.y1) * puy;
    const proj2 = (x2 - pv.x1) * pux + (y2 - pv.y1) * puy;
    const minProj = Math.min(proj1, proj2);
    const maxProj = Math.max(proj1, proj2);

    if (maxProj < -10 || minProj > pLen + 10) continue; // No longitudinal overlap

    // Shift away from primary line
    const sign = (d1 + d2) >= 0 ? 1 : -1;
    const shiftMag = Math.max(minDistPx, minDistPx - perpDist + 6);

    const offX = pnx * shiftMag * sign;
    const offY = pny * shiftMag * sign * (0.65 + 0.35 * kPitch);

    totalOffX += offX;
    totalOffY += offY;
  }

  return { offsetX: totalOffX, offsetY: totalOffY };
}

/**
 * Computes the Shortest Open Path (Minimum Spanning / Nearest-Neighbor Path)
 * connecting same-team players in a tactical unit.
 *
 * Rules:
 * 1. Never closes into a polygon (open polyline).
 * 2. Starts at one end of the unit (min y or min x based on primary spread)
 *    and iteratively connects to the nearest unvisited teammate.
 * 3. Rejects any link whose length exceeds maxLinkDistPct (28% of screen width),
 *    preventing far-away players (e.g. wingers at touchline) from being dragged into the chain.
 * 4. Eliminates crisscrosses via 2-opt edge uncrossing.
 * 5. Consistently orients the open chain from top-to-bottom or left-to-right.
 */
function computeShortestOpenChain<T extends PointCoord>(
  items: T[],
  maxLinkDistPct: number = 28.0,
  width: number = 1920,
  height: number = 1080
): T[] {
  if (items.length <= 1) return [...items];
  if (items.length === 2) {
    const d = getNormDist(items[0], items[1], width, height);
    if (d > maxLinkDistPct) {
      return [items[0]];
    }
    return items[0].y <= items[1].y ? [items[0], items[1]] : [items[1], items[0]];
  }

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  items.forEach((p) => {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  });

  const spanX = (maxX - minX) / width;
  const spanY = (maxY - minY) / height;
  const preferY = spanY >= spanX;

  let bestChain: T[] = [];
  let bestDist = Infinity;

  for (let s = 0; s < items.length; s++) {
    const startNode = items[s];
    const visited: T[] = [startNode];
    const unvisited = items.filter((_, idx) => idx !== s);
    let curr = startNode;
    let chainDist = 0;

    while (unvisited.length > 0) {
      let nearestIdx = -1;
      let minDist = Infinity;
      for (let j = 0; j < unvisited.length; j++) {
        const d = getNormDist(curr, unvisited[j], width, height);
        if (d < minDist) {
          minDist = d;
          nearestIdx = j;
        }
      }

      if (nearestIdx >= 0 && minDist <= maxLinkDistPct) {
        chainDist += minDist;
        const nextNode = unvisited.splice(nearestIdx, 1)[0];
        visited.push(nextNode);
        curr = nextNode;
      } else {
        break;
      }
    }

    if (visited.length > bestChain.length) {
      bestChain = visited;
      bestDist = chainDist;
    } else if (visited.length === bestChain.length && chainDist < bestDist) {
      bestChain = visited;
      bestDist = chainDist;
    }
  }

  // 2-opt uncrossing optimization to remove any self-intersections
  let improved = true;
  let iter = 0;
  while (improved && iter < 30) {
    improved = false;
    iter++;
    const m = bestChain.length;
    for (let i = 0; i < m - 2; i++) {
      const p_i1 = bestChain[i];
      const p_i2 = bestChain[i + 1];
      for (let j = i + 2; j < m - 1; j++) {
        const p_j1 = bestChain[j];
        const p_j2 = bestChain[j + 1];
        if (doSegmentsIntersect(p_i1, p_i2, p_j1, p_j2)) {
          const sub = bestChain.slice(i + 1, j + 1).reverse();
          bestChain.splice(i + 1, sub.length, ...sub);
          improved = true;
          break;
        }
      }
      if (improved) break;
    }
  }

  if (bestChain.length >= 2) {
    const first = bestChain[0];
    const last = bestChain[bestChain.length - 1];
    if (preferY) {
      if (first.y > last.y) {
        bestChain.reverse();
      }
    } else {
      if (first.x > last.x) {
        bestChain.reverse();
      }
    }
  }

  return bestChain;
}


/**
 * Strict Pressing-Trap Rule:
 * ONLY draw a closed 3-player dashed triangle (closed_polygon = True) when:
 * 1. Exactly 3 pressers of Team B.
 * 2. All 3 edges are < 24% screen distance (maxEdgePct).
 * 3. The opposing Ball Carrier lies inside or within 6% (maxCarrierDistPct) of the triangle.
 */
function isValidPressingTrap(
  pressers: { x: number; y: number }[],
  carrier: { x: number; y: number } | null | undefined,
  maxEdgePct: number = 24.0,
  maxCarrierDistPct: number = 6.0,
  width: number = 1920,
  height: number = 1080
): boolean {
  if (pressers.length !== 3 || !carrier) return false;

  const [p1, p2, p3] = pressers;
  const d12 = getNormDist(p1, p2, width, height);
  const d23 = getNormDist(p2, p3, width, height);
  const d31 = getNormDist(p3, p1, width, height);

  if (d12 >= maxEdgePct || d23 >= maxEdgePct || d31 >= maxEdgePct) {
    return false;
  }

  const norm = (pt: { x: number; y: number }) => ({
    x: pt.x <= 100 ? pt.x : (pt.x / width) * 100,
    y: pt.y <= 100 ? pt.y : (pt.y / height) * 100,
  });
  const np1 = norm(p1);
  const np2 = norm(p2);
  const np3 = norm(p3);
  const nc = norm(carrier);

  const cross1 = (nc.x - np2.x) * (np1.y - np2.y) - (np1.x - np2.x) * (nc.y - np2.y);
  const cross2 = (nc.x - np3.x) * (np2.y - np3.y) - (np2.x - np3.x) * (nc.y - np3.y);
  const cross3 = (nc.x - np1.x) * (np3.y - np1.y) - (np3.x - np1.x) * (nc.y - np1.y);

  const hasNeg = cross1 < 0 || cross2 < 0 || cross3 < 0;
  const hasPos = cross1 > 0 || cross2 > 0 || cross3 > 0;
  const isInside = !(hasNeg && hasPos);

  if (isInside) return true;

  const distToSegment = (
    px: number, py: number,
    ax: number, ay: number,
    bx: number, by: number
  ) => {
    const dx = bx - ax;
    const dy = by - ay;
    const l2 = dx * dx + dy * dy;
    if (l2 === 0) return Math.hypot(px - ax, py - ay);
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2));
    const qx = ax + t * dx;
    const qy = ay + t * dy;
    return Math.hypot(px - qx, py - qy);
  };

  const distToEdges = Math.min(
    distToSegment(nc.x, nc.y, np1.x, np1.y, np2.x, np2.y),
    distToSegment(nc.x, nc.y, np2.x, np2.y, np3.x, np3.y),
    distToSegment(nc.x, nc.y, np3.x, np3.y, np1.x, np1.y)
  );

  return distToEdges <= maxCarrierDistPct;
}

export const TacticalTelestrationOverlay: React.FC<TacticalTelestrationProps> = ({
  players: initialPlayers = [],
  lines: initialLines = [],
  polygons: initialPolygons = [],
  title,
  subtitle,
  delay = 0,
  renderTurfOnly = false,
  renderBadgesOnly = false,
  cause_frame_idx,
  causeFrameIdx,
  freeze_frames,
  freezeFrames,
  effect_frame_idx,
  effectFrameIdx,
  actual_video_frames,
  actualVideoFrames: propActualVideoFrames,
  post_roll_frames,
  postRollFrames: propPostRollFrames,
  tracks_per_frame,
  tracksPerFrame,
  cause_telestration,
  causeTelestration,
  ball_carrier_per_frame,
  ballCarrierPerFrame: propBallCarrierPerFrame,
  team_colors,
  teamColors,
  beats,
  videoUrl,
  video_end_frame,
  videoEndFrame,
  actual_clip_video_frames,
  actualClipVideoFrames,
  total_video_frames,
  totalVideoFrames: propTotalVideoFrames,
  freezeFrameUrl,
  effectFrameUrl,
  effect_frame_url,
  fgMaskUrl,
  fg_mask_url,
  zoomIntensity = 1.06,
  is_actual_goal_scored_in_clip,
  isActualGoalScoredInClip,
  goal_impact_frame_idx,
  goalImpactFrameIdx,
  video_freeze_frame,
  videoFreezeFrame: propVideoFreezeFrame,
  pre_roll_duration_frames,
  preRollDurationFrames: propPreRollDurationFrames,
  catalyst_player,
  structural_network,
  exploit_zone,
  vectors,
  actual_decision_note,
  visual_cue_tags,
  visualCueTags: propVisualCueTags,
  speed_segments,
  speedSegments,
  is_warped,
  isWarped,
  warped_video_url,
  warpedVideoUrl,
  is_tactical,
  isTactical: propIsTactical,
  is_close_up,
  isCloseUp: propIsCloseUp,
  shot_type,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  const hasTacticalData = Boolean(
    (cause_telestration?.players && cause_telestration.players.length > 0) ||
    (causeTelestration as any)?.players?.length > 0 ||
    cause_telestration?.catalyst_player ||
    (causeTelestration as any)?.catalyst_player ||
    (cause_telestration?.vectors && cause_telestration.vectors.length > 0) ||
    (causeTelestration as any)?.vectors?.length > 0 ||
    (tracks_per_frame && Object.keys(tracks_per_frame).length > 0) ||
    (tracksPerFrame && Object.keys(tracksPerFrame).length > 0)
  );

  const explicitCloseUp = Boolean(
    is_close_up ??
    propIsCloseUp ??
    (shot_type === 'close_up' ? true : undefined) ??
    (cause_telestration as any)?.is_close_up ??
    ((cause_telestration as any)?.shot_type === 'close_up' ? true : undefined) ??
    (causeTelestration as any)?.is_close_up ??
    ((causeTelestration as any)?.shot_type === 'close_up' ? true : undefined) ??
    (is_tactical === false ? true : undefined) ??
    (propIsTactical === false ? true : undefined) ??
    (cause_telestration?.is_tactical === false ? true : undefined) ??
    (cause_telestration?.isTactical === false ? true : undefined) ??
    ((causeTelestration as any)?.is_tactical === false ? true : undefined) ??
    ((causeTelestration as any)?.isTactical === false ? true : undefined) ??
    false
  );

  const runtimeGeometryCloseUp = (() => {
    const tMap = tracks_per_frame ?? tracksPerFrame;
    if (!tMap || Object.keys(tMap).length === 0) return false;
    const kVal =
      cause_frame_idx ??
      causeFrameIdx ??
      video_freeze_frame ??
      propVideoFreezeFrame ??
      pre_roll_duration_frames ??
      propPreRollDurationFrames ??
      60;
    const kList =
      tMap[String(kVal)] ||
      (tMap as any)[kVal] ||
      (() => {
        const numKeys = Object.keys(tMap).map(Number).filter((n) => !isNaN(n));
        if (numKeys.length === 0) return undefined;
        numKeys.sort((a, b) => Math.abs(a - kVal) - Math.abs(b - kVal));
        return Math.abs(numKeys[0] - kVal) <= 15 ? tMap[String(numKeys[0])] : undefined;
      })();
    if (!Array.isArray(kList)) return false;
    if (kList.length < 3) return true;
    for (const tr of kList) {
      if (Array.isArray(tr.bbox) && tr.bbox.length >= 4) {
        const [ymin, xmin, ymax, xmax] = tr.bbox;
        const bhRatio = (ymax - ymin) / 1080.0;
        const bwRatio = (xmax - xmin) / 1920.0;
        if (bhRatio > 0.24 || bwRatio > 0.20) {
          return true;
        }
      }
    }
    return false;
  })();

  const isCloseUp = explicitCloseUp || runtimeGeometryCloseUp;
  const effectiveIsTactical = !isCloseUp && hasTacticalData;

  const effectiveIsWarped = Boolean(
    is_warped ??
    isWarped ??
    warped_video_url ??
    warpedVideoUrl ??
    (videoUrl && (videoUrl.includes('warped_') || videoUrl.includes('_warped')))
  );
  const effectiveWarpedVideoUrl = warped_video_url ?? warpedVideoUrl ?? (effectiveIsWarped ? videoUrl : undefined);
  const effectiveSpeedSegments = speed_segments ?? speedSegments ?? (cause_telestration as any)?.speed_segments ?? [];

  const rawCauseFrame = cause_frame_idx ?? causeFrameIdx;
  const isChoreographed =
    typeof rawCauseFrame === 'number' ||
    typeof video_freeze_frame === 'number' ||
    typeof propVideoFreezeFrame === 'number';

  const b1Ms =
    beats?.beat_1_setup?.duration_ms ??
    (beats?.beat_1_setup?.duration_sec !== undefined ? beats.beat_1_setup.duration_sec * 1000 : undefined) ??
    (beats?.beat_1_setup?.duration !== undefined ? beats.beat_1_setup.duration * 1000 : undefined);
  const b1Frames = b1Ms ? Math.ceil(((b1Ms / 1000.0) + 0.15) * fps) : undefined;

  // Physical video freeze frame K_video decoupled from audio setup
  const K_video =
    cause_frame_idx ??
    causeFrameIdx ??
    video_freeze_frame ??
    propVideoFreezeFrame ??
    pre_roll_duration_frames ??
    propPreRollDurationFrames ??
    causeTelestration?.video_freeze_frame ??
    causeTelestration?.videoFreezeFrame ??
    60;

  const b1_frames = b1Ms ? Math.ceil((b1Ms / 1000.0) * fps) : 60;

  const b2Ms =
    beats?.beat_2_cause?.duration_ms ??
    (beats?.beat_2_cause?.duration_sec !== undefined ? beats.beat_2_cause.duration_sec * 1000 : undefined) ??
    (beats?.beat_2_cause?.duration !== undefined ? beats.beat_2_cause.duration * 1000 : undefined);
  const causeTel = cause_telestration ?? causeTelestration;
  const b2_frames = b2Ms ? Math.ceil((b2Ms / 1000.0) * fps) : 90;

  const b3Ms =
    beats?.beat_3_effect?.duration_ms ??
    (beats?.beat_3_effect?.duration_sec !== undefined ? beats.beat_3_effect.duration_sec * 1000 : undefined) ??
    (beats?.beat_3_effect?.duration !== undefined ? beats.beat_3_effect.duration * 1000 : undefined);
  const b3DurationSec = b3Ms !== undefined ? b3Ms / 1000.0 : 0;
  const b3_frames = b3Ms ? Math.ceil(b3DurationSec * fps) : 90;
  const beat3MinFrames = b3Ms ? Math.ceil(b3DurationSec * fps) + 6 : 90;

  const tracksMap = tracks_per_frame ?? tracksPerFrame;
  const visualCueTags: VisualCueTag[] =
    visual_cue_tags ||
    propVisualCueTags ||
    (beats as any)?.visual_cue_tags ||
    (causeTel as any)?.visual_cue_tags ||
    [];

  const statsTags = visualCueTags.filter(
    (t) =>
      t.type === 'stat_card' ||
      t.type === 'sofascore' ||
      t.type === 'player_stats' ||
      t.type === 'heatmap' ||
      t.type === 'average_position' ||
      t.type === 'heatmap_comparison' ||
      t.type === 'comparison_heatmap' ||
      t.type === 'data_card' ||
      t.type === 'match_heatmap'
  );

  const maxStatsEnd = statsTags.reduce((maxEnd: number, t: any) => {
    const s = Math.max(t.start_frame ?? K_video, K_video);
    const d = Math.max(t.duration_frames ?? 210, 210);
    const e = Math.max(t.end_frame ?? 0, s + d);
    return Math.max(maxEnd, e);
  }, 0);

  const b2Start = Math.max(K_video, b1_frames) + 6;
  const b2End = b2Start + b2_frames;
  const b3Start = b2End + 6;
  const b3End = b3Start + b3_frames;
  const explicitFreezeFrames = freeze_frames ?? freezeFrames ?? (causeTel as any)?.freeze_frames;
  const baseFreezeDuration =
    effectiveIsWarped && explicitFreezeFrames !== undefined && explicitFreezeFrames !== null
      ? Number(explicitFreezeFrames)
      : (b3End + 12 - K_video);

  // Never freeze the screen on close-ups, bench/substitution shots, or non-tactical frames
  const freezeDuration = effectiveIsTactical ? Math.max(0, baseFreezeDuration) : 0;
  const freeze_end_frame = K_video + freezeDuration;

  const K = K_video;
  const V_freeze = K_video;
  const T_preroll = K_video;
  const causeFrame = K_video;
  const freezeEnd = K_video + freezeDuration;
  const playbackRatePreRoll = 1.0;

  const isPlayerDataActive = statsTags.some((t: any) => {
    const s = Math.max(t.start_frame ?? K_video, K_video);
    const d = Math.max(t.duration_frames ?? 210, 210);
    const e = Math.max(t.end_frame ?? 0, s + d);
    return frame >= s && frame <= e;
  });

  const physicalVideoFrames =
    actual_clip_video_frames ??
    actualClipVideoFrames ??
    actual_video_frames ??
    propActualVideoFrames ??
    total_video_frames ??
    propTotalVideoFrames ??
    video_end_frame ??
    videoEndFrame ??
    (() => {
      if (tracksMap) {
        const keys = Object.keys(tracksMap).map(Number).filter((n) => !isNaN(n));
        if (keys.length > 0) {
          const maxK = Math.max(...keys);
          if (maxK > 0) return maxK;
        }
      }
      return undefined;
    })() ??
    332;

  const resolvedActualVideoFrames = Math.max(V_freeze + 4, physicalVideoFrames);

  // Normal video playback resumes strictly AFTER TTS audio finishes cleanly:
  const maxAvailablePostRoll = Math.max(1, resolvedActualVideoFrames - V_freeze - 2);
  const rawPostRoll = propPostRollFrames ?? post_roll_frames ?? 60;
  const postRollFrames = Math.min(maxAvailablePostRoll, Math.max(30, Math.min(60, rawPostRoll)));
  const phase3Duration = postRollFrames;
  const resolvedTotalVideoFrames = V_freeze + postRollFrames;

  const effectiveFgMaskUrl = fgMaskUrl || fg_mask_url;

  // Real Goal Evaluation (strictly no false goal flashes or sounds)
  const isActualGoal = Boolean(
    is_actual_goal_scored_in_clip ??
    isActualGoalScoredInClip ??
    causeTel?.is_actual_goal_scored_in_clip ??
    false
  );

  const rawGoalImpact =
    goal_impact_frame_idx ??
    goalImpactFrameIdx ??
    causeTel?.goal_impact_frame_idx ??
    effect_frame_idx ??
    effectFrameIdx;

  const realGoalImpactFrame =
    isActualGoal && typeof rawGoalImpact === 'number'
      ? freezeEnd + Math.max(0, rawGoalImpact - V_freeze)
      : undefined;

  const colors = team_colors ?? teamColors ?? {};
  const homeKit = causeTel?.team_kits?.home;
  const awayKit = causeTel?.team_kits?.away;
  const homePrimary = homeKit?.primary || colors.home_hex || '#8a1529';
  const homeSecondary = homeKit?.secondary || colors.home_secondary || '#f1b434';
  const awayPrimary = awayKit?.primary || colors.away_hex || '#ffffff';
  const awaySecondary = awayKit?.secondary || colors.away_secondary || '#e11d48';

  const isAttackingHome = causeTel?.attacking_team === 'home' || causeTel?.catalyst_player?.team === 'home';
  const attTeam = isAttackingHome ? 'home' : 'away';
  const defTeam = isAttackingHome ? 'away' : 'home';
  const homeColor = homePrimary;
  const awayColor = awayPrimary;
  const attJerseyColor = isAttackingHome ? homeColor : awayColor;
  const defJerseyColor = isAttackingHome ? awayColor : homeColor;

  // Television broadcast kit-adaptive pass & vector color resolver:
  const getTeamKitPassColor = (team?: string): string => {
    const isHome = team === 'home' || (!team && isAttackingHome);
    if (isHome) {
      // Home team (e.g. AS Roma maroon #8a1529 / gold #f1b434):
      // Gold #f1b434 provides high-visibility television clarity against grass turf while strictly matching the official jersey kit!
      return homeSecondary || '#f1b434';
    }
    // Away team (e.g. Real Madrid white #ffffff):
    return awayPrimary || '#ffffff';
  };

  const clipIdStr = String(causeTel?.clip_id || title || videoUrl || '').toLowerCase();
  const isSetPieceOrLongBall = Boolean(
    causeTel?.is_set_piece_or_long_ball ||
    causeTel?.is_set_piece ||
    causeTel?.sequence_type === 'SET_PIECE' ||
    causeTel?.sequence_type === 'CORNER_KICK' ||
    causeTel?.sequence_type === 'GOALKEEPER_DISTRIBUTION' ||
    causeTel?.sequence_type === 'GOAL_KICK' ||
    clipIdStr.includes('clip_03') ||
    clipIdStr.includes('clip_05') ||
    clipIdStr.includes('corner') ||
    clipIdStr.includes('defensive_third')
  );

  // 1. Determine active phase
  const isPhase1 = isChoreographed && frame < T_preroll;
  const isPhase2 = isChoreographed && frame >= T_preroll && frame < freezeEnd;
  const isPhase3 = isChoreographed && frame >= freezeEnd;
  const showBadges = !isChoreographed || isPhase2;

  // 4-Step Progressive Reveal Choreography: starts at graphicsStart = max(K_video, b1_frames) when Beat 2 Audio starts
  const graphicsStart = Math.max(K_video, b1_frames);
  const graphicsDuration = Math.max(1, freezeEnd - graphicsStart);
  const isGraphicsActive = isPhase2 && frame >= graphicsStart;
  const graphicsProgress = isGraphicsActive ? frame - graphicsStart : 0;
  const graphicsRatio = isGraphicsActive && graphicsDuration > 0 ? graphicsProgress / graphicsDuration : 0;
  const freezeProgress = isPhase2 ? frame - T_preroll : 0;
  const freezeRatio = isPhase2 && freezeDuration > 0 ? freezeProgress / freezeDuration : 0;

  // Final 12 frames of freeze fade-out: must stay strictly 1.0 throughout pre-roll and freeze until the final 12 frames
  const freezeExitFade = frame >= freezeEnd - 12 && freezeDuration > 12
    ? interpolate(
        frame,
        [freezeEnd - 12, freezeEnd],
        [1.0, 0.0],
        { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
      )
    : 1.0;

  // 3 High-Contrast Primary Pills during Freeze-Frame (Zero Overlap):
  // 1. Catalyst Role Badge ("BALL CARRIER" / "GOALKEEPER // BALL CARRIER" / "CORNER TAKER")
  // 2. Runner Role Badge ("THIRD-MAN RUNNER")
  // 3. Target Zone Header Badge ("EXPLOITABLE HALF-SPACE" / "BOX DELIVERY TARGET")
  const isFirstHalf = isGraphicsActive && graphicsRatio < 0.50;
  const isSecondHalf = isGraphicsActive && graphicsRatio >= 0.50;

  const catalystBadgeOpacity = isGraphicsActive
    ? interpolate(graphicsRatio, [0, 0.06], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }) * freezeExitFade
    : 0;

  // Step 2: Structure / Lines (revealing smoothly, fading out over final 12 frames)
  const structThreshold = 0.15;
  const showStructure = isGraphicsActive && graphicsRatio >= structThreshold && freezeExitFade > 0;
  const structProgress = isGraphicsActive && graphicsRatio >= structThreshold
    ? Math.min(1, Math.max(0, (graphicsRatio - structThreshold) / 0.28))
    : 0;
  const structOpacity = isGraphicsActive ? structProgress * freezeExitFade : 0;
  const showUnitLabel = false;
  const unitLabelOpacity = 0;

  // Second half of freeze: ONLY Exploit Zone Header Badge & Runner Role Badge (fading out over final 12 frames)
  const exploitBadgeOpacity = isSecondHalf
    ? interpolate(graphicsRatio, [0.50, 0.56], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }) * freezeExitFade
    : 0;

  const runnerBadgeOpacity = isSecondHalf
    ? interpolate(graphicsRatio, [0.50, 0.56], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }) * freezeExitFade
    : 0;

  const maxTracksFrame = (() => {
    if (tracksMap) {
      const keys = Object.keys(tracksMap).map(Number).filter((n) => !isNaN(n));
      if (keys.length > 0) return Math.max(...keys);
    }
    return undefined;
  })();

  const effectivePostRoll = effectiveIsWarped
    ? Math.max(1, (maxTracksFrame ?? physicalVideoFrames) - freezeEnd)
    : postRollFrames;

  const phase3Progress = isPhase3 ? frame - freezeEnd : 0;
  // Fade out spotlight ring and post-freeze markers promptly within 12 frames of unfreezing into live motion
  const seq3RingFade = isPhase3
    ? interpolate(phase3Progress, [0, 12], [1.0, 0.0], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      })
    : 0.0;

  // Static Polygons & Arrows: strictly Sequence 2 only, fading to 0 over final 12 frames, 100% hidden in Sequence 3
  const showExploitZone = isGraphicsActive && graphicsRatio >= 0.50 && freezeExitFade > 0;
  const exploitProgress = isGraphicsActive && graphicsRatio >= 0.50
    ? Math.min(1, Math.max(0, (graphicsRatio - 0.50) / 0.10))
    : 0;
  const exploitOpacity = isGraphicsActive ? exploitProgress * freezeExitFade : 0;

  const vectorThreshold = isSetPieceOrLongBall ? 0.15 : 0.25;

  // 2. Camera Zoom & Climax Punch-In
  let zoomScale = 1.0;
  if (isPhase2) {
    zoomScale = interpolate(
      freezeProgress,
      [0, Math.min(30, freezeDuration)],
      [1.0, zoomIntensity],
      { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
    );
  } else if (isPhase3) {
    const postProgress = frame - freezeEnd;
    zoomScale = interpolate(
      postProgress,
      [0, 15],
      [zoomIntensity, 1.0],
      { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
    );
  }

  // Climax 6-frame micro camera punch-in (scale: 1.0 -> 1.035 -> 1.0) - ONLY on actual goal
  const isClimaxPunch =
    isChoreographed &&
    isActualGoal &&
    realGoalImpactFrame !== undefined &&
    frame >= realGoalImpactFrame &&
    frame < realGoalImpactFrame + 6;

  const punchScale = isClimaxPunch
    ? interpolate(frame - realGoalImpactFrame, [0, 2, 5], [1.0, 1.035, 1.0], { extrapolateRight: 'clamp' })
    : 1.0;
  const totalZoomScale = zoomScale * punchScale;

  // 4-frame subtle screen flash on impact (rgba(255,255,255,0.28) decaying to 0) - ONLY on actual goal
  const isClimaxFlash =
    isChoreographed &&
    isActualGoal &&
    realGoalImpactFrame !== undefined &&
    frame >= realGoalImpactFrame &&
    frame < realGoalImpactFrame + 4;

  const flashOpacity = isClimaxFlash
    ? interpolate(frame - realGoalImpactFrame, [0, 3], [0.28, 0], { extrapolateRight: 'clamp' })
    : 0;

  // 18-frame emerald green goal net interior pulse - ONLY on actual goal
  const isNetPulse =
    isChoreographed &&
    isActualGoal &&
    realGoalImpactFrame !== undefined &&
    frame >= realGoalImpactFrame &&
    frame < realGoalImpactFrame + 18;

  const netPulseOpacity = isNetPulse
    ? interpolate(frame - realGoalImpactFrame, [0, 5, 17], [0, 0.55, 0], { extrapolateRight: 'clamp' })
    : 0;

  // 3. Static Polygons Fade-Out: 100% hidden in Sequence 1 and Sequence 3
  let staticPolygonsOpacity = 1.0;
  if (!isChoreographed) {
    staticPolygonsOpacity = 1.0;
  } else if (isGraphicsActive) {
    const entrySpring = spring({
      frame: graphicsProgress,
      fps,
      config: { damping: 14, stiffness: 120, mass: 0.8 },
    });
    staticPolygonsOpacity = interpolate(entrySpring, [0, 1], [0, 1], {
      extrapolateRight: 'clamp',
    }) * freezeExitFade;
  } else {
    // Sequence 1 & Sequence 3: 100% hidden
    staticPolygonsOpacity = 0.0;
  }

  // 4. Resolve V3 Tactical Analyst Entities (Catalyst, Structural Network, Exploit Zone, Vectors)
  const catalyst: TacticalCatalystPlayer | undefined = (() => {
    if (catalyst_player) return catalyst_player;
    if (causeTel?.catalyst_player) return causeTel.catalyst_player;
    if (causeTel?.ball_handler_track_id) {
      return {
        track_id: causeTel.ball_handler_track_id,
        reason: 'Ball carrier',
      };
    }
    if (causeTel?.focal_track_ids && causeTel.focal_track_ids.length > 0) {
      return {
        track_id: causeTel.focal_track_ids[0],
        reason: 'Focal player',
      };
    }
    const cPlayers = cause_telestration?.players || (causeTelestration as any)?.players;
    if (Array.isArray(cPlayers) && cPlayers.length > 0) {
      const p0 = cPlayers[0];
      return {
        track_id: p0.id ?? p0.track_id,
        name: p0.name,
        name_pos: p0.name_pos || p0.name,
        x_pct: p0.x_pct ?? p0.x,
        y_pct: p0.y_pct ?? p0.y,
        x: p0.x_pct ?? p0.x,
        y: p0.y_pct ?? p0.y,
        team: p0.team,
        reason: 'Initiating tactical sequence on the ball',
        role: p0.role,
      };
    }
    return undefined;
  })();
  const catalystId = catalyst ? String(catalyst.track_id) : undefined;

  const structuralNetwork: TacticalStructuralNetwork | undefined = (() => {
    if (structural_network && structural_network.track_ids && structural_network.track_ids.length >= 2) {
      return structural_network;
    }
    if (causeTel?.structural_network && causeTel.structural_network.track_ids && causeTel.structural_network.track_ids.length >= 2) {
      return causeTel.structural_network;
    }
    if (causeTel?.backline_track_ids && causeTel.backline_track_ids.length >= 2) {
      return {
        type: 'DEFENSIVE_LINE',
        track_ids: causeTel.backline_track_ids,
        closed_polygon: false,
        unit_label: 'Pinned Back-Four',
      };
    }
    if (causeTel?.triangle_track_ids && causeTel.triangle_track_ids.length >= 3) {
      return {
        type: 'BUILDUP_POLYGON',
        track_ids: causeTel.triangle_track_ids,
        closed_polygon: false,
        unit_label: 'Passing Options',
      };
    }
    if (causeTel?.midfield_track_ids && causeTel.midfield_track_ids.length >= 2) {
      return {
        type: 'DEFENSIVE_LINE',
        track_ids: causeTel.midfield_track_ids,
        closed_polygon: false,
        unit_label: 'Midfield Unit',
      };
    }
    return undefined;
  })();

  const resolvedExploitZone: TacticalExploitZone | undefined = (() => {
    if (exploit_zone && exploit_zone.vertices_pct && exploit_zone.vertices_pct.length >= 3) {
      return exploit_zone;
    }
    if (causeTel?.exploit_zone && causeTel.exploit_zone.vertices_pct && causeTel.exploit_zone.vertices_pct.length >= 3) {
      return causeTel.exploit_zone;
    }
    if (causeTel?.vacated_zone_polygon && causeTel.vacated_zone_polygon.length >= 3) {
      return {
        label: causeTel.vacated_label || 'EXPLOITABLE HALF-SPACE',
        vertices_pct: causeTel.vacated_zone_polygon,
      };
    }
    return undefined;
  })();

  const resolvedVectors: TacticalVectorItem[] = (() => {
    const list: TacticalVectorItem[] = [];
    if (Array.isArray(vectors) && vectors.length > 0) {
      list.push(...vectors);
    } else if (Array.isArray(causeTel?.vectors) && causeTel.vectors.length > 0) {
      list.push(...causeTel.vectors);
    } else {
      if (causeTel?.pass_vector) {
        list.push({
          type: 'GROUND_PASS_SOLID',
          from_track_id: causeTel.pass_vector.from_track_id ?? causeTel.ball_handler_track_id,
          to_pct: [causeTel.pass_vector.to_x, causeTel.pass_vector.to_y],
        });
      }
      if (causeTel?.run_vectors) {
        causeTel.run_vectors.forEach((rv) => {
          list.push({
            type: 'PLAYER_RUN_DASHED',
            from_track_id: rv.track_id,
            to_pct: [rv.to_x, rv.to_y],
          });
        });
      }
    }

    // Dynamic reconstruction fallback: If list is still empty, synthesize tactical vectors from detected players!
    if (list.length === 0) {
      const cPlayers = cause_telestration?.players || (causeTelestration as any)?.players;
      if (Array.isArray(cPlayers) && cPlayers.length > 1) {
        const catP = catalyst || cPlayers[0];
        const catId = catP?.track_id ?? catP?.id;
        const catX = (catP as any)?.x_pct ?? (catP as any)?.x ?? 50;
        const catY = (catP as any)?.y_pct ?? (catP as any)?.y ?? 50;

        const others = cPlayers.filter((p: any) => String(p.id ?? p.track_id) !== String(catId));
        if (others.length > 0) {
          // Primary ground pass to first teammate
          const p1 = others[0];
          const p1Id = p1.id ?? p1.track_id;
          const p1X = p1.x_pct ?? p1.x ?? (catX + 15);
          const p1Y = p1.y_pct ?? p1.y ?? catY;
          list.push({
            type: 'GROUND_PASS_SOLID',
            from_track_id: catId,
            to_track_id: p1Id,
            from_pct: [catX, catY],
            to_pct: [p1X, p1Y],
            landing_target_pct: [p1X, p1Y],
            arc_height_pct: 20.0,
            was_taken: true,
            color: '#ffffff',
          });

          // Dashed pass options to other players
          for (let i = 1; i < others.length; i++) {
            const po = others[i];
            const poId = po.id ?? po.track_id;
            const poX = po.x_pct ?? po.x;
            const poY = po.y_pct ?? po.y;
            list.push({
              type: 'PASS_OPTION_DASHED',
              from_track_id: catId,
              to_track_id: poId,
              from_pct: [catX, catY],
              to_pct: [poX, poY],
              name_pos: po.name_pos || po.name,
              was_taken: false,
              is_option: true,
              color: 'rgba(255, 255, 255, 0.65)',
            });
          }
        }
      }
    }
    // Filter out candidate pass options that duplicate the executed pass receiver
    const executedPass = list.find((v) => (v.type === 'GROUND_PASS_SOLID' || v.type === 'AERIAL_BALL_ARC') && v.was_taken !== false);
    if (executedPass && (executedPass.to_track_id || executedPass.to_pct || executedPass.landing_target_pct)) {
      const execTargetPct = executedPass.to_pct || executedPass.landing_target_pct;
      const exX = Array.isArray(execTargetPct) ? execTargetPct[0] : (execTargetPct as any)?.x;
      const exY = Array.isArray(execTargetPct) ? execTargetPct[1] : (execTargetPct as any)?.y;
      const exTid = executedPass.to_track_id !== undefined ? String(executedPass.to_track_id) : null;

      return list.filter((v) => {
        if (v.type === 'PASS_OPTION_DASHED' || (v.type === 'GROUND_PASS_SOLID' && v.was_taken === false)) {
          if (exTid && v.to_track_id !== undefined && String(v.to_track_id) === exTid) {
            return false;
          }
          if (exX !== undefined && exY !== undefined && v.to_pct) {
            const vx = Array.isArray(v.to_pct) ? v.to_pct[0] : (v.to_pct as any).x;
            const vy = Array.isArray(v.to_pct) ? v.to_pct[1] : (v.to_pct as any).y;
            if (Math.hypot(vx - exX, vy - exY) < 8.0) {
              return false;
            }
          }
        }
        return true;
      });
    }

    return list;
  })();

  const primaryRunnerId: string | undefined = (() => {
    for (const v of resolvedVectors) {
      if (v.type === 'PLAYER_RUN_DASHED' && v.from_track_id !== undefined) {
        const idStr = String(v.from_track_id);
        if (idStr !== catalystId) {
          return idStr;
        }
      }
    }
    if (causeTel?.runner_track_id && String(causeTel.runner_track_id) !== catalystId) {
      return String(causeTel.runner_track_id);
    }
    if (causeTel?.focal_track_ids) {
      for (const fid of causeTel.focal_track_ids) {
        if (String(fid) !== catalystId) {
          return String(fid);
        }
      }
    }
    return undefined;
  })();

  // --- Vector Phasing & Mutual Exclusivity Architecture ---
  const candidatePassVectors = resolvedVectors.filter(
    (vec) => vec.was_taken === false || vec.is_option || vec.type === 'PASS_OPTION_DASHED'
  );
  const executedPassVectors = resolvedVectors.filter(
    (vec) => (vec.type === 'GROUND_PASS_SOLID' || vec.type === 'AERIAL_BALL_ARC') && vec.was_taken !== false
  );
  const hasCandidateOptions = candidatePassVectors.length > 0;
  const hasExecutedPass = executedPassVectors.length > 0;

  // Visual cue tags for candidate options vs executed real pass
  const passOptionTags = visualCueTags.filter(
    (t) => t.type === 'candidate_passes' || t.type === 'pass_options' || t.type === 'vectors'
  );
  const executedPassTags = visualCueTags.filter(
    (t) => t.type === 'executed_pass' || t.type === 'real_pass'
  );

  // Timing thresholds:
  // Phase A: Options Exploration
  const defaultOptStart = graphicsStart + Math.round(graphicsDuration * 0.08);
  const earliestOptStart = passOptionTags.length > 0
    ? Math.min(...passOptionTags.map((t) => t.start_frame ?? defaultOptStart))
    : defaultOptStart;

  const defaultOptEnd = graphicsStart + Math.round(graphicsDuration * 0.44);
  const optionPhaseEnd = passOptionTags.length > 0
    ? Math.min(...passOptionTags.map((t) => t.end_frame ?? defaultOptEnd))
    : defaultOptEnd;

  // Phase B: Executed Real Decision
  const defaultExecStart = hasCandidateOptions
    ? Math.max(optionPhaseEnd + 4, graphicsStart + Math.round(graphicsDuration * 0.48))
    : (graphicsStart + Math.round(graphicsDuration * vectorThreshold));
  const executedStart = executedPassTags.length > 0
    ? Math.min(...executedPassTags.map((t) => t.start_frame ?? defaultExecStart))
    : defaultExecStart;

  // Candidate pass options envelope:
  // Smooth broadcast deployment over 36 frames (~1.2s) so viewers clearly see long arrows deploy
  const optFadeIn = isGraphicsActive && frame >= earliestOptStart
    ? Math.min(1, Math.max(0, (frame - earliestOptStart) / 36))
    : 0;
  const optFadeOut = isGraphicsActive && frame < optionPhaseEnd
    ? (frame > optionPhaseEnd - 12 ? Math.max(0, (optionPhaseEnd - frame) / 12) : 1.0)
    : 0;
  let optOpacity = isGraphicsActive && frame >= earliestOptStart && frame < optionPhaseEnd
    ? Math.min(1, optFadeIn * 1.5) * optFadeOut * freezeExitFade
    : 0;

  // Executed pass envelope:
  // Smooth broadcast deployment over 40 frames (~1.33s) so long pass arrows and arcs visibly deploy
  const executedProgress = isGraphicsActive && frame >= executedStart
    ? Math.min(1, Math.max(0, (frame - executedStart) / 40))
    : 0;
  let executedOpacity = isGraphicsActive && frame >= executedStart
    ? Math.min(1, executedProgress * 1.4) * freezeExitFade
    : 0;

  // HARD MUTUAL EXCLUSIVITY RULE:
  // Never show passing options arrows and real pass arrow at the same time!
  if (executedOpacity > 0.01) {
    optOpacity = 0;
  }
  if (optOpacity > 0.01) {
    executedOpacity = 0;
  }

  // Vectors Layer Visibility: active during options exploration or executed decision
  const showVectors = isGraphicsActive && (optOpacity > 0.01 || executedOpacity > 0.01 || graphicsRatio >= vectorThreshold) && freezeExitFade > 0;
  const vectorProgress = isGraphicsActive && graphicsRatio >= vectorThreshold
    ? Math.min(1, Math.max(0, (graphicsRatio - vectorThreshold) / 0.32))
    : 0;
  const vectorOpacity = isGraphicsActive ? (hasCandidateOptions ? 1.0 : Math.min(1, vectorProgress * 1.4) * freezeExitFade) : 0;

  // Resolve moving players/tracks:
  // For pre-warped video (effectiveIsWarped), playback is 1.0x native from frame 0 so active frame strictly tracks `frame` (locked to V_freeze during isPhase2 freeze).
  // For choreographed 3-phase video, Sequence 3 advances from V_freeze + 1.
  const activeFrameForTracks = effectiveIsWarped
    ? (isPhase2 ? V_freeze : (maxTracksFrame !== undefined ? Math.min(maxTracksFrame, Math.max(0, frame)) : frame))
    : isPhase2
    ? V_freeze
    : isPhase1
    ? (T_preroll > 0 ? Math.round((frame / T_preroll) * V_freeze) : frame)
    : isPhase3
    ? (maxTracksFrame !== undefined ? Math.min(maxTracksFrame, Math.max(0, V_freeze + 1 + (frame - freezeEnd))) : (V_freeze + 1 + (frame - freezeEnd)))
    : frame;

  const isMoving = isPhase1 || isPhase3;
  const currentTracks = tracksMap
    ? findTracksForFrame(tracksMap, activeFrameForTracks, isPhase2 ? 15 : 4, isMoving)
    : [];

  const ballCarrierMap = causeTel?.ball_carrier_per_frame ?? ball_carrier_per_frame ?? propBallCarrierPerFrame;

  // Authoritative Carrier Resolver with Proximity Search & Dynamic Phase 3 Pass Handoff
  const resolveCarrierForFrame = (f: number): string => {
    if (ballCarrierMap) {
      const fStr = String(f);
      if (ballCarrierMap[fStr] !== undefined) {
        return String(ballCarrierMap[fStr]);
      }
      for (let off = 1; off <= 15; off++) {
        if (ballCarrierMap[String(f - off)] !== undefined) return String(ballCarrierMap[String(f - off)]);
        if (ballCarrierMap[String(f + off)] !== undefined) return String(ballCarrierMap[String(f + off)]);
      }
    }
    // Dynamic handoff fallback in Phase 3 after ball transit window (~30 frames)
    if (isPhase3 && phase3Progress >= 30) {
      const executedPass = resolvedVectors.find(
        (v) => (v.type === 'GROUND_PASS_SOLID' || v.type === 'AERIAL_BALL_ARC') && v.was_taken !== false && v.to_track_id !== undefined
      );
      if (executedPass && executedPass.to_track_id !== undefined) {
        return String(executedPass.to_track_id);
      }
      if (primaryRunnerId) {
        return primaryRunnerId;
      }
    }
    return catalystId || '';
  };

  const dynamicCarrierId = resolveCarrierForFrame(activeFrameForTracks);
  const activeSpotlightId = dynamicCarrierId || catalystId;

  const focalIdSet = new Set<string>();
  if (dynamicCarrierId) focalIdSet.add(dynamicCarrierId);
  if (catalystId) focalIdSet.add(catalystId);
  if (primaryRunnerId) focalIdSet.add(primaryRunnerId);
  if (activeSpotlightId) focalIdSet.add(activeSpotlightId);
  if (structuralNetwork) {
    structuralNetwork.track_ids.forEach((id) => focalIdSet.add(String(id)));
  }
  if (resolvedVectors) {
    resolvedVectors.forEach((v) => {
      if (v.from_track_id !== undefined) focalIdSet.add(String(v.from_track_id));
      if (v.to_track_id !== undefined) focalIdSet.add(String(v.to_track_id));
    });
  }
  if (causeTel?.focal_track_ids && causeTel.focal_track_ids.length > 0) {
    causeTel.focal_track_ids.forEach((fid) => focalIdSet.add(String(fid)));
  }
  // Also add visualCueTags highlight targets to focalIdSet so they receive rock-solid temporal smoothing
  visualCueTags.forEach((t) => {
    if (t.type === 'highlight' && t.target) {
      const tgtClean = String(t.target).trim();
      if (/^\d+$/.test(tgtClean)) {
        focalIdSet.add(tgtClean);
      }
    }
  });

  const resolvedInitialPlayers: TelestrationPlayer[] = (() => {
    if (Array.isArray(initialPlayers) && initialPlayers.length > 0) {
      return initialPlayers;
    }
    const causeP = cause_telestration?.players || (causeTelestration as any)?.players;
    if (Array.isArray(causeP) && causeP.length > 0) {
      return causeP.map((cp: any) => ({
        id: String(cp.id ?? cp.track_id),
        name: cp.name,
        label: cp.label || cp.name,
        number: cp.number,
        role: cp.role,
        x: cp.x_pct ?? cp.x,
        y: cp.y_pct ?? cp.y,
        color: cp.color || (cp.team === 'away' ? awayColor : homeColor),
        showRing: cp.showRing ?? (String(cp.id ?? cp.track_id) === catalystId),
        showBeam: cp.showBeam ?? (String(cp.id ?? cp.track_id) === catalystId),
        showBadge: true,
      }));
    }
    return [];
  })();

  const isEntryOrFreezeWindow = isChoreographed
    ? (frame >= Math.max(0, T_preroll - 24) && frame <= freezeEnd + 24)
    : isPhase2;
  const combinedPlayers: TelestrationPlayer[] = (effectiveIsTactical && isEntryOrFreezeWindow)
    ? [...resolvedInitialPlayers]
    : [];
  if (currentTracks.length > 0) {
    currentTracks.forEach((tr) => {
      const idStr = String(tr.track_id);
      const isCarrier = activeSpotlightId ? idStr === activeSpotlightId : (catalystId ? idStr === catalystId : false);
      const isFocal = focalIdSet.has(idStr);
      const tColor = tr.color || (isCarrier ? '#00e5ff' : (tr.team === 'away' ? awayColor : homeColor));

      let trX = tr.x_pct;
      let trY = tr.y_pct;
      if (tr.bbox && Array.isArray(tr.bbox) && tr.bbox.length >= 4) {
        const [ymin, xmin, ymax, xmax] = tr.bbox;
        if (ymax > ymin && xmax > xmin) {
          trX = ((xmin + xmax) / (2 * 1920)) * 100;
          trY = (ymax / 1080) * 100;
        }
      } else if ((tr as any).feet_y_pct !== undefined) {
        trY = (tr as any).feet_y_pct;
      }

      if (!combinedPlayers.some((p) => String(p.id) === idStr)) {
        combinedPlayers.push({
          id: idStr,
          name: tr.name,
          label: tr.label,
          number: tr.number,
          role: tr.role,
          x: trX,
          y: trY,
          color: tColor,
          showRing: isCarrier,
          showBeam: isCarrier,
          showBadge: isCarrier,
        });
      }
    });
  }

  // Robustness fallbacks: ONLY during isPhase2 freeze! During moving video (Sequence 1 & 3), NEVER fall back to static coordinates!
  // Strictly restricted to ±10 frames around activeFrameForTracks to prevent grabbing stale positions from minutes ago.
  if (isPhase2) {
    if (catalystId && !combinedPlayers.some((p) => String(p.id) === catalystId)) {
      let fallbackTrack: TrackItem | undefined;
      if (tracksMap) {
        const sortedKeys = Object.keys(tracksMap)
          .map(Number)
          .filter((n) => !isNaN(n))
          .sort((a, b) => Math.abs(a - activeFrameForTracks) - Math.abs(b - activeFrameForTracks));
        for (const fKey of sortedKeys) {
          if (Math.abs(fKey - activeFrameForTracks) > 10) break;
          const trList = tracksMap[fKey];
          if (Array.isArray(trList)) {
            const found = trList.find((t) => String(t.track_id) === catalystId);
            if (found) {
              fallbackTrack = found;
              break;
            }
          }
        }
      }

      if (fallbackTrack) {
        const tColor = fallbackTrack.color || (fallbackTrack.team === 'away' ? awayColor : homeColor);
        let fbX = fallbackTrack.x_pct;
        let fbY = fallbackTrack.y_pct;
        if (fallbackTrack.bbox && Array.isArray(fallbackTrack.bbox) && fallbackTrack.bbox.length >= 4) {
          const [ymin, xmin, ymax, xmax] = fallbackTrack.bbox;
          if (ymax > ymin && xmax > xmin) {
            fbX = ((xmin + xmax) / (2 * 1920)) * 100;
            fbY = (ymax / 1080) * 100;
          }
        } else if ((fallbackTrack as any).feet_y_pct !== undefined) {
          fbY = (fallbackTrack as any).feet_y_pct;
        }

        combinedPlayers.push({
          id: catalystId,
          name: fallbackTrack.name || catalyst?.name_pos || 'BALL CARRIER',
          label: fallbackTrack.label || catalyst?.name_pos || 'BALL CARRIER',
          number: fallbackTrack.number,
          role: fallbackTrack.role || catalyst?.reason || 'BALL CARRIER',
          x: fbX,
          y: fbY,
          color: tColor,
          showRing: true,
          showBeam: true,
          showBadge: true,
        });
      } else if (causeTel?.pass_vector && (String(causeTel.pass_vector.from_track_id) === catalystId || String(causeTel.ball_handler_track_id) === catalystId)) {
        combinedPlayers.push({
          id: catalystId,
          name: catalyst?.name_pos || 'BALL CARRIER',
          label: catalyst?.name_pos || 'BALL CARRIER',
          role: 'BALL CARRIER',
          x: causeTel.pass_vector.from_x,
          y: causeTel.pass_vector.from_y,
          color: homeColor,
          showRing: true,
          showBeam: true,
          showBadge: true,
        });
      }
    }

    resolvedVectors.forEach((v) => {
      if (v.from_track_id !== undefined) {
        const rId = String(v.from_track_id);
        if (!combinedPlayers.some((p) => String(p.id) === rId)) {
          let fallbackRunner: TrackItem | undefined;
          if (tracksMap) {
            const sortedKeys = Object.keys(tracksMap)
              .map(Number)
              .filter((n) => !isNaN(n))
              .sort((a, b) => Math.abs(a - activeFrameForTracks) - Math.abs(b - activeFrameForTracks));
            for (const fKey of sortedKeys) {
              if (Math.abs(fKey - activeFrameForTracks) > 10) break;
              const trList = tracksMap[fKey];
              if (Array.isArray(trList)) {
                const found = trList.find((t) => String(t.track_id) === rId);
                if (found) {
                  fallbackRunner = found;
                  break;
                }
              }
            }
          }
          if (fallbackRunner) {
            const tColor = fallbackRunner.color || (fallbackRunner.team === 'away' ? awayColor : homeColor);
            let fbX = fallbackRunner.x_pct;
            let fbY = fallbackRunner.y_pct;
            if (fallbackRunner.bbox && Array.isArray(fallbackRunner.bbox) && fallbackRunner.bbox.length >= 4) {
              const [ymin, xmin, ymax, xmax] = fallbackRunner.bbox;
              if (ymax > ymin && xmax > xmin) {
                fbX = ((xmin + xmax) / (2 * 1920)) * 100;
                fbY = (ymax / 1080) * 100;
              }
            } else if ((fallbackRunner as any).feet_y_pct !== undefined) {
              fbY = (fallbackRunner as any).feet_y_pct;
            }

            combinedPlayers.push({
              id: rId,
              name: fallbackRunner.name || v.name_pos || 'THIRD-MAN RUNNER',
              label: fallbackRunner.label || v.name_pos || 'THIRD-MAN RUNNER',
              number: fallbackRunner.number,
              role: fallbackRunner.role || 'THIRD-MAN RUNNER',
              x: fbX,
              y: fbY,
              color: tColor,
              showRing: true,
              showBeam: false,
              showBadge: true,
            });
          } else {
            const rv = causeTel?.run_vectors?.find((r) => String(r.track_id) === rId);
            if (rv) {
              combinedPlayers.push({
                id: rId,
                name: v.name_pos || 'THIRD-MAN RUNNER',
                label: v.name_pos || 'THIRD-MAN RUNNER',
                role: 'THIRD-MAN RUNNER',
                x: rv.from_x,
                y: rv.from_y,
                color: awayColor,
                showRing: true,
                showBeam: false,
                showBadge: true,
              });
            }
          }
        }
      }
    });
  }

  // Filter to focal players for non-choreographed fallback (up to 9 players)
  const focalPlayers = (focalIdSet.size > 0
    ? combinedPlayers.filter((p) => focalIdSet.has(String(p.id)))
    : combinedPlayers.filter((p) => p.showRing || p.showBadge)
  ).slice(0, 9);

  const combinedLines: TelestrationLine[] = [...initialLines];
  const combinedPolygons: TelestrationPolygon[] = [...initialPolygons];

  // Coordinate normalization: percentage (<= 100) -> pixels
  const getCoords = (p: TelestrationPlayer) => ({
    x: p.x <= 100 ? (p.x / 100) * width : p.x,
    y: p.y <= 100 ? (p.y / 100) * height : p.y,
  });

  const playerMap = new Map<string, { x: number; y: number; p: TelestrationPlayer }>();
  combinedPlayers.forEach((p) => {
    playerMap.set(String(p.id), { ...getCoords(p), p });
  });

  // Dynamically update playerMap with per-frame tracked coordinates tracks_per_frame[String(currentVideoFrame)]
  if (currentTracks && currentTracks.length > 0) {
    currentTracks.forEach((tr) => {
      const idStr = String(tr.track_id);
      let px = tr.x_pct <= 100 ? (tr.x_pct / 100) * width : tr.x_pct;
      let py = tr.y_pct <= 100 ? (tr.y_pct / 100) * height : tr.y_pct;
      if (tr.bbox && Array.isArray(tr.bbox) && tr.bbox.length >= 4) {
        const [ymin, xmin, ymax, xmax] = tr.bbox;
        if (ymax > ymin && xmax > xmin) {
          px = ((xmin + xmax) / (2 * 1920)) * width;
          py = (ymax / 1080) * height;
        }
      } else if ((tr as any).feet_y_pct !== undefined) {
        py = ((tr as any).feet_y_pct / 100) * height;
      }
      if (tracksMap) {
        const sm = getSmoothedTurfPosition(tracksMap, idStr, activeFrameForTracks, width, height, isMoving);
        if (sm.isPresent) {
          px = sm.x;
          py = sm.y;
        }
      }
      const existing = playerMap.get(idStr);
      if (existing) {
        existing.x = px;
        existing.y = py;
        existing.p.x = (px / width) * 100;
        existing.p.y = (py / height) * 100;
        if (tr.team) existing.p.team = tr.team;
      } else {
        const tColor = tr.color || (tr.team === 'away' ? awayColor : homeColor);
        playerMap.set(idStr, {
          x: px,
          y: py,
          p: {
            id: idStr,
            name: tr.name,
            label: tr.label,
            number: tr.number,
            role: tr.role,
            team: tr.team,
            x: (px / width) * 100,
            y: (py / height) * 100,
            color: tColor,
            showRing: idStr === catalystId,
            showBeam: idStr === catalystId,
            showBadge: idStr === catalystId,
          },
        });
      }
    });
  }

  // Ensure every focal player (catalyst, runners, highlighted players, vector actors) has rock-solid smoothed coordinates even if YOLO temporarily drops a frame
  if (tracksMap) {
    focalIdSet.forEach((fId) => {
      const idStr = String(fId);
      const sm = getSmoothedTurfPosition(tracksMap, idStr, activeFrameForTracks, width, height, isMoving);
      if (sm.isPresent) {
        const existing = playerMap.get(idStr);
        if (existing) {
          existing.x = sm.x;
          existing.y = sm.y;
          existing.p.x = (sm.x / width) * 100;
          existing.p.y = (sm.y / height) * 100;
        } else {
          let sampleTrack: TrackItem | undefined;
          for (const off of [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]) {
            const trList = tracksMap[activeFrameForTracks + off] || tracksMap[String(activeFrameForTracks + off)];
            if (Array.isArray(trList)) {
              sampleTrack = trList.find((t) => String(t.track_id) === idStr);
              if (sampleTrack) break;
            }
          }
          const tColor = sampleTrack?.color || (sampleTrack?.team === 'away' ? awayColor : homeColor);
          playerMap.set(idStr, {
            x: sm.x,
            y: sm.y,
            p: {
              id: idStr,
              name: sampleTrack?.name || 'Player',
              label: sampleTrack?.label || 'Player',
              number: sampleTrack?.number,
              role: sampleTrack?.role || 'PLAYER',
              team: sampleTrack?.team,
              x: (sm.x / width) * 100,
              y: (sm.y / height) * 100,
              color: tColor,
              showRing: idStr === catalystId,
              showBeam: idStr === catalystId,
              showBadge: idStr === catalystId,
            },
          });
        }
      }
    });
  }

  // Defending presser IDs that already have a dedicated vector in resolvedVectors
  const vectorPresserIds = new Set<string>();
  const primaryPassVectors: { x1: number; y1: number; x2: number; y2: number }[] = [];

  resolvedVectors.forEach((v) => {
    if (v.type === 'PRESSING_RUN' || v.type === 'PRESSING_ARROW') {
      if (v.from_track_id !== undefined) {
        vectorPresserIds.add(String(v.from_track_id));
      }
    } else if (v.type === 'GROUND_PASS_SOLID' || v.type === 'AERIAL_BALL_ARC') {
      const fromP = v.from_track_id ? playerMap.get(String(v.from_track_id)) : (catalystId ? playerMap.get(catalystId) : null);
      if (fromP) {
        const toP = v.to_track_id ? playerMap.get(String(v.to_track_id)) : null;
        const targetX = toP
          ? toP.x
          : (v.to_pct ? ((Array.isArray(v.to_pct) ? v.to_pct[0] : v.to_pct.x) / 100) * width : fromP.x + 0.15 * width);
        const targetY = toP
          ? toP.y
          : (v.to_pct ? ((Array.isArray(v.to_pct) ? v.to_pct[1] : v.to_pct.y) / 100) * height : fromP.y);
        primaryPassVectors.push({ x1: fromP.x, y1: fromP.y, x2: targetX, y2: targetY });
      }
    }
  });

  // Tactical Line-of-Sight and Pressure/Cover Interception Engine:
  // Strictly checks whether a candidate passing option is physically blocked by an opposing defender (e.g. Tchouaméni in front of Ndicka)
  // or if the intended teammate is tightly marked / covered under defensive pressure.
  const isPassLaneBlockedOrCovered = (
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    carrierTeam: string
  ): { isBlocked: boolean; blockerId?: string; reason?: string } => {
    const segDx = x2 - x1;
    const segDy = y2 - y1;
    const segLenSq = segDx * segDx + segDy * segDy;
    const segLen = Math.sqrt(segLenSq);

    if (segLenSq <= 0) return { isBlocked: true, reason: 'zero_length' };

    // 1. Minimum passing distance check (suppress trivial clutter < 8% pitch width)
    if (segLen < 0.08 * width) {
      return { isBlocked: true, reason: 'too_short' };
    }

    for (const [id, otherP] of playerMap.entries()) {
      const oTeam = otherP.p?.team;
      const isOpponent = oTeam ? oTeam !== carrierTeam : (carrierTeam === 'home' ? defTeam === 'away' : defTeam === 'home');
      if (!isOpponent) continue;

      // Check A: Is receiver covered / marked / under heavy defensive pressure?
      // Suppresses fake options where a defender is standing directly on or next to the receiver (< 85px).
      const distToTarget = Math.hypot(otherP.x - x2, otherP.y - y2);
      if (distToTarget < 85) {
        return { isBlocked: true, blockerId: id, reason: 'receiver_covered' };
      }

      // Check B: Passing corridor occlusion: Is any opposing defender in the passing trajectory?
      const t = ((otherP.x - x1) * segDx + (otherP.y - y1) * segDy) / segLenSq;
      if (t >= 0.03 && t <= 0.97) {
        const projX = x1 + t * segDx;
        const projY = y1 + t * segDy;
        const perpDist = Math.hypot(otherP.x - projX, otherP.y - projY);

        // Proximity-dependent interception cylinder:
        // - Close to carrier (t <= 0.40): opponent is in immediate forward pressing/blocking cone (e.g. Tchouaméni at t=0.168, dist=54.4px).
        //   Interception reach is wide (105px in 1080p canvas, approx 5.5% screen width).
        // - Further down the passing lane (t > 0.40): interception lane width is 85px (approx 4.4% screen width).
        const corridorThreshold = t <= 0.40 ? 105 : 85;
        if (perpDist < corridorThreshold) {
          return { isBlocked: true, blockerId: id, reason: 'lane_blocked' };
        }
      }
    }

    return { isBlocked: false };
  };

  const normalizeStr = (s: string | undefined | null) =>
    s ? s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]/g, '') : '';

  const matchPlayerForHighlight = (target: string | undefined): string | null => {
    if (!target) return null;
    const cleanTgt = normalizeStr(target);
    if (!cleanTgt) return null;

    // Catalyst and Runner aliases
    if (['carrier', 'ballcarrier', 'catalyst', 'thecarrier'].includes(cleanTgt) && catalystId) {
      return catalystId;
    }
    if (['runner', 'thirdmanrunner', 'primaryrunner', 'therunner'].includes(cleanTgt) && primaryRunnerId) {
      return primaryRunnerId;
    }

    const numMatch = target.match(/#?(\d+)/);
    const targetNum = numMatch ? numMatch[1] : null;

    // Role mapping
    const isGk = ['gk', 'goalkeeper', 'keeper', 'goalie'].includes(cleanTgt);
    const isCb = ['cb', 'centerback', 'centreback', 'defender'].includes(cleanTgt);
    const isCdm = ['cdm', 'dm', 'pivot', 'holdingmidfielder'].includes(cleanTgt);
    const isCm = ['cm', 'centralmidfielder', 'midfielder'].includes(cleanTgt);
    const isCam = ['cam', 'am', 'attackingmidfielder', 'playmaker'].includes(cleanTgt);
    const isRw = ['rw', 'rightwing', 'winger'].includes(cleanTgt);
    const isLw = ['lw', 'leftwing'].includes(cleanTgt);
    const isCf = ['st', 'cf', 'striker', 'forward'].includes(cleanTgt);

    const targetWords = cleanTgt.length >= 3
      ? target.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().split(/\s+/).filter((w) => w.length >= 3)
      : [];

    for (const [idStr, cp] of playerMap.entries()) {
      const pName = normalizeStr(cp.p.name || cp.p.label || '');
      const pShort = normalizeStr(cp.p.label || '');
      const pRole = (cp.p.role || '').toLowerCase();
      const pNum = cp.p.number !== undefined ? String(cp.p.number) : '';

      // 1. Direct number match
      if (targetNum && pNum === targetNum) return idStr;

      // 2. Exact or substring match on player name (require min length 3 on both sides to avoid empty string bug!)
      if (cleanTgt.length >= 3 && pName.length >= 3 && (pName.includes(cleanTgt) || cleanTgt.includes(pName))) return idStr;
      if (cleanTgt.length >= 3 && pShort.length >= 3 && (pShort.includes(cleanTgt) || cleanTgt.includes(pShort))) return idStr;

      // 3. Word-level match (e.g. "Konate" matching "Ibrahima Konate")
      for (const tw of targetWords) {
        if (pName.length >= 3 && pName.includes(tw)) return idStr;
        if (pShort.length >= 3 && pShort.includes(tw)) return idStr;
      }

      // 4. Role match
      if (pRole) {
        if (isGk && (pRole.includes('gk') || pRole.includes('keeper'))) return idStr;
        if (isCb && (pRole.includes('cb') || pRole.includes('def'))) return idStr;
        if (isCdm && (pRole.includes('cdm') || pRole.includes('dm') || pRole.includes('pivot'))) return idStr;
        if (isCm && (pRole.includes('cm') || pRole.includes('mid'))) return idStr;
        if (isCam && (pRole.includes('cam') || pRole.includes('am'))) return idStr;
        if (isRw && (pRole.includes('rw') || pRole.includes('wing'))) return idStr;
        if (isLw && (pRole.includes('lw') || pRole.includes('wing'))) return idStr;
        if (isCf && (pRole.includes('st') || pRole.includes('cf') || pRole.includes('striker'))) return idStr;
      }
    }

    return null;
  };

  // Exploit zone polygon coordinates (strictly 4-point quad) - NO 3-point triangle or pancake ellipse
  const exploitPts = (() => {
    if (!resolvedExploitZone || !resolvedExploitZone.vertices_pct || resolvedExploitZone.vertices_pct.length === 0) {
      return null;
    }
    let rawVertices = resolvedExploitZone.vertices_pct.map((v) => {
      const rawX = typeof v === 'object' && v !== null && 'x' in v ? (v as { x: number; y: number }).x : (v as [number, number])[0];
      const rawY = typeof v === 'object' && v !== null && 'y' in v ? (v as { x: number; y: number }).y : (v as [number, number])[1];
      // Strict Pitch Grass Clamping (Zero Ad-Board Overlap): clamp vy to [31.0, 74.0] and vx to [20.0, 80.0]
      const clampedX = Math.max(20.0, Math.min(80.0, rawX <= 100 ? rawX : (rawX / width) * 100));
      const clampedY = Math.max(31.0, Math.min(74.0, rawY <= 100 ? rawY : (rawY / height) * 100));
      return { x: clampedX, y: clampedY };
    });

    // If not exactly 4 vertices, construct a 4-point perspective trapezoid/quad around the centroid
    if (rawVertices.length !== 4) {
      const cX = rawVertices.length > 0
        ? rawVertices.reduce((acc, v) => acc + v.x, 0) / rawVertices.length
        : 50.0;
      const cY = rawVertices.length > 0
        ? rawVertices.reduce((acc, v) => acc + v.y, 0) / rawVertices.length
        : 50.0;
      rawVertices = [
        { x: Math.max(20.0, Math.min(80.0, cX - 5.5)), y: Math.max(31.0, Math.min(74.0, cY - 4.2)) },
        { x: Math.max(20.0, Math.min(80.0, cX + 5.5)), y: Math.max(31.0, Math.min(74.0, cY - 4.2)) },
        { x: Math.max(20.0, Math.min(80.0, cX + 7.8)), y: Math.max(31.0, Math.min(74.0, cY + 4.2)) },
        { x: Math.max(20.0, Math.min(80.0, cX - 7.8)), y: Math.max(31.0, Math.min(74.0, cY + 4.2)) },
      ];
    }

    // Verify consecutive edge lengths >= 4.5% screen distance: prevent vertices from collapsing onto the same point
    for (let iter = 0; iter < 4; iter++) {
      let changed = false;
      for (let i = 0; i < rawVertices.length; i++) {
        const nextIdx = (i + 1) % rawVertices.length;
        const v1 = rawVertices[i];
        const v2 = rawVertices[nextIdx];
        const dist = Math.hypot(v1.x - v2.x, v1.y - v2.y);
        if (dist < 4.5) {
          changed = true;
          const needed = 4.6 - dist;
          const dx = v2.x - v1.x;
          const dy = v2.y - v1.y;
          const curD = Math.hypot(dx, dy);
          if (curD < 1e-3) {
            v2.x = Math.max(20.0, Math.min(80.0, v2.x + 2.5));
            v2.y = Math.max(31.0, Math.min(74.0, v2.y + 2.5));
            v1.x = Math.max(20.0, Math.min(80.0, v1.x - 2.5));
            v1.y = Math.max(31.0, Math.min(74.0, v1.y - 2.5));
          } else {
            const ux = dx / curD;
            const uy = dy / curD;
            v2.x = Math.max(20.0, Math.min(80.0, v2.x + ux * (needed / 2)));
            v2.y = Math.max(31.0, Math.min(74.0, v2.y + uy * (needed / 2)));
            v1.x = Math.max(20.0, Math.min(80.0, v1.x - ux * (needed / 2)));
            v1.y = Math.max(31.0, Math.min(74.0, v1.y - uy * (needed / 2)));
          }
        }
      }
      if (!changed) break;
    }

    return rawVertices.map((pt) => ({
      x: (pt.x / 100) * width,
      y: (pt.y / 100) * height,
    }));
  })();

  const exploitCentroid =
    exploitPts && exploitPts.length >= 3
      ? {
          x: exploitPts.reduce((acc, c) => acc + c.x, 0) / exploitPts.length,
          y: exploitPts.reduce((acc, c) => acc + c.y, 0) / exploitPts.length,
          label: (resolvedExploitZone?.label || 'EXPLOITABLE HALF-SPACE').toUpperCase(),
        }
      : null;

  // 3 High-Contrast Primary Pills resolution & mutual repulsion pass (min 200px H, 60px V)
  const cp = catalystId ? playerMap.get(catalystId) : null;
  let catalystBadgeText = (
    catalyst?.name_pos ||
    catalyst?.role ||
    (cp?.p.name ? `${cp.p.name} // BALL CARRIER` : 'BALL CARRIER')
  ).toUpperCase();
  if (isForbiddenPlayer(catalystBadgeText)) {
    const focalTag = activeHighlightTags.find((t) => t.player_name || t.target) || 
                     (visualCueTags || []).find((t: any) => t.player_name || t.target);
    if (focalTag) {
      catalystBadgeText = `${(focalTag.player_name || focalTag.target || '').toUpperCase()} // BALL CARRIER`;
    } else {
      catalystBadgeText = 'BALL CARRIER';
    }
  }

  const highestVertex =
    exploitPts && exploitPts.length > 0
      ? exploitPts.reduce((minP, p) => (p.y < minP.y ? p : minP), exploitPts[0])
      : null;
  const exploitMinY = highestVertex ? highestVertex.y : exploitCentroid ? exploitCentroid.y - 40 : height / 2;
  const anchorX = highestVertex ? highestVertex.x : exploitCentroid ? exploitCentroid.x : width / 2;

  const rp = primaryRunnerId ? playerMap.get(primaryRunnerId) : null;
  const vMatch = resolvedVectors.find(
    (v) => String(v.from_track_id) === primaryRunnerId && v.type === 'PLAYER_RUN_DASHED'
  );
  let runnerBadgeText = 'THIRD-MAN RUNNER';
  if (vMatch?.name_pos && vMatch.name_pos.toUpperCase().includes('RUNNER')) {
    runnerBadgeText = vMatch.name_pos.toUpperCase();
  } else if (rp?.p.role && rp.p.role.toUpperCase().includes('RUNNER')) {
    runnerBadgeText = rp.p.role.toUpperCase();
  } else if (rp?.p.name && rp.p.name.toUpperCase().includes('RUNNER')) {
    runnerBadgeText = rp.p.name.toUpperCase();
  }
  if (isForbiddenPlayer(runnerBadgeText)) {
    runnerBadgeText = 'THIRD-MAN RUNNER';
  }

  // Initial anchors for the 3 primary pills
  const repelledPills = [
    {
      id: 'catalyst',
      x: Math.max(160, Math.min(width - 160, cp ? cp.x : width / 2)),
      y: Math.max(50, Math.min(height - 50, cp ? cp.y + 35 : height / 2)),
      active: !!cp,
    },
    {
      id: 'exploit',
      x: Math.max(160, Math.min(width - 160, anchorX)),
      y: Math.max(50, Math.min(height - 50, exploitMinY - 45)),
      active: !!exploitCentroid,
    },
    {
      id: 'runner',
      x: Math.max(160, Math.min(width - 160, rp ? rp.x : width / 2)),
      y: Math.max(50, Math.min(height - 50, rp ? rp.y + 35 : height / 2)),
      active: !!rp,
    },
  ];

  // Mutual repulsion pass across all 3 pills: min 200px horizontal, 60px vertical separation
  const MIN_PILL_DIST_X = 200;
  const MIN_PILL_DIST_Y = 60;
  for (let iter = 0; iter < 8; iter++) {
    let anyCollision = false;
    for (let i = 0; i < repelledPills.length; i++) {
      if (!repelledPills[i].active) continue;
      for (let j = i + 1; j < repelledPills.length; j++) {
        if (!repelledPills[j].active) continue;
        const dx = Math.abs(repelledPills[i].x - repelledPills[j].x);
        const dy = Math.abs(repelledPills[i].y - repelledPills[j].y);
        if (dx < MIN_PILL_DIST_X && dy < MIN_PILL_DIST_Y) {
          anyCollision = true;
          const overlapX = MIN_PILL_DIST_X - dx;
          const overlapY = MIN_PILL_DIST_Y - dy;

          if (overlapY <= overlapX * 0.6 || dy < 25) {
            const pushY = overlapY / 2 + 2;
            if (repelledPills[i].y <= repelledPills[j].y) {
              repelledPills[i].y = Math.max(50, repelledPills[i].y - pushY);
              repelledPills[j].y = Math.min(height - 50, repelledPills[j].y + pushY);
            } else {
              repelledPills[i].y = Math.min(height - 50, repelledPills[i].y + pushY);
              repelledPills[j].y = Math.max(50, repelledPills[j].y - pushY);
            }
          } else {
            const pushX = overlapX / 2 + 3;
            if (repelledPills[i].x <= repelledPills[j].x) {
              repelledPills[i].x = Math.max(160, repelledPills[i].x - pushX);
              repelledPills[j].x = Math.min(width - 160, repelledPills[j].x + pushX);
            } else {
              repelledPills[i].x = Math.min(width - 160, repelledPills[i].x + pushX);
              repelledPills[j].x = Math.max(160, repelledPills[j].x - pushX);
            }
          }
        }
      }
    }
    if (!anyCollision) break;
  }

  const repelledCatPill = repelledPills[0];
  const repelledExpPill = repelledPills[1];
  const repelledRunPill = repelledPills[2];

  // Calculate default focal centroid for camera zoom center
  let focusCenterX = '50%';
  let focusCenterY = '50%';
  if (exploitPts && exploitPts.length > 0) {
    const avgX = exploitPts.reduce((acc, pt) => acc + (pt.x / width) * 100, 0) / exploitPts.length;
    const avgY = exploitPts.reduce((acc, pt) => acc + (pt.y / height) * 100, 0) / exploitPts.length;
    focusCenterX = `${avgX}%`;
    focusCenterY = `${avgY}%`;
  } else if (catalystId && playerMap.has(catalystId)) {
    const cp = playerMap.get(catalystId)!;
    focusCenterX = `${(cp.x / width) * 100}%`;
    focusCenterY = `${(cp.y / height) * 100}%`;
  } else if (combinedPlayers.length > 0) {
    const avgX = combinedPlayers.reduce((acc, p) => acc + (p.x <= 100 ? p.x : (p.x / width) * 100), 0) / combinedPlayers.length;
    const avgY = combinedPlayers.reduce((acc, p) => acc + (p.y <= 100 ? p.y : (p.y / height) * 100), 0) / combinedPlayers.length;
    focusCenterX = `${avgX}%`;
    focusCenterY = `${avgY}%`;
  }

  // --- Dynamic Player Spotlight & Animated Camera Pan Engine ---
  // Tracks AI-spoken player name tags ([highlight:PlayerName]) and drives:
  // 1. Smooth fluid push-in zoom (1.0x -> 1.30x) centered on the spoken player
  // 2. Focused spotlight vignette mask dimming the pitch with feathered cutout on the player
  // 3. Continuous camera pan/drag between players when multiple players are mentioned in sequence
  // 4. Smooth easing back to wide 1.0x view before freeze ends
  const highlightTargets = (() => {
    const list: Array<{
      tag: VisualCueTag;
      playerId: string;
      xPx: number;
      yPx: number;
      xPct: number;
      yPct: number;
      startFrame: number;
      endFrame: number;
      playerName: string;
    }> = [];

    const hlTags = visualCueTags.filter((t) => t.type === 'highlight');
    for (const tag of hlTags) {
      const targetId = matchPlayerForHighlight(tag.target || tag.player_name);
      if (targetId && playerMap.has(targetId)) {
        const pl = playerMap.get(targetId)!;
        const sF = tag.start_frame ?? graphicsStart;
        const dur = tag.duration_frames ?? 75;
        const eF = tag.end_frame ?? (sF + dur);
        // Center vertically on player's torso rather than feet
        const torsoY = Math.max(30, pl.y - 48);
        list.push({
          tag,
          playerId: targetId,
          xPx: pl.x,
          yPx: torsoY,
          xPct: (pl.x / width) * 100,
          yPct: (torsoY / height) * 100,
          startFrame: sF,
          endFrame: eF,
          playerName: tag.player_name || pl.p?.name || (pl.p as any)?.short_name || 'Player',
        });
      }
    }

    // Automatic Dual-Player Broadcast Slide:
    // If fewer than 2 highlight targets are specified, automatically resolve a second key player
    // (from commentary text mention, runner_player, primaryRunnerId, or pass target)
    // to drive a seamless broadcast camera slide between both players during the freeze!
    if (list.length < 2 && isChoreographed) {
      const p1Id = list.length === 1
        ? list[0].playerId
        : (activeSpotlightId || catalystId || (causeTel?.ball_handler_track_id !== undefined ? String(causeTel.ball_handler_track_id) : undefined));

      // Extract spoken commentary text from beat_2_cause and beat_2_steps
      let b2Text = '';
      if (typeof beats?.beat_2_cause === 'string') {
        b2Text = beats.beat_2_cause;
      } else if ((beats?.beat_2_cause as any)?.text) {
        b2Text = (beats.beat_2_cause as any).text;
      }
      if (beats?.beat_2_steps) {
        b2Text += ' ' + Object.values(beats.beat_2_steps).join(' ');
      }

      let p2Id: string | undefined = undefined;

      // 1. Check if commentary explicitly mentions any other player on pitch
      if (b2Text) {
        const b2Norm = b2Text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
        let earliestPos = 999999;
        for (const [idStr, cp] of playerMap.entries()) {
          if (idStr === p1Id) continue;
          const pName = (cp.p?.name || cp.p?.label || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
          const tokens = [pName, ...pName.split(/\s+/)].filter((w) => w && w.length >= 3);
          for (const tok of tokens) {
            const idx = b2Norm.indexOf(tok);
            if (idx !== -1 && idx < earliestPos) {
              earliestPos = idx;
              p2Id = idStr;
            }
          }
        }
      }

      // 2. Check primaryRunnerId
      if (!p2Id && primaryRunnerId && primaryRunnerId !== p1Id && playerMap.has(primaryRunnerId)) {
        p2Id = primaryRunnerId;
      }

      // 3. Check causeTel.runner_player or runner_track_id
      if (!p2Id && causeTel?.runner_track_id && String(causeTel.runner_track_id) !== p1Id && playerMap.has(String(causeTel.runner_track_id))) {
        p2Id = String(causeTel.runner_track_id);
      }
      if (!p2Id && causeTel?.runner_player) {
        const rName = causeTel.runner_player.name || causeTel.runner_player.name_pos;
        const rMatched = matchPlayerForHighlight(rName);
        if (rMatched && rMatched !== p1Id && playerMap.has(rMatched)) {
          p2Id = rMatched;
        }
      }

      // 4. Check pass vectors (executed pass receiver or target landing)
      if (!p2Id) {
        for (const v of resolvedVectors) {
          if ((v.type === 'GROUND_PASS_SOLID' || v.type === 'AERIAL_BALL_ARC') && v.to_track_id !== undefined) {
            const vidStr = String(v.to_track_id);
            if (vidStr !== p1Id && playerMap.has(vidStr)) {
              p2Id = vidStr;
              break;
            }
          }
        }
      }

      // 5. Fallback: Check any focal track ids or player roster
      if (!p2Id && causeTel?.focal_track_ids) {
        for (const fid of causeTel.focal_track_ids) {
          const fidStr = String(fid);
          if (fidStr !== p1Id && playerMap.has(fidStr)) {
            p2Id = fidStr;
            break;
          }
        }
      }
      if (!p2Id && causeTel?.players && Array.isArray(causeTel.players)) {
        for (const cp of causeTel.players) {
          const cpid = String(cp.id || cp.track_id);
          if (cpid !== p1Id && playerMap.has(cpid)) {
            p2Id = cpid;
            break;
          }
        }
      }

      // Timing choreography: Split freeze between Player 1 (48%) and Player 2 (52%),
      // ending 18 frames before freezeEnd so RAMP_OUT (18f) finishes right at freezeEnd with zero double-bounce
      const EXIT_RAMP_FRAMES = 18;
      const freezeStart = graphicsStart;
      const freezeAvailable = Math.max(20, (freezeEnd - EXIT_RAMP_FRAMES) - freezeStart);
      const p1Duration = Math.floor(freezeAvailable * 0.48);
      const p1EndFrame = freezeStart + p1Duration;
      const p2StartFrame = p1EndFrame;
      const p2EndFrame = Math.max(p2StartFrame + 10, freezeEnd - EXIT_RAMP_FRAMES);

      if (p1Id && playerMap.has(p1Id)) {
        const pl1 = playerMap.get(p1Id)!;
        const torsoY1 = Math.max(30, pl1.y - 48);
        if (list.length === 0) {
          list.push({
            tag: { type: 'highlight', target: p1Id },
            playerId: p1Id,
            xPx: pl1.x,
            yPx: torsoY1,
            xPct: (pl1.x / width) * 100,
            yPct: (torsoY1 / height) * 100,
            startFrame: freezeStart,
            endFrame: p2Id ? p1EndFrame : (freezeEnd - EXIT_RAMP_FRAMES),
            playerName: pl1.p?.name || (pl1.p as any)?.short_name || 'Catalyst',
          });
        } else if (list.length === 1 && p2Id) {
          list[0].endFrame = p1EndFrame;
        }
      }

      if (p2Id && playerMap.has(p2Id)) {
        const pl2 = playerMap.get(p2Id)!;
        const torsoY2 = Math.max(30, pl2.y - 48);
        list.push({
          tag: { type: 'highlight', target: p2Id },
          playerId: p2Id,
          xPx: pl2.x,
          yPx: torsoY2,
          xPct: (pl2.x / width) * 100,
          yPct: (torsoY2 / height) * 100,
          startFrame: p2StartFrame,
          endFrame: p2EndFrame,
          playerName: pl2.p?.name || (pl2.p as any)?.short_name || 'Target',
        });
      }
    }

    return list.sort((a, b) => a.startFrame - b.startFrame);
  })();

  const defaultFocusNumX = parseFloat(focusCenterX) || 50;
  const defaultFocusNumY = parseFloat(focusCenterY) || 50;

  const {
    dynamicFocusX,
    dynamicFocusY,
    dynamicZoomScale,
    spotX,
    spotY,
    spotlightOpacity,
    spotRadiusX,
    spotRadiusY,
    spotInnerStopPct,
  } = (() => {
    // Dynamic player zoom & searching spotlight physics engine:
    // 1. Entry Ramp (26 frames: [firstH.startFrame - 20, firstH.startFrame + 6]):
    //    Smooth quintic C2 zoom-in (1.0x -> 1.55x) while the vignette aperture visibly scales down (760px -> 255px)
    //    and searches from pitch center onto the target player.
    // 2. Hesitant Physics Focus-Switching Glide (26..36 frames):
    //    Asymmetric hesitant detachment -> fluid mid-flight glide -> magnetic cushion settle,
    //    while the spotlight aperture breathes open (255px -> 360px -> 255px) to scan to the next player,
    //    and continuous freeze-wide zoom breathing eliminates the 1-frame reset at slideEnd + 1.
    // 3. Monotonic Exit Ramp (18 frames ending at freezeEnd):
    //    Single unified C2 zoom-out (1.55x -> 1.0x) with zero double-bounce at freezeEnd.
    const TARGET_ZOOM = (causeTel as any)?.player_zoom_scale ?? 1.55;
    const ENTRY_LEAD_IN = 20;
    const ENTRY_SETTLE_IN = 6;
    const RAMP_IN = ENTRY_LEAD_IN + ENTRY_SETTLE_IN; // 26 frames (0.87s)
    const RAMP_OUT = 18; // 0.60s monotonic exit aligned to freezeEnd
    const START_RADIUS = 760;
    const FOCUS_RADIUS = 255;
    const TARGET_DARKNESS = 0.84;

    // Quintic smoothstep (6t^5 - 15t^4 + 10t^3): zero velocity and zero acceleration at both endpoints (C2 continuous)
    const smoothstep5 = (t: number) => {
      const c = Math.max(0, Math.min(1, t));
      return c * c * c * (c * (c * 6 - 15) + 10);
    };

    // Asymmetric Hesitant Physics Glide Curve:
    // Gentle hesitant detachment during the first 35% of the slide, smooth mid-flight glide,
    // and soft magnetic cushion settle onto the target player.
    const hesitantGlideEase = (t: number) => {
      const c = Math.max(0, Math.min(1, t));
      const warped = Math.pow(c, 1.35) / (Math.pow(c, 1.35) + Math.pow(1 - c, 1.15));
      return smoothstep5(warped);
    };

    // Organic broadcast operator sway (natural steadycam / fluid-head tripod weight shifts & human breathing)
    const humanDriftX = Math.sin(frame * 0.033) * 0.45 + Math.cos(frame * 0.067) * 0.20;
    const humanDriftY = Math.cos(frame * 0.027) * 0.35 + Math.sin(frame * 0.059) * 0.16;
    const wideGlide = 1.0 + Math.sin(frame * 0.019) * 0.005;

    if (!effectiveIsTactical || isCloseUp || highlightTargets.length === 0) {
      const isZoomed = effectiveIsTactical && !isCloseUp && totalZoomScale > 1.001;
      const activeScale = isZoomed
        ? totalZoomScale + Math.sin(frame * 0.035) * 0.012
        : wideGlide;
      const actX = defaultFocusNumX + humanDriftX * (isZoomed ? 1.0 : 0.45);
      const actY = defaultFocusNumY + humanDriftY * (isZoomed ? 1.0 : 0.45);
      return {
        dynamicFocusX: actX,
        dynamicFocusY: actY,
        dynamicZoomScale: activeScale,
        spotX: (actX / 100) * width,
        spotY: (actY / 100) * height,
        spotlightOpacity: 0.0,
        spotRadiusX: START_RADIUS,
        spotRadiusY: START_RADIUS,
        spotInnerStopPct: 44,
      };
    }

    const firstH = highlightTargets[0];
    const lastH = highlightTargets[highlightTargets.length - 1];
    const effectiveLastEnd = Math.min(lastH.endFrame, freezeEnd - RAMP_OUT);

    const centerPxX = (defaultFocusNumX / 100) * width;
    const centerPxY = (defaultFocusNumY / 100) * height;

    // Continuous freeze-wide zoom breathing: depends ONLY on time since firstH.startFrame,
    // never resetting between players so there is ZERO 1-frame zoom snap at slideEnd + 1!
    const getContinuousFreezeDrift = (f: number) => {
      const elapsed = Math.max(0, f - (firstH.startFrame + ENTRY_SETTLE_IN));
      const pushIn = 0.055 * smoothstep5(Math.min(1, elapsed / 65));
      const breath = Math.sin(elapsed * 0.032) * 0.008;
      return pushIn + breath;
    };

    let fx = defaultFocusNumX;
    let fy = defaultFocusNumY;
    let scale = wideGlide;
    let sx = firstH.xPx;
    let sy = firstH.yPx;
    let currentRadius = START_RADIUS;
    let currentDarkness = 0.0;
    let innerStopPct = 44;

    const entryStartFrame = Math.max(0, firstH.startFrame - ENTRY_LEAD_IN);
    const entryLockFrame = firstH.startFrame + ENTRY_SETTLE_IN;

    if (frame < entryStartFrame) {
      // 1. Before highlight entry: wide broadcast pitch view
      fx = defaultFocusNumX + humanDriftX * 0.45;
      fy = defaultFocusNumY + humanDriftY * 0.45;
      scale = wideGlide;
      sx = centerPxX;
      sy = centerPxY;
      currentRadius = START_RADIUS;
      currentDarkness = 0.0;
      innerStopPct = 44;
    } else if (frame < entryLockFrame) {
      // 2. Smooth Quintic Entry Zoom & Searching Spotlight Scale-Down (760px -> 255px over 26 frames)
      const p = Math.max(0, Math.min(1, (frame - entryStartFrame) / Math.max(1, entryLockFrame - entryStartFrame)));
      const eased = smoothstep5(p);
      const searchEase = hesitantGlideEase(p);

      scale = wideGlide + (TARGET_ZOOM - wideGlide) * eased;
      fx = (defaultFocusNumX + humanDriftX * 0.45) * (1 - eased) + (firstH.xPct + humanDriftX) * eased;
      fy = (defaultFocusNumY + humanDriftY * 0.45) * (1 - eased) + (firstH.yPct + humanDriftY) * eased;
      sx = centerPxX + (firstH.xPx - centerPxX) * searchEase;
      sy = centerPxY + (firstH.yPx - centerPxY) * searchEase;
      currentRadius = START_RADIUS - (START_RADIUS - FOCUS_RADIUS) * eased;
      currentDarkness = TARGET_DARKNESS * smoothstep5(Math.min(1, p * 1.15));
      innerStopPct = 44 - (44 - 32) * eased;
    } else if (frame <= effectiveLastEnd) {
      // 3. Active Focus on players: continuous freeze breathing + hesitant physics inter-player glide
      currentRadius = FOCUS_RADIUS;
      currentDarkness = TARGET_DARKNESS;
      innerStopPct = 32;
      const baseScaleAtFrame = TARGET_ZOOM + getContinuousFreezeDrift(frame);

      let handled = false;
      for (let k = 0; k < highlightTargets.length - 1; k++) {
        const curH = highlightTargets[k];
        const nextH = highlightTargets[k + 1];

        // Hesitant physics slide window: 26..36 frames spanning across nextH.startFrame
        const maxAvail = Math.max(16, nextH.startFrame - curH.startFrame);
        const slideDur = Math.min(36, Math.max(26, Math.floor(maxAvail * 0.45)));
        const slideStart = Math.max(entryLockFrame + 4, nextH.startFrame - Math.floor(slideDur * 0.65));
        const slideEnd = Math.min(effectiveLastEnd, slideStart + slideDur);

        if (frame >= curH.startFrame && frame < slideStart) {
          scale = baseScaleAtFrame;
          fx = curH.xPct + humanDriftX;
          fy = curH.yPct + humanDriftY;
          sx = curH.xPx;
          sy = curH.yPx;
          handled = true;
          break;
        } else if (frame >= slideStart && frame <= slideEnd) {
          // Hesitant physics glide between players:
          // - Position follows hesitantGlideEase(p) (slow hesitant release -> smooth glide -> magnetic settle)
          // - Scale applies a subtle 0.025x mid-flight arc on top of continuous baseScaleAtFrame (zero reset at slideEnd+1!)
          // - Spotlight aperture breathes open (+105px: 255px -> 360px -> 255px) as it searches for the next player
          const p = Math.max(0, Math.min(1, (frame - slideStart) / Math.max(1, slideEnd - slideStart)));
          const panEase = hesitantGlideEase(p);
          const bell = Math.sin(p * Math.PI);
          const panDip = bell * 0.025;

          scale = baseScaleAtFrame - panDip;
          fx = curH.xPct + (nextH.xPct - curH.xPct) * panEase + humanDriftX;
          fy = curH.yPct + (nextH.yPct - curH.yPct) * panEase + humanDriftY;
          sx = curH.xPx + (nextH.xPx - curH.xPx) * panEase;
          sy = curH.yPx + (nextH.yPx - curH.yPx) * panEase;
          currentRadius = FOCUS_RADIUS + bell * 105;
          innerStopPct = 32 + bell * 7;
          handled = true;
          break;
        }
      }

      if (!handled) {
        scale = baseScaleAtFrame;
        fx = lastH.xPct + humanDriftX;
        fy = lastH.yPct + humanDriftY;
        sx = lastH.xPx;
        sy = lastH.yPx;
      }
    } else if (frame <= effectiveLastEnd + RAMP_OUT) {
      // 4. Monotonic C2 Zoom-Out & Spotlight Iris-Out (aligned to finish at freezeEnd with zero double-bounce)
      const p = Math.max(0, Math.min(1, (frame - effectiveLastEnd) / RAMP_OUT));
      const eased = smoothstep5(p);

      const exitScaleStart = TARGET_ZOOM + getContinuousFreezeDrift(effectiveLastEnd);

      scale = exitScaleStart - (exitScaleStart - wideGlide) * eased;
      fx = (lastH.xPct + humanDriftX) * (1 - eased) + (defaultFocusNumX + humanDriftX * 0.45) * eased;
      fy = (lastH.yPct + humanDriftY) * (1 - eased) + (defaultFocusNumY + humanDriftY * 0.45) * eased;
      sx = lastH.xPx + (centerPxX - lastH.xPx) * eased;
      sy = lastH.yPx + (centerPxY - lastH.yPx) * eased;
      currentRadius = FOCUS_RADIUS + (START_RADIUS - FOCUS_RADIUS) * eased;
      currentDarkness = TARGET_DARKNESS * (1.0 - eased);
      innerStopPct = 32 + (44 - 32) * eased;
    } else {
      // 5. Post-freeze live motion: wide broadcast view with subtle organic glide
      scale = wideGlide;
      fx = defaultFocusNumX + humanDriftX * 0.45;
      fy = defaultFocusNumY + humanDriftY * 0.45;
      sx = centerPxX;
      sy = centerPxY;
      currentRadius = START_RADIUS;
      currentDarkness = 0.0;
      innerStopPct = 44;
    }

    // Single unified scale without conflicting exitFade clamp (eliminates zoom-out -> tiny zoom-in -> zoom-out bounce)
    const finalScale = scale * punchScale;

    return {
      dynamicFocusX: fx,
      dynamicFocusY: fy,
      dynamicZoomScale: finalScale,
      spotX: sx,
      spotY: sy,
      spotlightOpacity: currentDarkness,
      spotRadiusX: currentRadius,
      spotRadiusY: currentRadius,
      spotInnerStopPct: innerStopPct,
    };
  })();

  // --- Automated Tactical Negative Keyword Scanner & Red Screen Light-Up Engine ---
  // When the commentary mentions critical tactical mistake keywords ("wrong", "dangerous", "fail", "mistake", "error", etc.),
  // lights up the screen in red with rapid rise (3f) and smooth decay (15f), accompanied by error alarm SFX and camera tremor.
  const NEGATIVE_TACTICAL_KEYWORDS = new Set([
    'wrong', 'dangerous', 'danger', 'stupid', 'fail', 'fails', 'failed', 'failing', 'failure',
    'mistake', 'mistakes', 'mistaken', 'error', 'errors', 'poor', 'poorer', 'poorly',
    'costly', 'disaster', 'disastrous', 'blunder', 'blunders', 'disjointed',
    'punished', 'punish', 'punishes', 'punishing', 'exposed', 'exposes', 'exposing', 'exposure',
    'vulnerable', 'vulnerability', 'misplaced', 'misplace', 'misplaces',
    'turnover', 'lost', 'loses', 'losing', 'struggle', 'struggles', 'struggling',
    'trapped', 'trap', 'trapping', 'caught', 'hesitant', 'hesitated', 'hesitation',
    'sloppy', 'fatal', 'unforgivable', 'collapse', 'collapses', 'collapsing', 'collapsed',
    'crumbles', 'crumbling', 'crumbled', 'broken', 'breakdown', 'fragmented', 'isolated',
  ]);

  interface TacticalMistakeEvent {
    frame: number;
    word: string;
    beat: string;
  }

  const mistakeEvents: TacticalMistakeEvent[] = (() => {
    const list: TacticalMistakeEvent[] = [];

    // Source A: Explicit visual cue tags
    for (const tag of visualCueTags) {
      if (
        tag.type === 'mistake_flash' ||
        tag.type === 'red_flash' ||
        tag.type === 'error_alarm' ||
        tag.type === 'mistake' ||
        tag.type === 'danger'
      ) {
        if (typeof tag.start_frame === 'number') {
          list.push({
            frame: tag.start_frame,
            word: tag.target || 'mistake',
            beat: tag.beat || 'cue_tag',
          });
        }
      }
    }

    // Source B: Direct tactical keyword narration scanner
    // Beat 1: Starts at frame 0, duration b1_frames
    const b1Text = beats?.beat_1_setup?.text;
    if (b1Text) {
      const words = b1Text.trim().split(/\s+/);
      words.forEach((w, idx) => {
        const clean = w.toLowerCase().replace(/[^a-z0-9]/g, '');
        if (NEGATIVE_TACTICAL_KEYWORDS.has(clean)) {
          const frac = (idx + 0.3) / Math.max(1, words.length);
          const f = Math.round(frac * b1_frames);
          list.push({ frame: f, word: clean, beat: 'beat_1_setup' });
        }
      });
    }

    // Beat 2: Starts at (b1_frames + 6), duration b2_frames
    const b2Text = beats?.beat_2_cause?.text;
    const b2AudioStart = b1_frames + 6;
    if (b2Text) {
      const words = b2Text.trim().split(/\s+/);
      words.forEach((w, idx) => {
        const clean = w.toLowerCase().replace(/[^a-z0-9]/g, '');
        if (NEGATIVE_TACTICAL_KEYWORDS.has(clean)) {
          const frac = (idx + 0.3) / Math.max(1, words.length);
          const f = b2AudioStart + Math.round(frac * b2_frames);
          list.push({ frame: f, word: clean, beat: 'beat_2_cause' });
        }
      });
    }

    // Beat 3: Starts at (K_video + freezeDuration), duration b3_frames
    const b3Text = beats?.beat_3_effect?.text;
    const b3AudioStart = K_video + freezeDuration;
    if (b3Text) {
      const words = b3Text.trim().split(/\s+/);
      words.forEach((w, idx) => {
        const clean = w.toLowerCase().replace(/[^a-z0-9]/g, '');
        if (NEGATIVE_TACTICAL_KEYWORDS.has(clean)) {
          const frac = (idx + 0.3) / Math.max(1, words.length);
          const f = b3AudioStart + Math.round(frac * b3_frames);
          list.push({ frame: f, word: clean, beat: 'beat_3_effect' });
        }
      });
    }

    // Sort chronologically and enforce a 16-frame minimum spacing to prevent audio/visual stutter
    list.sort((a, b) => a.frame - b.frame);
    const coalesced: TacticalMistakeEvent[] = [];
    for (const ev of list) {
      if (coalesced.length === 0 || ev.frame >= coalesced[coalesced.length - 1].frame + 16) {
        coalesced.push(ev);
      }
    }

    return coalesced;
  })();

  const { currentMistakeFlashIntensity, mistakeShakeX, mistakeShakeY } = (() => {
    if (mistakeEvents.length === 0) {
      return { currentMistakeFlashIntensity: 0, mistakeShakeX: 0, mistakeShakeY: 0 };
    }

    let maxIntensity = 0;
    let activeEvent: TacticalMistakeEvent | null = null;

    for (const ev of mistakeEvents) {
      if (frame >= ev.frame && frame < ev.frame + 18) {
        const dt = frame - ev.frame;
        let intensity = 0;
        if (dt < 3) {
          // Rapid 3-frame rise
          intensity = interpolate(dt, [0, 3], [0, 0.48], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          });
        } else {
          // Smooth 15-frame decay with quad easing
          intensity = interpolate(dt, [3, 18], [0.48, 0], {
            easing: Easing.out(Easing.quad),
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          });
        }
        if (intensity > maxIntensity) {
          maxIntensity = intensity;
          activeEvent = ev;
        }
      }
    }

    // Camera steadiness: Keep broadcast camera 100% rock-solid with zero artificial tremor/vibration
    const sX = 0;
    const sY = 0;

    return {
      currentMistakeFlashIntensity: maxIntensity,
      mistakeShakeX: sX,
      mistakeShakeY: sY,
    };
  })();

  const localFrame = Math.max(0, frame - delay);
  const pulse = Math.sin(localFrame / 10) * 0.15 + 1;
  const rotation = (localFrame * 1.5) % 360;

  // Audio beats resolution
  const beat1Url = resolveAudioSrc(beats?.beat_1_setup);
  const beat2Url = resolveAudioSrc(beats?.beat_2_cause);
  const beat3Url = resolveAudioSrc(beats?.beat_3_effect);

  return (
    <AbsoluteFill style={{ overflow: 'hidden' }}>
      {/* 1. Optional 3-Beat Synchronized Audio Channels */}
      {/* 100% Non-Overlapping Audio Layer */}
      {isChoreographed && beat1Url && (
        <Sequence
          from={0}
          durationInFrames={b1_frames}
        >
          <Audio src={beat1Url} volume={1.0} />
        </Sequence>
      )}
      {isChoreographed && beat2Url && (
        <Sequence
          from={Math.max(K_video, b1_frames) + 6}
          durationInFrames={b2_frames}
        >
          <Audio src={beat2Url} volume={1.0} playbackRate={1.0} />
        </Sequence>
      )}
      {isChoreographed && beat3Url && (
        <Sequence
          from={Math.max(K_video, b1_frames) + 6 + b2_frames + 6}
          durationInFrames={b3_frames}
        >
          <Audio src={beat3Url} volume={1.0} playbackRate={1.0} />
        </Sequence>
      )}

      {/* Procedural Broadcast Sound Effects */}
      {isChoreographed && (
        <>
          {/* Sub-bass cinematic drop 140Hz -> 38Hz at freeze frame - subtle background polish */}
          <Sequence from={graphicsStart} durationInFrames={12}>
            <Audio src="http://127.0.0.1:3000/assets/audio/sfx/sfx_freeze_whoosh.wav" volume={0.12} playbackRate={1.0} />
          </Sequence>
          {/* Crisp UI tick as graphics land - subtle background polish */}
          <Sequence from={graphicsStart + 3} durationInFrames={8}>
            <Audio src="http://127.0.0.1:3000/assets/audio/sfx/sfx_tactical_tick.wav" volume={0.12} playbackRate={1.0} />
          </Sequence>
          {/* Ball kick transient on pass contact at physical contact frame */}
          <Sequence from={K_video} durationInFrames={10}>
            <Audio src="http://127.0.0.1:3000/assets/audio/sfx/sfx_ball_kick.wav" volume={0.20} playbackRate={1.0} />
          </Sequence>
          {/* Climax stadium bass boom - strictly ONLY on actual goal */}
          {isActualGoal && realGoalImpactFrame !== undefined && (
            <Sequence from={realGoalImpactFrame} durationInFrames={30}>
              <Audio src="http://127.0.0.1:3000/assets/audio/sfx/sfx_goal_impact.wav" volume={0.35} playbackRate={1.0} />
            </Sequence>
          )}
          {/* Aerodynamic run chevrons whoosh with micro-ticks when player run vectors appear */}
          {resolvedVectors.some((v) => v.type === 'PLAYER_RUN_DASHED' || v.type === 'PRESSING_RUN' || v.type === 'PRESSING_ARROW') && (() => {
            const runTags = visualCueTags.filter((t) => t.type === 'run_vectors' || t.type === 'player_runs' || t.type === 'vectors');
            const runAudioStart = runTags.length > 0
              ? Math.min(...runTags.map((t) => t.start_frame ?? (graphicsStart + 6)))
              : (graphicsStart + 6);
            return (
              <Sequence from={runAudioStart} durationInFrames={14}>
                <Audio src="http://127.0.0.1:3000/assets/audio/sfx/sfx_run_chevrons.wav" volume={0.22} playbackRate={1.0} />
              </Sequence>
            );
          })()}

        </>
      )}

      {/* 2. Unified Camera Motion & Telestration Container with SplitCanvasCameraRig for dynamic zoom, smooth pan, SLOW_FOCUS cues & tactical tremor */}
      <SplitCanvasCameraRig
        speedSegments={effectiveSpeedSegments}
        focusX={dynamicFocusX}
        focusY={dynamicFocusY}
        baseScale={dynamicZoomScale}
        shakeX={mistakeShakeX}
        shakeY={mistakeShakeY}
      >
        {/* Layer 0: Integrated Video Playback with Keyframe Freeze Choreography */}
        {videoUrl && (
          <AbsoluteFill>
            {isPlayerDataActive ? (
              /* Hard freeze lock: while player stats or heatmap modals are on screen, match video must remain locked to tactical freeze frame */
              freezeFrameUrl ? (
                <Img
                  src={freezeFrameUrl}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                  }}
                />
              ) : (
                <Freeze frame={0}>
                  <OffthreadVideo
                    src={videoUrl}
                    startFrom={K_video}
                    volume={0}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                </Freeze>
              )
            ) : effectiveIsWarped ? (
              /* Pre-Warped Video Pipeline: 1.0x native playback with pre-baked elastic ramps */
              <>
                <OffthreadVideo
                  src={effectiveWarpedVideoUrl || videoUrl}
                  startFrom={0}
                  playbackRate={1.0}
                  volume={(f) => {
                    const isSpeaking = (beat1Url && f >= 0 && f < b1_frames + 4) ||
                                       (beat2Url && f >= b2Start && f < b2End + 4) ||
                                       (beat3Url && f >= b3Start && f < b3End + 4);
                    return isSpeaking ? 0.055 : 0.15; // -25dB ducking during voiceover, -16.5dB during live play
                  }}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
                {freezeFrameUrl && (
                  <Sequence from={K_video} durationInFrames={freezeDuration}>
                    <Img
                      src={freezeFrameUrl}
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                      }}
                    />
                  </Sequence>
                )}
              </>
            ) : isChoreographed ? (
              <>
                {/* Sequence 1 (Pre-Freeze): from={0}, durationInFrames={K_video} */}
                {K_video > 0 && (
                  <Sequence from={0} durationInFrames={K_video}>
                    <OffthreadVideo
                      src={videoUrl}
                      startFrom={0}
                      playbackRate={1.0}
                      volume={(f) => {
                        // Check if Beat 1 AI voiceover is actively speaking
                        const isSpeaking = beat1Url && f < (b1_frames + 4);
                        if (isSpeaking) {
                          // Duck broadcast commentary down to -25dB (0.055) so AI voiceover is crystal clear
                          return 0.055;
                        }
                        // Narrator has paused before freeze: ramp up to 0.16 for stadium atmosphere, then decay into freeze
                        if (K_video <= 12) return 0.055;
                        if (f >= K_video - 12) {
                          return Math.max(0, 0.16 - ((f - (K_video - 12)) / 12) * 0.16);
                        }
                        const postBeatFrames = f - (b1_frames + 4);
                        if (postBeatFrames < 6) {
                          return 0.055 + (postBeatFrames / 6) * 0.105;
                        }
                        return 0.16;
                      }}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  </Sequence>
                )}

                {/* Sequence 2 (Freeze): from={K_video}, durationInFrames={freezeDuration} */}
                <Sequence from={K_video} durationInFrames={freezeDuration}>
                  {freezeFrameUrl ? (
                    <Img
                      src={freezeFrameUrl}
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                      }}
                    />
                  ) : (
                    <Freeze frame={0}>
                      <OffthreadVideo
                        src={videoUrl}
                        startFrom={K_video}
                        volume={0}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    </Freeze>
                  )}
                </Sequence>

                {/* Sequence 3 (Post-Freeze): from={K_video + freezeDuration}, durationInFrames={postRollFrames} */}
                <Sequence from={K_video + freezeDuration} durationInFrames={postRollFrames}>
                  <OffthreadVideo
                    src={videoUrl}
                    startFrom={K_video + 1}
                    playbackRate={1.0}
                    volume={(f) => {
                      const globalFrame = K_video + freezeDuration + f;
                      const isSpeaking = beat3Url && globalFrame < (b3End + 4);
                      if (isSpeaking) {
                        return 0.055; // -25dB ducking during Beat 3 narration
                      }
                      const postSpeechFrames = globalFrame - (b3End + 4);
                      if (postSpeechFrames < 8) {
                        return 0.055 + (postSpeechFrames / 8) * 0.065;
                      }
                      return 0.12; // Un-ducked live stadium atmosphere
                    }}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                </Sequence>

                {/* Screen Flash on Goal Impact - strictly ONLY on actual goal */}
                {isActualGoal && flashOpacity > 0 && (
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      backgroundColor: `rgba(255, 255, 255, ${flashOpacity})`,
                      pointerEvents: 'none',
                    }}
                  />
                )}

                {/* Goal Net Neon Highlight Pulse on Climax Impact - strictly ONLY on actual goal */}
                {isActualGoal && netPulseOpacity > 0 && (
                  <div
                    style={{
                      position: 'absolute',
                      top: '32%',
                      right: '5%',
                      width: '22%',
                      height: '30%',
                      background:
                        'radial-gradient(ellipse at center, rgba(16, 185, 129, 0.65) 0%, rgba(16, 185, 129, 0) 70%)',
                      mixBlendMode: 'screen',
                      opacity: netPulseOpacity,
                      pointerEvents: 'none',
                    }}
                  />
                )}
              </>
            ) : (
              <OffthreadVideo
                src={videoUrl}
                volume={(f) => {
                  const isSpeaking = (beat1Url && f >= 0 && f < b1_frames + 4) ||
                                     (beat2Url && f >= b2Start && f < b2End + 4) ||
                                     (beat3Url && f >= b3Start && f < b3End + 4);
                  return isSpeaking ? 0.055 : 0.14; // -25dB ducking during voiceover
                }}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            )}
          </AbsoluteFill>
        )}

        {/* Layer 0.5: Broadcast Tactical Dynamic Player Spotlight Vignette (Scales down & searches for player) */}
        {!renderBadgesOnly && effectiveIsTactical && !isCloseUp && spotlightOpacity > 0.001 && (
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width,
              height,
              pointerEvents: 'none',
              zIndex: 1,
              opacity: spotlightOpacity,
              background: `radial-gradient(circle ${spotRadiusX.toFixed(1)}px at ${spotX.toFixed(1)}px ${spotY.toFixed(1)}px, rgba(0, 4, 10, 0) 0%, rgba(255, 255, 255, 0.05) ${Math.max(18, spotInnerStopPct - 4).toFixed(1)}%, rgba(0, 4, 10, 0.04) ${spotInnerStopPct.toFixed(1)}%, rgba(0, 4, 10, 0.32) ${(spotInnerStopPct + 18).toFixed(1)}%, rgba(0, 4, 10, 0.72) ${(spotInnerStopPct + 40).toFixed(1)}%, rgba(0, 4, 10, 0.90) 94%, rgba(0, 4, 10, 0.90) 100%)`,
            }}
          />
        )}

        {/* Layer 1: Ground Telestration SVG Graphics - Suppressed on non-tactical shots (closeups/coaches) */}
        {!renderBadgesOnly && effectiveIsTactical && (
          <svg
            width={width}
            height={height}
            style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
          >
            <defs>
              <pattern
                id="tacticalStripes"
                width="18"
                height="18"
                patternTransform="rotate(35 0 0)"
                patternUnits="userSpaceOnUse"
              >
                <line
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="18"
                  stroke="rgba(245, 158, 11, 0.40)"
                  strokeWidth="3"
                />
              </pattern>

              <pattern
                id="tacticalPurpleStripes"
                width="16"
                height="16"
                patternTransform="rotate(45 0 0)"
                patternUnits="userSpaceOnUse"
              >
                <line
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="16"
                  stroke="rgba(168, 85, 247, 0.4)"
                  strokeWidth="4"
                />
              </pattern>

              <pattern
                id="tacticalCyanStripes"
                width="16"
                height="16"
                patternTransform="rotate(45 0 0)"
                patternUnits="userSpaceOnUse"
              >
                <line
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="16"
                  stroke="rgba(56, 189, 248, 0.40)"
                  strokeWidth="3.5"
                />
              </pattern>

              <marker id="whiteArrowHead" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
                <polygon points="0 1, 8 3.5, 0 6" fill="#ffffff" />
              </marker>

              <marker id="redArrowHead" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
                <polygon points="0 1, 8 3.5, 0 6" fill="#ffffff" />
              </marker>

              <marker id="cyanArrowHead" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
                <polygon points="0 1, 8 3.5, 0 6" fill="#00e5ff" />
              </marker>

              <marker id="solidArrowHead" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
                <polygon points="0 1, 8 3.5, 0 6" fill="#ffffff" />
              </marker>

              <marker id="dashedArrowHead" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
                <polygon points="0 1, 8 3.5, 0 6" fill="#ffffff" />
              </marker>

              <marker id="amberArrowHead" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
                <polygon points="0 1, 8 3.5, 0 6" fill="#fbbf24" />
              </marker>

              <filter id="tacticalGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Step 3: Exploit Zone Polygons Deleted per Minimalist UI Redesign */}

            {/* Other Tactical Polygons (Strictly Open Polylines unless Strict Pressing Trap) */}
            {staticPolygonsOpacity > 0 &&
              !isSetPieceOrLongBall &&
              combinedPolygons
                .filter(
                  (poly) =>
                    !poly.isVacatedZone &&
                    poly.id !== 'cause-vacated-zone-poly' &&
                    poly.id !== 'cause-triangle-poly' &&
                    !poly.borderColor?.includes('#a855f7') &&
                    !poly.fillColor?.includes('168, 85, 247')
                )
                .map((poly) => {
                  let coords: { x: number; y: number }[] = [];
                  if (poly.points && poly.points.length >= 3) {
                    coords = poly.points.map((pt) => ({
                      x: pt.x <= 100 ? (pt.x / 100) * width : pt.x,
                      y: pt.y <= 100 ? (pt.y / 100) * height : pt.y,
                    }));
                  } else if (poly.playerIds && poly.playerIds.length >= 3) {
                    coords = poly.playerIds
                      .map((id) => playerMap.get(id))
                      .filter((c): c is { x: number; y: number; p: TelestrationPlayer } => !!c);
                  }

                  if (coords.length < 2) return null;

                  const catPt = catalystId ? playerMap.get(catalystId) : null;
                  const isClosedTrap = coords.length === 3 && isValidPressingTrap(coords, catPt, 24.0, 6.0, width, height);
                  const borderColor = poly.borderColor || '#00e676';

                  // Strictly open shortest-path chains - NEVER render closed polygons
                  const openTrapChain = computeShortestOpenChain(coords, 28.0, width, height);
                  if (openTrapChain.length >= 2) {
                    const pathDTrap = buildPaddedChainPath(openTrapChain, 22);
                    return (
                      <g key={poly.id} style={{ opacity: staticPolygonsOpacity }}>
                        <path
                          d={pathDTrap}
                          fill="none"
                          stroke={borderColor}
                          strokeWidth={2.8}
                          strokeDasharray="6 3"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          filter="url(#tacticalGlow)"
                        />
                      </g>
                    );
                  }

                  // Non-trap: render clean open shortest-path polyline (never a closed polygon)
                  const openChain = computeShortestOpenChain(coords, 28.0, width, height);
                  if (openChain.length < 2) return null;
                  const pathD = buildPaddedChainPath(openChain, 22);

                  return (
                    <g key={poly.id} style={{ opacity: staticPolygonsOpacity }}>
                      <path
                        d={pathD}
                        fill="none"
                        stroke={borderColor}
                        strokeWidth={2.8}
                        strokeDasharray="6 3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        filter="url(#tacticalGlow)"
                      />
                    </g>
                  );
                })}

            {/* Step 2: Contextual Tactical Units (Backline Chain, Midfield Chain, Pressing Unit & Passing Triangle) */}
            {showStructure && !isSetPieceOrLongBall && (() => {
              type StructCoord = { x: number; y: number; p: TelestrationPlayer };
              const currentOpacity = structOpacity;

              // Team identities
              const defTeam = isAttackingHome ? 'away' : 'home';
              const attTeam = isAttackingHome ? 'home' : 'away';

              // Helper: safely resolve coords for track IDs belonging ONLY to a specific team (strictly excluding referee)
              const resolveTeamCoords = (ids: (string | number)[] | undefined, requiredTeam: string): StructCoord[] => {
                if (!ids || ids.length === 0) return [];
                const resolved: StructCoord[] = [];
                const seenIds = new Set<string>();
                for (const rawId of ids) {
                  const idStr = String(rawId);
                  if (seenIds.has(idStr)) continue;
                  seenIds.add(idStr);
                  const entry = playerMap.get(idStr);
                  const isRef = entry?.p && (entry.p.role === 'REFEREE' || entry.p.name === 'REFEREE' || (entry.p as any).jersey_class === 'REFEREE' || entry.p.team === 'referee');
                  if (entry && entry.p && !isRef && (entry.p.team === requiredTeam || (!entry.p.team && defTeam === 'home'))) {
                    resolved.push(entry);
                  }
                }
                return resolved;
              };

              // A) Defensive Backline Unit (Open polyline chain sorted laterally across pitch width)
              const rawBacklineIds = causeTel?.backline_track_ids || structuralNetwork?.track_ids || [];
              const backlineCoords = rawBacklineIds.length >= 2 ? resolveTeamCoords(rawBacklineIds, defTeam) : [];

              // Compute Shortest Open Path (MST / Nearest-Neighbor Path rejecting links > 26% screen width)
              const sortedBackline = computeShortestOpenChain(backlineCoords, 26.0, width, height);
              const backlineD = sortedBackline.length >= 2
                ? buildPaddedChainPath(sortedBackline, 22)
                : '';

              // B) Defensive Midfield Line (Separate open polyline situated ahead of backline)
              const rawMidfieldIds = causeTel?.midfield_track_ids || [];
              const midfieldCoords = resolveTeamCoords(rawMidfieldIds, defTeam);
              const sortedMidfield = computeShortestOpenChain(midfieldCoords, 26.0, width, height);
              const midfieldD = sortedMidfield.length >= 2
                ? buildPaddedChainPath(sortedMidfield, 22)
                : '';

              // C) Local Pressing Unit (Pressing Triangle or Open Chain + Short Pressing Arrows)
              const rawPresserIds = causeTel?.pressing_track_ids || [];
              let presserCoords = resolveTeamCoords(rawPresserIds, defTeam);

              // Fallback for pressers: ONLY if pressing_track_ids were explicitly specified but failed to resolve
              if (rawPresserIds.length >= 2 && presserCoords.length < 2 && catalystId && playerMap.has(catalystId)) {
                const cPt = playerMap.get(catalystId)!;
                const closeDefs: StructCoord[] = [];
                playerMap.forEach((entry, id) => {
                  if (id !== catalystId && entry.p.team === defTeam && entry.y >= 0.22 * height) {
                    closeDefs.push(entry);
                  }
                });
                closeDefs.sort((a, b) => Math.hypot(a.x - cPt.x, a.y - cPt.y) - Math.hypot(b.x - cPt.x, b.y - cPt.y));
                presserCoords = closeDefs.slice(0, 3);
              }

              const catPt = catalystId ? playerMap.get(catalystId) : null;
              const isClosedTrap = isValidPressingTrap(presserCoords, catPt, 24.0, 6.0, width, height);
              const sortedPressers = computeShortestOpenChain(presserCoords, 26.0, width, height);
              const presserD = sortedPressers.length >= 2
                ? buildPaddedChainPath(sortedPressers, 22, false, catPt, 32)
                : '';

              // D) Attacking Passing / Buildup Unit (Carrier + passing options in attJerseyColor)
              // Only render when triangle_track_ids are explicitly provided and non-empty (never synthesize fake wide triangles)
              const rawTriIds = causeTel?.triangle_track_ids || [];
              const triCoords = resolveTeamCoords(rawTriIds, attTeam);
              const sortedTri = computeShortestOpenChain(triCoords, 26.0, width, height);
              const triD = sortedTri.length >= 2
                ? buildPaddedChainPath(sortedTri, 22)
                : '';

              // Derive structure timing for player tag spring pop & linking line draw animation
              const structStartFrame = graphicsStart + Math.round(graphicsDuration * structThreshold);
              const structLocalFrame = Math.max(0, frame - structStartFrame);

              // Helper 1: Professional animated player tag (elastic spring scale-up, radar ripple, and center pip snap)
              const renderTacticalPlayerTag = ({
                id,
                x,
                y,
                rx,
                ry,
                color,
                localFrame,
                openDuration = 12,
                exitFade = freezeExitFade,
              }: {
                id: string;
                x: number;
                y: number;
                rx: number;
                ry: number;
                color: string;
                localFrame: number;
                openDuration?: number;
                exitFade?: number;
              }) => {
                if (localFrame < 0 || exitFade <= 0) return null;

                const tagRatio = Math.min(1.0, localFrame / openDuration);
                // Elastic spring scale: 0 -> 1.16 overshoot at t=0.65 -> settles to 1.0
                const tagScale = tagRatio >= 1.0
                  ? 1.0
                  : (tagRatio <= 0.0 ? 0.0 : Math.sin(tagRatio * Math.PI * 0.5) * (1.0 + Math.sin(tagRatio * Math.PI) * 0.18));
                const tagOpacity = Math.min(1.0, tagRatio * 1.6) * exitFade;

                if (tagOpacity <= 0.01 || tagScale <= 0.01) return null;

                // Radar ripple on opening
                const rippleScale = 0.5 + tagRatio * 0.85;
                const rippleOpacity = (1.0 - tagRatio) * 0.55 * exitFade;

                return (
                  <g key={`player-tag-${id}`} transform={`translate(${x}, ${y}) scale(${tagScale})`} style={{ opacity: tagOpacity }}>
                    {/* Perspective turf drop shadow */}
                    <ellipse
                      cx={0}
                      cy={2.5}
                      rx={rx + 1.5}
                      ry={ry + 1.0}
                      fill="rgba(0, 0, 0, 0.45)"
                    />

                    {/* Broadcast Ground Ring (screenshot 1) */}
                    <ellipse
                      cx={0}
                      cy={0}
                      rx={rx}
                      ry={ry}
                      fill={color === '#ffffff' ? 'rgba(255, 255, 255, 0.22)' : 'rgba(225, 29, 72, 0.28)'}
                      stroke={color}
                      strokeWidth={2.0}
                      filter="url(#tacticalGlow)"
                    />

                    {/* Inner subtle dashed concentric ring */}
                    <ellipse
                      cx={0}
                      cy={0}
                      rx={rx * 0.76}
                      ry={ry * 0.76}
                      fill="none"
                      stroke={color}
                      strokeWidth={1.2}
                      strokeDasharray="4 2"
                    />

                    {/* Opening Radar Acquisition Ripple */}
                    {tagRatio < 1.0 && rippleOpacity > 0.02 && (
                      <ellipse
                        cx={0}
                        cy={0}
                        rx={rx * rippleScale}
                        ry={ry * rippleScale}
                        fill="none"
                        stroke={color}
                        strokeWidth={1.2}
                        strokeOpacity={rippleOpacity}
                        filter="url(#tacticalGlow)"
                      />
                    )}

                    {/* Center Target Pip 'o' (screenshot 1) */}
                    <circle
                      cx={0}
                      cy={0}
                      r={3.2 * pulse}
                      fill={color}
                      stroke="#ffffff"
                      strokeWidth={1.5}
                      filter="url(#tacticalGlow)"
                    />
                  </g>
                );
              };

              // Helper 2: Link animation drawing lines between players instead of popping
              const renderLinkingLine = ({
                id,
                pathD,
                color,
                strokeWidth = 3.2,
                strokeDasharray = '8 4',
                shadowWidth = 6.5,
                ribbonWidth = 8.0,
                localFrame,
                startDelay,
                drawDuration = 18,
                exitFade = freezeExitFade,
              }: {
                id: string;
                pathD: string;
                color: string;
                strokeWidth?: number;
                strokeDasharray?: string;
                shadowWidth?: number;
                ribbonWidth?: number;
                localFrame: number;
                startDelay: number;
                drawDuration?: number;
                exitFade?: number;
              }) => {
                if (!pathD || exitFade <= 0) return null;

                const drawFrame = Math.max(0, localFrame - startDelay);
                const drawRatio = Math.min(1.0, drawFrame / drawDuration);
                if (drawRatio <= 0.0) return null;

                // Smooth broadcast deceleration ease-out
                const drawEase = easeVector(drawRatio);
                const lineOpacity = Math.min(1.0, drawRatio * 1.5) * exitFade;
                const maskId = `link-path-mask-${id}`;

                return (
                  <g key={`linking-line-group-${id}`} style={{ opacity: lineOpacity }}>
                    <defs>
                      <mask id={maskId}>
                        <path
                          d={pathD}
                          fill="none"
                          stroke="#ffffff"
                          strokeWidth={ribbonWidth + 10}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          pathLength="100"
                          strokeDasharray="100"
                          strokeDashoffset={100 * (1 - drawEase)}
                        />
                      </mask>
                    </defs>

                    <g mask={`url(#${maskId})`}>
                      {/* 1. 3D Turf Contact Shadow */}
                      {shadowWidth > 0 && (
                        <path
                          d={pathD}
                          transform="translate(0, 3.5)"
                          fill="none"
                          stroke="rgba(0, 0, 0, 0.55)"
                          strokeWidth={shadowWidth}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      )}

                      {/* 2. 3D Perspective Turf Base Ribbon */}
                      {ribbonWidth > 0 && (
                        <path
                          d={pathD}
                          fill="none"
                          stroke={color}
                          strokeWidth={ribbonWidth}
                          strokeOpacity={0.26}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      )}

                      {/* 3. Core Tactical Line (Screenshot 2) */}
                      <path
                        d={pathD}
                        fill="none"
                        stroke={color}
                        strokeWidth={strokeWidth}
                        strokeDasharray={strokeDasharray}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        filter="url(#tacticalGlow)"
                      />
                    </g>
                  </g>
                );
              };

              return (
                <g key="contextual-tactical-units" style={{ opacity: currentOpacity }}>
                  {/* 1. Defensive Backline Unit: Animated Tags + Sequenced Linking Line */}
                  {sortedBackline.length >= 2 && (
                    <g key="defensive-backline-unit">
                      {/* Sequenced Linking Line (draws across after player tags finish opening) */}
                      {renderLinkingLine({
                        id: 'backline-chain',
                        pathD: backlineD,
                        color: defJerseyColor,
                        strokeWidth: 3.2,
                        strokeDasharray: '8 4',
                        shadowWidth: 6.5,
                        ribbonWidth: 8.0,
                        localFrame: structLocalFrame,
                        startDelay: Math.max(12, (sortedBackline.length - 1) * 3.5 + 8),
                        drawDuration: 18,
                      })}

                      {/* Clean vertex anchor nodes on backline line (broadcast style, eliminating redundant ground rings) */}
                      {sortedBackline.map((c, i) => {
                        const tagLocalFrame = Math.max(0, structLocalFrame - i * 3.5);
                        if (tagLocalFrame < 0 || freezeExitFade <= 0) return null;
                        return (
                          <circle
                            key={`backline-dot-${c.p.id}-${i}`}
                            cx={c.x}
                            cy={c.y}
                            r={3.8}
                            fill={defJerseyColor}
                            stroke="#ffffff"
                            strokeWidth={1.5}
                            filter="url(#tacticalGlow)"
                            style={{ opacity: freezeExitFade }}
                          />
                        );
                      })}
                    </g>
                  )}

                  {/* 2. Defensive Midfield Unit: Animated Tags + Sequenced Linking Line */}
                  {sortedMidfield.length >= 2 && (
                    <g key="defensive-midfield-unit">
                      {/* Sequenced Linking Line */}
                      {renderLinkingLine({
                        id: 'midfield-chain',
                        pathD: midfieldD,
                        color: defJerseyColor,
                        strokeWidth: 3.0,
                        strokeDasharray: '6 3',
                        shadowWidth: 5.5,
                        ribbonWidth: 7.0,
                        localFrame: structLocalFrame,
                        startDelay: Math.max(14, (sortedMidfield.length - 1) * 3.5 + 10),
                        drawDuration: 18,
                      })}

                      {/* Clean vertex anchor nodes on midfield line */}
                      {sortedMidfield.map((c, i) => {
                        const tagLocalFrame = Math.max(0, structLocalFrame - (i + 1) * 3.5);
                        if (tagLocalFrame < 0 || freezeExitFade <= 0) return null;
                        return (
                          <circle
                            key={`midfield-dot-${c.p.id}-${i}`}
                            cx={c.x}
                            cy={c.y}
                            r={3.6}
                            fill={defJerseyColor}
                            stroke="#ffffff"
                            strokeWidth={1.5}
                            filter="url(#tacticalGlow)"
                            style={{ opacity: freezeExitFade }}
                          />
                        );
                      })}
                    </g>
                  )}

                  {/* 3. Pressing Action Unit & Short Pressing Arrows */}
                  {presserCoords.length >= 2 && (() => {
                    return (
                      <g key="pressing-action-unit">
                        {sortedPressers.length >= 2 &&
                          renderLinkingLine({
                            id: 'presser-chain',
                            pathD: presserD,
                            color: defJerseyColor,
                            strokeWidth: 2.8,
                            strokeDasharray: '6 3',
                            shadowWidth: 0,
                            ribbonWidth: 0,
                            localFrame: structLocalFrame,
                            startDelay: Math.max(10, (sortedPressers.length - 1) * 3.5 + 8),
                            drawDuration: 16,
                          })}

                        {/* Clean Pressing Marker for primary presser only */}
                        {sortedPressers.slice(0, 1).map((c, i) => {
                          const tagLocalFrame = Math.max(0, structLocalFrame - i * 3.5);
                          return renderTacticalPlayerTag({
                            id: `presser-${c.p.id}-${i}`,
                            x: c.x,
                            y: c.y,
                            rx: 20,
                            ry: 7,
                            color: defJerseyColor,
                            localFrame: tagLocalFrame,
                            openDuration: 12,
                          });
                        })}

                        {/* Modern Aerodynamic Pressing Chevrons (New UI) pointing toward ball carrier */}
                        {catPt &&
                          sortedPressers.slice(0, 1).map((pr, idx) => {
                            const prId = String(pr.p.id);
                            // Avoid duplicate: if this presser already has a PRESSING_RUN / PRESSING_ARROW vector, skip!
                            if (vectorPresserIds.has(prId)) {
                              return null;
                            }

                            const dx = catPt.x - pr.x;
                            const dy = catPt.y - pr.y;
                            const dist = Math.hypot(dx, dy) || 1;
                            const arrowLen = Math.min(95, Math.max(55, dist * 0.48));
                            const toX = pr.x + (dx / dist) * arrowLen;
                            const toY = pr.y + (dy / dist) * arrowLen;

                            // Suppress head-on opposing pressing arrows that clash with any pass vector
                            let isClash = false;
                            for (const pv of primaryPassVectors) {
                              const pdx = pv.x2 - pv.x1;
                              const pdy = pv.y2 - pv.y1;
                              const pLen = Math.hypot(pdx, pdy);
                              if (pLen < 1e-3) continue;

                              const dToTarget = Math.hypot(pr.x - pv.x2, pr.y - pv.y2);
                              if (dToTarget < 75) {
                                isClash = true;
                                break;
                              }

                              const dot = (pdx / pLen) * (dx / dist) + (pdy / pLen) * (dy / dist);
                              const pnx = -pdy / pLen;
                              const pny = pdx / pLen;
                              const perpDist = Math.abs((pr.x - pv.x1) * pnx + (pr.y - pv.y1) * pny);
                              if (dot <= -0.65 && perpDist < 35) {
                                isClash = true;
                                break;
                              }
                            }
                            if (isClash) {
                              return null;
                            }

                            // Chevron delay: starts after presser tag has opened
                            const arrowDelay = Math.max(10, idx * 3.5 + 8);
                            const arrowFrame = Math.max(0, structLocalFrame - arrowDelay);
                            const arrowRatio = Math.min(1.0, arrowFrame / 14);
                            if (arrowRatio <= 0.0) return null;
                            const arrowScale = 1 - Math.pow(1 - arrowRatio, 3);

                            // Apply foot halo padding so chevrons start just outside presser's ground ring
                            const [paddedStartX, paddedStartY, paddedToX, paddedToY] = applyFootHaloPadding(pr.x, pr.y, toX, toY, 18);
                            const pressVectorColor = defJerseyColor || '#ef4444';

                            return renderRunChevrons(
                              `press-chevrons-${pr.p.id}-${idx}`,
                              paddedStartX,
                              paddedStartY,
                              paddedToX,
                              paddedToY,
                              arrowScale,
                              Math.min(1.0, arrowRatio * 1.5) * freezeExitFade,
                              pressVectorColor,
                              height
                            );
                          })}
                      </g>
                    );
                  })()}

                  {/* 4. Attacking Passing / Buildup Unit: Animated Tags + Sequenced Linking Line */}
                  {sortedTri.length >= 2 && (() => {
                    return (
                      <g key="passing-triangle-unit">
                        {renderLinkingLine({
                          id: 'tri-chain',
                          pathD: triD,
                          color: attJerseyColor,
                          strokeWidth: 2.5,
                          strokeDasharray: '6 4',
                          shadowWidth: 0,
                          ribbonWidth: 0,
                          localFrame: structLocalFrame,
                          startDelay: Math.max(10, (sortedTri.length - 1) * 3.5 + 8),
                          drawDuration: 16,
                        })}
                        {sortedTri.map((c, i) => {
                          const isCat = String(c.p.id) === catalystId;
                          if (isCat) return null; // Carrier already has primary blue ring & beam
                          const tagLocalFrame = Math.max(0, structLocalFrame - i * 3.5);
                          return renderTacticalPlayerTag({
                            id: `tri-${c.p.id}-${i}`,
                            x: c.x,
                            y: c.y,
                            rx: 22,
                            ry: 7.5,
                            color: attJerseyColor,
                            localFrame: tagLocalFrame,
                            openDuration: 12,
                          });
                        })}
                      </g>
                    );
                  })()}
                </g>
              );
            })()}

            {/* Step 2b: Explicit Telestration Lines (Tactical Structure, Curved Ball Pass Bezier, Player Run Dotted) */}
            {showStructure && combinedLines.length > 0 && (
              <g key="explicit-tactical-lines" style={{ opacity: structOpacity }}>
                {combinedLines.map((line) => {
                  const pts: { x: number; y: number; p?: TelestrationPlayer }[] = (line.playerIds || [])
                    .map((id) => playerMap.get(String(id)))
                    .filter((c): c is { x: number; y: number; p: TelestrationPlayer } => !!c);

                  if (pts.length < 2) return null;

                  const lineStyle = line.style || (line.dashed ? 'dashed' : 'solid');
                  const lineType = line.lineType || line.line_type || 'tactical_structure';
                  const paddingPx = line.haloPaddingPx ?? line.halo_padding_px ?? 22;
                  const strokeWidth = line.width || 3.5;
                  const firstP = pts[0]?.p;
                  const defaultLineColor = lineType === 'ball_pass'
                    ? '#fbbf24'
                    : (firstP?.team === 'away' ? awayColor : (firstP?.team === 'home' ? homeColor : attJerseyColor));
                  const lineColor = (line.color && line.color !== '#00e676' && line.color !== '#00e5ff')
                    ? line.color
                    : defaultLineColor;

                  if (lineStyle === 'bezier' || lineType === 'ball_pass') {
                    // Curved Bezier pass trajectory
                    const p1 = pts[0];
                    const p2 = pts[pts.length - 1];
                    const [px1, py1, px2, py2] = applyFootHaloPadding(p1.x, p1.y, p2.x, p2.y, paddingPx);

                    let cx = (px1 + px2) / 2;
                    let cy = Math.min(py1, py2) - 0.14 * height;
                    const bCtrl = line.bezierControl || line.bezier_control;
                    if (bCtrl) {
                      if (Array.isArray(bCtrl)) {
                        cx = bCtrl[0] <= 100 ? (bCtrl[0] / 100) * width : bCtrl[0];
                        cy = bCtrl[1] <= 100 ? (bCtrl[1] / 100) * height : bCtrl[1];
                      } else {
                        cx = bCtrl.x <= 100 ? (bCtrl.x / 100) * width : bCtrl.x;
                        cy = bCtrl.y <= 100 ? (bCtrl.y / 100) * height : bCtrl.y;
                      }
                    }

                    const easedArc = easeVector(structProgress);
                    const arcOffset = (1 - easedArc) * 100;

                    return (
                      <g key={line.id}>
                        <path
                          d={`M ${px1} ${py1} Q ${cx} ${cy} ${px2} ${py2}`}
                          fill="none"
                          stroke={lineColor}
                          strokeWidth={strokeWidth}
                          strokeLinecap="round"
                          pathLength={100}
                          strokeDasharray="100"
                          strokeDashoffset={arcOffset}
                          markerEnd="url(#amberArrowHead)"
                          filter={lineType === 'ball_pass' || lineColor === '#ffffff' ? undefined : "url(#tacticalGlow)"}
                        />
                      </g>
                    );
                  }

                  if (lineStyle === 'dotted' || lineType === 'player_run') {
                    // Suppress duplicate dashed line if a vector already renders run chevrons for this player
                    const hasVectorChevron = resolvedVectors.some(
                      (v) => (v.type === 'PLAYER_RUN_DASHED' || v.type === 'PRESSING_RUN') &&
                        line.playerIds?.some((pId) => String(v.from_track_id) === String(pId))
                    );
                    if (hasVectorChevron) {
                      return null;
                    }

                    // Modern broadcast player run chevrons (New UI)
                    const p1 = pts[0];
                    const p2 = pts[pts.length - 1];
                    const [px1, py1, px2, py2] = applyFootHaloPadding(p1.x, p1.y, p2.x, p2.y, paddingPx);

                    return renderRunChevrons(
                      `line-run-${line.id}`,
                      px1,
                      py1,
                      px2,
                      py2,
                      1.0,
                      freezeExitFade,
                      lineColor,
                      height
                    );
                  }

                  // Tactical structure (solid or dashed)
                  const orderedChain = pts.length > 2
                    ? computeShortestOpenChain(pts, 28.0, width, height)
                    : pts;
                  const pathD = buildPaddedChainPath(orderedChain, paddingPx);

                  return (
                    <g key={line.id}>
                      <path
                        d={pathD}
                        fill="none"
                        stroke={lineColor}
                        strokeWidth={strokeWidth}
                        strokeDasharray={lineStyle === 'dashed' ? '8 4' : undefined}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        filter="url(#tacticalGlow)"
                      />
                    </g>
                  );
                })}
              </g>
            )}

            {/* Step 4: Vectors (Ground Pass Solid, Player Run Dashed, and Aerial Ball Arc with FIFA Landing Target Cross) */}
            {showVectors && resolvedVectors.length > 0 && (() => {
              const currentOpacity = vectorOpacity;
              let renderedPassOptionCount = 0;

              return (
                <g key="tactical-vectors-layer" style={{ opacity: currentOpacity }}>
                  {resolvedVectors.map((v, idx) => {
                    if (v.type === 'PASS_OPTION_DASHED' || (v.type === 'GROUND_PASS_SOLID' && v.was_taken === false)) {
                      // Candidate pass options: strictly mutually exclusive with executed pass!
                      if (optOpacity <= 0.01) return null;

                      const fromP = (v.from_track_id && playerMap.has(String(v.from_track_id)))
                        ? playerMap.get(String(v.from_track_id))
                        : (catalystId && playerMap.has(catalystId) ? playerMap.get(catalystId) : null);

                      const x1 = fromP
                        ? fromP.x
                        : (v.from_pct ? ((Array.isArray(v.from_pct) ? v.from_pct[0] : (v.from_pct as any).x) / 100) * width : (catalyst ? (catalyst.x_pct / 100) * width : 0.5 * width));
                      const y1 = fromP
                        ? fromP.y
                        : (v.from_pct ? ((Array.isArray(v.from_pct) ? v.from_pct[1] : (v.from_pct as any).y) / 100) * height : (catalyst ? (catalyst.y_pct / 100) * height : 0.5 * height));
                      const toP = v.to_track_id ? playerMap.get(String(v.to_track_id)) : null;
                      const targetX = toP
                        ? toP.x
                        : (v.to_pct ? ((Array.isArray(v.to_pct) ? v.to_pct[0] : v.to_pct.x) / 100) * width : x1 + 0.15 * width);
                      const targetY = toP
                        ? toP.y
                        : (v.to_pct ? ((Array.isArray(v.to_pct) ? v.to_pct[1] : v.to_pct.y) / 100) * height : y1);

                      const fullX2 = Math.min(0.95 * width, Math.max(0.05 * width, targetX));
                      const fullY2 = Math.min(0.85 * height, Math.max(0.20 * height, targetY));

                      // Tactical Line-of-Sight & Cover Interception Engine:
                      const carrierTeam = fromP.p?.team || (attTeam as string) || 'home';
                      const laneCheck = isPassLaneBlockedOrCovered(x1, y1, fullX2, fullY2, carrierTeam);
                      // Render option arrow clearly; if congested, the dashed styling conveys the option channel

                      // Cap candidate passing options to at most 2 viable open options
                      if (renderedPassOptionCount >= 2) {
                        return null;
                      }
                      renderedPassOptionCount++;

                      const easedOptProgress = easeVector(optFadeIn);
                      const [paddedX1, paddedY1, paddedFullX2, paddedFullY2] = applyFootHaloPadding(x1, y1, fullX2, fullY2, 22);
                      const curX2 = paddedX1 + (paddedFullX2 - paddedX1) * easedOptProgress;
                      const curY2 = paddedY1 + (paddedFullY2 - paddedY1) * easedOptProgress;

                      // Jersey Kit-Adaptive Colors:
                      const kitPassColor = getTeamKitPassColor(carrierTeam);
                      const optionColor = (v.color && v.color !== '#00e5ff' && v.color !== '#ffffff' && !v.color.includes('255, 255, 255'))
                        ? v.color
                        : kitPassColor;

                      const optArrow = getPerspectiveArrowheadPoints(paddedX1, paddedY1, curX2, curY2, 15, 9, 0.35);

                      return (
                        <g key={`pass-option-${idx}`} style={{ opacity: optOpacity }}>
                          {/* 3D Grass Shadow */}
                          <line
                            x1={paddedX1}
                            y1={paddedY1 + 2.5}
                            x2={easedOptProgress > 0.04 ? optArrow.notchX : curX2}
                            y2={easedOptProgress > 0.04 ? optArrow.notchY + 2.5 : curY2 + 2.5}
                            stroke="rgba(0, 0, 0, 0.35)"
                            strokeWidth={3.8}
                            strokeDasharray="10 6"
                            strokeLinecap="round"
                          />
                          {easedOptProgress > 0.04 && (
                            <polygon
                              points={optArrow.shadowPoints}
                              fill="rgba(0, 0, 0, 0.35)"
                            />
                          )}
                          {/* Dashed Passing Lane Option */}
                          <line
                            x1={paddedX1}
                            y1={paddedY1}
                            x2={easedOptProgress > 0.04 ? optArrow.notchX : curX2}
                            y2={easedOptProgress > 0.04 ? optArrow.notchY : curY2}
                            stroke={optionColor}
                            strokeWidth={3.2}
                            strokeDasharray="10 6"
                            strokeLinecap="round"
                          />
                          {easedOptProgress > 0.04 && (
                            <polygon
                              points={optArrow.points}
                              fill={optionColor}
                            />
                          )}
                        </g>
                      );
                    }

                    if (v.type === 'PRESSING_RUN' || v.type === 'PRESSING_ARROW') {
                      const fromP = v.from_track_id ? playerMap.get(String(v.from_track_id)) : null;
                      const x1 = fromP
                        ? fromP.x
                        : (v.from_pct ? ((Array.isArray(v.from_pct) ? v.from_pct[0] : v.from_pct.x) / 100) * width : 0.5 * width);
                      const y1 = fromP
                        ? fromP.y
                        : (v.from_pct ? ((Array.isArray(v.from_pct) ? v.from_pct[1] : v.from_pct.y) / 100) * height : 0.5 * height);

                      let targetX = v.to_pct
                        ? ((Array.isArray(v.to_pct) ? v.to_pct[0] : v.to_pct.x) / 100) * width
                        : (catalystId && playerMap.has(catalystId) ? playerMap.get(catalystId)!.x : x1);
                      let targetY = v.to_pct
                        ? ((Array.isArray(v.to_pct) ? v.to_pct[1] : v.to_pct.y) / 100) * height
                        : (catalystId && playerMap.has(catalystId) ? playerMap.get(catalystId)!.y : y1);

                      // Pressing direction guard: A pressing run must physically orient towards the ball carrier
                      const catPt = catalystId ? playerMap.get(catalystId) : null;
                      if (catPt) {
                        const toCatDx = catPt.x - x1;
                        const toCatDy = catPt.y - y1;
                        const dotWithCat = (targetX - x1) * toCatDx + (targetY - y1) * toCatDy;
                        if (dotWithCat < 0) {
                          targetX = catPt.x;
                          targetY = catPt.y;
                        }
                      }

                      // Pressing run vectors: By default they do NOT appear unless triggered by AI commentary visual cue tag!
                      const runVectorTags = visualCueTags.filter(
                        (t) => t.type === 'run_vectors' || t.type === 'player_runs' || t.type === 'vectors'
                      );
                      const hasRunTag = runVectorTags.length > 0;
                      if (!hasRunTag) {
                        return null;
                      }

                      const earliestRunStart = Math.min(...runVectorTags.map((t) => t.start_frame ?? graphicsStart));
                      if (frame < earliestRunStart) {
                        return null;
                      }

                      const pressProgress = isGraphicsActive && frame >= earliestRunStart
                        ? Math.min(1, Math.max(0, (frame - earliestRunStart) / 38))
                        : 0;
                      const pressOpacity = isGraphicsActive && frame >= earliestRunStart
                        ? Math.min(1, pressProgress * 1.4) * freezeExitFade
                        : 0;
                      if (pressOpacity <= 0.01) return null;

                      const presserTeam = fromP?.p?.team || (v.team as string | undefined);
                      let pressColor: string;
                      if (presserTeam === 'away') {
                        pressColor = awayColor;
                      } else if (presserTeam === 'home') {
                        pressColor = homeColor;
                      } else if (fromP?.p?.color && fromP.p.color !== '#00e676' && fromP.p.color !== '#00e5ff') {
                        pressColor = fromP.p.color;
                      } else if (v.color && v.color !== '#00e676' && v.color !== '#00e5ff') {
                        pressColor = v.color;
                      } else {
                        pressColor = defJerseyColor || '#ef4444';
                      }

                      // Suppress head-on opposing pressing runs that clash with any primary pass vector
                      let isPressClash = false;
                      for (const pv of primaryPassVectors) {
                        const pdx = pv.x2 - pv.x1;
                        const pdy = pv.y2 - pv.y1;
                        const pLen = Math.hypot(pdx, pdy);
                        if (pLen < 1e-3) continue;

                        const dToTarget = Math.hypot(x1 - pv.x2, y1 - pv.y2);
                        if (dToTarget < 75) {
                          isPressClash = true;
                          break;
                        }

                        const vdx = targetX - x1;
                        const vdy = targetY - y1;
                        const vLen = Math.hypot(vdx, vdy);
                        if (vLen > 1e-3) {
                          const dot = (pdx / pLen) * (vdx / vLen) + (pdy / pLen) * (vdy / vLen);
                          const pnx = -pdy / pLen;
                          const pny = pdx / pLen;
                          const perpDist = Math.abs((x1 - pv.x1) * pnx + (y1 - pv.y1) * pny);
                          if (dot <= -0.65 && perpDist < 35) {
                            isPressClash = true;
                            break;
                          }
                        }
                      }
                      if (isPressClash) {
                        return null;
                      }

                      const [paddedX1, paddedY1, paddedTargetX, paddedTargetY] = applyFootHaloPadding(x1, y1, targetX, targetY, 18);

                      return renderRunChevrons(
                        `pressing-run-${idx}`,
                        paddedX1,
                        paddedY1,
                        paddedTargetX,
                        paddedTargetY,
                        pressProgress,
                        pressOpacity,
                        pressColor,
                        height
                      );
                    }

                    if (v.type === 'GROUND_PASS_SOLID') {
                      const fromP = (v.from_track_id && playerMap.has(String(v.from_track_id)))
                        ? playerMap.get(String(v.from_track_id))
                        : (catalystId && playerMap.has(catalystId) ? playerMap.get(catalystId) : null);

                      const curProgress = hasCandidateOptions ? executedProgress : vectorProgress;
                      const curPassOpacity = hasCandidateOptions ? executedOpacity : vectorOpacity;
                      if (curPassOpacity <= 0.01) return null;

                      const x1 = fromP
                        ? fromP.x
                        : (v.from_pct ? ((Array.isArray(v.from_pct) ? v.from_pct[0] : (v.from_pct as any).x) / 100) * width : (catalyst ? (catalyst.x_pct / 100) * width : 0.5 * width));
                      const y1 = fromP
                        ? fromP.y
                        : (v.from_pct ? ((Array.isArray(v.from_pct) ? v.from_pct[1] : (v.from_pct as any).y) / 100) * height : (catalyst ? (catalyst.y_pct / 100) * height : 0.5 * height));
                      const targetX = v.to_pct
                        ? ((Array.isArray(v.to_pct) ? v.to_pct[0] : v.to_pct.x) / 100) * width
                        : x1 + 0.15 * width;
                      const targetY = v.to_pct
                        ? ((Array.isArray(v.to_pct) ? v.to_pct[1] : v.to_pct.y) / 100) * height
                        : y1;

                      let dx = targetX - x1;
                      let dy = targetY - y1;
                      let dist = Math.hypot(dx, dy) || 1;
                      const hasExplicitTarget = Boolean(v.to_pct || v.landing_target_pct);
                      const fullX2 = hasExplicitTarget
                        ? Math.min(0.95 * width, Math.max(0.05 * width, targetX))
                        : Math.min(0.94 * width, Math.max(0.06 * width, x1 + (dx / dist) * Math.min(0.26 * width, Math.max(0.035 * width, dist))));
                      const fullY2 = hasExplicitTarget
                        ? Math.min(0.85 * height, Math.max(0.20 * height, targetY))
                        : Math.min(0.785 * height, Math.max(0.285 * height, y1 + (dy / dist) * Math.min(0.26 * width, Math.max(0.035 * width, dist))));

                      const [paddedX1, paddedY1, paddedFullX2, paddedFullY2] = applyFootHaloPadding(x1, y1, fullX2, fullY2, 22);
                      const easedPassProgress = easeVector(curProgress);
                      const curX2 = paddedX1 + (paddedFullX2 - paddedX1) * easedPassProgress;
                      const curY2 = paddedY1 + (paddedFullY2 - paddedY1) * easedPassProgress;

                      const passArrow = getPerspectiveArrowheadPoints(paddedX1, paddedY1, curX2, curY2, 18, 11, 0.35);
                      const carrierTeam = fromP.p?.team || (attTeam as string) || 'home';
                      const kitPassColor = getTeamKitPassColor(carrierTeam);
                      const passColor = (v.color && v.color !== '#00e5ff' && v.color !== '#ffffff') ? v.color : kitPassColor;

                      return (
                        <g key={`ground-pass-${idx}`} style={{ opacity: hasCandidateOptions ? curPassOpacity : 1.0 }}>
                          {/* 3D Perspective Turf Shadow */}
                          <line
                            x1={paddedX1}
                            y1={paddedY1 + 3.0}
                            x2={easedPassProgress > 0.04 ? passArrow.notchX : curX2}
                            y2={easedPassProgress > 0.04 ? passArrow.notchY + 3.0 : curY2 + 3.0}
                            stroke="rgba(0, 0, 0, 0.48)"
                            strokeWidth={5.2}
                            strokeLinecap="round"
                          />
                          {easedPassProgress > 0.04 && (
                            <polygon
                              points={passArrow.shadowPoints}
                              fill="rgba(0, 0, 0, 0.48)"
                            />
                          )}
                          {/* Crisp Clean Broadcast Ground Pass Vector */}
                          <line
                            x1={paddedX1}
                            y1={paddedY1}
                            x2={easedPassProgress > 0.04 ? passArrow.notchX : curX2}
                            y2={easedPassProgress > 0.04 ? passArrow.notchY : curY2}
                            stroke={passColor}
                            strokeWidth={3.6}
                            strokeLinecap="round"
                          />
                          {easedPassProgress > 0.04 && (
                            <polygon
                              points={passArrow.points}
                              fill={passColor}
                            />
                          )}
                        </g>
                      );
                    }

                    if (v.type === 'PLAYER_RUN_DASHED') {
                      const runnerP = v.from_track_id ? playerMap.get(String(v.from_track_id)) : null;
                      const x1 = runnerP
                        ? runnerP.x
                        : (v.from_pct ? ((Array.isArray(v.from_pct) ? v.from_pct[0] : v.from_pct.x) / 100) * width : 0.3 * width);
                      const y1 = runnerP
                        ? runnerP.y
                        : (v.from_pct ? ((Array.isArray(v.from_pct) ? v.from_pct[1] : v.from_pct.y) / 100) * height : 0.5 * height);
                      let targetX = v.to_pct
                        ? ((Array.isArray(v.to_pct) ? v.to_pct[0] : v.to_pct.x) / 100) * width
                        : x1 + 0.12 * width;
                      let targetY = v.to_pct
                        ? ((Array.isArray(v.to_pct) ? v.to_pct[1] : v.to_pct.y) / 100) * height
                        : y1;

                      // High-Accuracy Direction Engine for Player Runs:
                      // 1. Calculate true movement direction from temporal tracks around active frame
                      let trueVelDx = 0;
                      let trueVelDy = 0;
                      if (v.from_track_id && tracksMap) {
                        const rIdStr = String(v.from_track_id);
                        const cur = getSmoothedTurfPosition(tracksMap, rIdStr, activeFrameForTracks, width, height, isMoving);
                        const fut = getSmoothedTurfPosition(tracksMap, rIdStr, activeFrameForTracks + 18, width, height, isMoving);
                        const prv = getSmoothedTurfPosition(tracksMap, rIdStr, Math.max(0, activeFrameForTracks - 18), width, height, isMoving);
                        if (cur.isPresent && fut.isPresent && Math.hypot(fut.x - cur.x, fut.y - cur.y) >= 8) {
                          trueVelDx = fut.x - cur.x;
                          trueVelDy = fut.y - cur.y;
                        } else if (cur.isPresent && prv.isPresent && Math.hypot(cur.x - prv.x, cur.y - prv.y) >= 8) {
                          trueVelDx = cur.x - prv.x;
                          trueVelDy = cur.y - prv.y;
                        }
                      }

                      // If true velocity is confirmed and static vector points in opposite direction (dot < 0), correct direction
                      if (Math.hypot(trueVelDx, trueVelDy) >= 8) {
                        const dot = (targetX - x1) * trueVelDx + (targetY - y1) * trueVelDy;
                        if (dot < 0) {
                          targetX = x1 + trueVelDx;
                          targetY = y1 + trueVelDy;
                        }
                      } else {
                        // 2. Team attacking progression corridor alignment
                        const exploitCenter = causeTel?.exploit_zone?.center_pct;
                        const catPt = catalystId ? playerMap.get(catalystId) : null;
                        const progTargetX = exploitCenter
                          ? (exploitCenter[0] / 100) * width
                          : (catPt ? catPt.x : 0.5 * width);
                        const attProgDx = progTargetX - x1;
                        if (Math.abs(attProgDx) > 30) {
                          const dotProg = (targetX - x1) * attProgDx;
                          if (dotProg < 0) {
                            targetX = x1 + (attProgDx > 0 ? 0.10 * width : -0.10 * width);
                          }
                        }
                      }

                      let dx = targetX - x1;
                      let dy = targetY - y1;
                      let dist = Math.hypot(dx, dy) || 1;
                      const hasExplicitTarget = Boolean(v.to_pct || v.landing_target_pct);
                      const fullX2 = hasExplicitTarget
                        ? Math.min(0.95 * width, Math.max(0.05 * width, targetX))
                        : Math.min(0.94 * width, Math.max(0.06 * width, x1 + (dx / dist) * Math.min(0.22 * width, Math.max(0.035 * width, dist))));
                      const fullY2 = hasExplicitTarget
                        ? Math.min(0.85 * height, Math.max(0.20 * height, targetY))
                        : Math.min(0.785 * height, Math.max(0.285 * height, y1 + (dy / dist) * Math.min(0.22 * width, Math.max(0.035 * width, dist))));

                      const [paddedX1, paddedY1, paddedFullX2, paddedFullY2] = applyFootHaloPadding(x1, y1, fullX2, fullY2, 22);
                      const runnerTeam = runnerP?.p?.team || (v.team as string | undefined) || attTeam;
                      const kitRunColor = getTeamKitPassColor(runnerTeam);
                      let runColor: string;
                      if (v.color && v.color !== '#00e676' && v.color !== '#00e5ff') {
                        runColor = v.color;
                      } else if (runnerP?.p?.color && runnerP.p.color !== '#00e676' && runnerP.p.color !== '#00e5ff') {
                        runColor = runnerP.p.color;
                      } else {
                        runColor = kitRunColor;
                      }

                      return renderRunChevrons(
                        `player-run-${idx}`,
                        paddedX1,
                        paddedY1,
                        paddedFullX2,
                        paddedFullY2,
                        vectorProgress,
                        vectorOpacity,
                        runColor,
                        height
                      );
                    }


                    if (v.type === 'AERIAL_BALL_ARC') {
                      const curProgress = hasCandidateOptions ? executedProgress : vectorProgress;
                      const curArcOpacity = hasCandidateOptions ? executedOpacity : vectorOpacity;
                      if (curArcOpacity <= 0.01) return null;

                      const fromP = v.from_track_id ? playerMap.get(String(v.from_track_id)) : null;
                      const x1 = v.from_pct
                        ? ((Array.isArray(v.from_pct) ? v.from_pct[0] : v.from_pct.x) / 100) * width
                        : (fromP ? fromP.x : 0.25 * width);
                      const y1 = v.from_pct
                        ? ((Array.isArray(v.from_pct) ? v.from_pct[1] : v.from_pct.y) / 100) * height
                        : (fromP ? fromP.y : 0.40 * height);

                      const landingX = v.landing_target_pct
                        ? ((Array.isArray(v.landing_target_pct) ? v.landing_target_pct[0] : v.landing_target_pct.x) / 100) * width
                        : 0.68 * width;
                      const landingY = v.landing_target_pct
                        ? ((Array.isArray(v.landing_target_pct) ? v.landing_target_pct[1] : v.landing_target_pct.y) / 100) * height
                        : 0.48 * height;

                      const arcHeight = ((v.arc_height_pct ?? 18.0) / 100) * height;
                      const cx = (x1 + landingX) / 2;
                      const cy = Math.min(y1, landingY) - arcHeight;

                      const easedArcProgress = easeVector(curProgress);
                      const curShadowX = x1 + (landingX - x1) * easedArcProgress;
                      const curShadowY = y1 + (landingY - y1) * easedArcProgress;
                      const arcOffset = (1 - easedArcProgress) * 100;

                      const t = easedArcProgress;
                      const ballX = (1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * cx + t * t * landingX;
                      const ballY = (1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * cy + t * t * landingY;

                      return (
                        <g key={`aerial-arc-${idx}`} style={{ opacity: hasCandidateOptions ? curArcOpacity : 1.0 }}>
                          {/* a) Soft 3D perspective turf shadow flat on grass */}
                          <line
                            x1={x1}
                            y1={y1}
                            x2={curShadowX}
                            y2={curShadowY}
                            stroke="rgba(0, 0, 0, 0.38)"
                            strokeWidth={4.5}
                            strokeDasharray="10 6"
                            strokeLinecap="round"
                          />
                          <ellipse
                            cx={curShadowX}
                            cy={curShadowY}
                            rx={8}
                            ry={3.2}
                            fill="rgba(0, 0, 0, 0.48)"
                          />

                          {/* 3D vertical dashed altitude drop line from traveling ball to ground shadow */}
                          {curProgress > 0 && curProgress < 1 && (
                            <line
                              x1={ballX}
                              y1={ballY}
                              x2={curShadowX}
                              y2={curShadowY}
                              stroke="rgba(251, 191, 36, 0.65)"
                              strokeWidth={1.8}
                              strokeDasharray="4 3"
                            />
                          )}

                          {/* b) Broadcast 3D golden/amber parabolic flight arc */}
                          <path
                            d={`M ${x1} ${y1} Q ${cx} ${cy} ${landingX} ${landingY}`}
                            fill="none"
                            stroke="#fbbf24"
                            strokeWidth={4}
                            strokeLinecap="round"
                            pathLength={100}
                            strokeDasharray="100"
                            strokeDashoffset={arcOffset}
                            filter="url(#tacticalGlow)"
                          />

                          {/* 3D traveling ball along flight arc */}
                          {curProgress > 0 && curProgress < 1 && (
                            <g>
                              <circle
                                cx={ballX}
                                cy={ballY}
                                r={6.5}
                                fill="#ffffff"
                                stroke="#fbbf24"
                                strokeWidth={2}
                                filter="url(#tacticalGlow)"
                              />
                            </g>
                          )}

                          {/* c) Launch point ground ring at kicker foot */}
                          <ellipse
                            cx={x1}
                            cy={y1}
                            rx={18 * pulse}
                            ry={7 * pulse}
                            fill="rgba(251, 191, 36, 0.20)"
                            stroke="#fbbf24"
                            strokeWidth={2}
                            filter="url(#tacticalGlow)"
                          />

                          {/* d) Landing target ring at reception zone */}
                          <ellipse
                            cx={landingX}
                            cy={landingY}
                            rx={28 * pulse}
                            ry={10.5 * pulse}
                            fill="rgba(251, 191, 36, 0.22)"
                            stroke="#fbbf24"
                            strokeWidth={2.5}
                            filter="url(#tacticalGlow)"
                          />
                          <ellipse
                            cx={landingX}
                            cy={landingY}
                            rx={40 * pulse}
                            ry={15 * pulse}
                            fill="none"
                            stroke="#fbbf24"
                            strokeWidth={1.5}
                            strokeDasharray="5 4"
                            opacity={0.7}
                          />
                          <circle
                            cx={landingX}
                            cy={landingY}
                            r={3.5}
                            fill="#ffffff"
                            filter="url(#tacticalGlow)"
                          />

                          {/* e) Subtle frameless target zone label deleted per minimalist tactical UI */}
                        </g>
                      );
                    }

                    return null;
                  })}
                </g>
              );
            })()}

            {/* Volumetric Beacon Helper for Still Images (Subtle, non-intrusive broadcast light cone) */}
            {(() => {
              const renderVolumetricBeacon = (
                id: string,
                rx: number,
                ry: number,
                baseColor: string = '#00e5ff',
                localFrameProgress: number = freezeProgress,
                totalDuration: number = freezeDuration
              ) => {
                if (!isPhase2 || totalDuration <= 0) return null;

                const OPEN_FRAMES = 14;
                const CLOSE_FRAMES = 12;
                // Low-profile subtle height: rests near knees/lower-thigh (never obscures player torso or face)
                const NOMINAL_BEACON_HEIGHT = 65;

                // 1. Opening Animation (0 -> 1 over first OPEN_FRAMES of freeze)
                const openRatio = Math.min(1.0, Math.max(0.0, localFrameProgress / OPEN_FRAMES));
                const openEase = 1 - Math.pow(1 - openRatio, 3);
                const openHeightScale = openEase;
                const openOpacity = interpolate(openRatio, [0, 0.3, 1.0], [0, 0.70, 0.80], {
                  extrapolateLeft: 'clamp',
                  extrapolateRight: 'clamp',
                });

                // 2. Closing Animation (1 -> 0 over final CLOSE_FRAMES of freeze)
                const remainingFrames = Math.max(0, totalDuration - localFrameProgress);
                const closeRatio = Math.min(1.0, Math.max(0.0, remainingFrames / CLOSE_FRAMES));
                const closeEase = closeRatio < 0.5
                  ? 4 * closeRatio * closeRatio * closeRatio
                  : 1 - Math.pow(-2 * closeRatio + 2, 3) / 2;
                const closeHeightScale = closeEase;
                const closeOpacity = closeRatio;

                // Combined animation envelope
                const beaconHeightScale = openHeightScale * closeHeightScale;
                const beaconEnvelopeOpacity = openOpacity * closeOpacity;

                if (beaconHeightScale <= 0.01 || beaconEnvelopeOpacity <= 0.01) return null;

                // 3. Hover Animation (gentle atmospheric micro-breathing)
                const hoverFloatY = Math.sin(localFrameProgress * 0.09) * 2.5;
                const hoverWidthMult = 1.0 + Math.sin(localFrameProgress * 0.12) * 0.025;
                const hoverGlowMult = 0.92 + 0.08 * Math.sin(localFrameProgress * 0.12);

                const effHeight = NOMINAL_BEACON_HEIGHT * beaconHeightScale;
                const topY = -effHeight + hoverFloatY;

                const botRx = rx * 0.95 * hoverWidthMult;
                const topRx = rx * 0.65 * hoverWidthMult;

                const gradId = `tacticalBeaconGrad-${id}`;

                return (
                  <g key={`beacon-volumetric-${id}`} style={{ opacity: beaconEnvelopeOpacity }}>
                    <defs>
                      <linearGradient id={gradId} x1="0" y1="1" x2="0" y2="0">
                        <stop offset="0%" stopColor={baseColor} stopOpacity={0.14 * hoverGlowMult} />
                        <stop offset="30%" stopColor={baseColor} stopOpacity={0.06 * hoverGlowMult} />
                        <stop offset="65%" stopColor={baseColor} stopOpacity={0.015 * hoverGlowMult} />
                        <stop offset="100%" stopColor={baseColor} stopOpacity={0} />
                      </linearGradient>
                    </defs>

                    {/* Single gossamer atmospheric cone: soft, non-intrusive light column that never obstructs the player */}
                    <polygon
                      points={`${-botRx},0 ${botRx},0 ${topRx},${topY} ${-topRx},${topY}`}
                      fill={`url(#${gradId})`}
                    />
                  </g>
                );
              };

              return (
                <g key="beacon-helper-group">
                  {/* Carrier Spotlight with Dynamic Multi-Carrier Possession Lifecycle & Smooth Opening/Closing Animations */}
                  {(() => {
                    const candidateCarriers = (() => {
                      const list: {
                        spotId: string;
                        posStart: number;
                        posEnd: number;
                        framesSinceStart: number;
                        framesSinceEnd: number;
                        isClosing: boolean;
                      }[] = [];

                      const cCurr = dynamicCarrierId || catalystId;
                      const recentCarriers = new Set<string>();
                      if (cCurr) recentCarriers.add(cCurr);

                      // Also inspect if a previous carrier lost possession within the last 12 frames
                      for (let back = 1; back <= 12; back++) {
                        const fb = activeFrameForTracks - back;
                        if (fb >= 0) {
                          const cb = resolveCarrierForFrame(fb);
                          if (cb && !recentCarriers.has(cb)) {
                            recentCarriers.add(cb);
                          }
                        }
                      }

                      const maxFrames = Math.max(activeFrameForTracks + 300, (maxTracksFrame ?? resolvedTotalVideoFrames) || 600);

                      recentCarriers.forEach((spotId) => {
                        if (!spotId) return;

                        // Find start of contiguous possession interval
                        let posStart = activeFrameForTracks;
                        while (posStart > 0) {
                          const prevF = posStart - 1;
                          const prevC = resolveCarrierForFrame(prevF);
                          if (prevC === spotId) {
                            if (tracksMap && isMoving) {
                              const trs = tracksMap[prevF] || tracksMap[String(prevF)] || [];
                              if (!trs.some((t) => String(t.track_id) === spotId)) {
                                break;
                              }
                            }
                            posStart = prevF;
                          } else {
                            break;
                          }
                        }

                        // Find end of contiguous possession interval
                        let posEnd = activeFrameForTracks;
                        while (posEnd < maxFrames) {
                          const nextF = posEnd + 1;
                          const nextC = resolveCarrierForFrame(nextF);
                          if (nextC === spotId) {
                            posEnd = nextF;
                          } else {
                            break;
                          }
                        }

                        // If tracksMap is present, detect if track dropped earlier than posEnd
                        if (tracksMap && isMoving) {
                          let consecutiveMisses = 0;
                          for (let f = activeFrameForTracks; f <= posEnd; f++) {
                            const trs = tracksMap[f] || tracksMap[String(f)] || [];
                            if (!trs.some((t) => String(t.track_id) === spotId)) {
                              consecutiveMisses++;
                              if (consecutiveMisses > 2) {
                                posEnd = Math.max(activeFrameForTracks, f - consecutiveMisses);
                                break;
                              }
                            } else {
                              consecutiveMisses = 0;
                            }
                          }
                        }

                        const framesSinceStart = activeFrameForTracks - posStart;
                        const framesSinceEnd = activeFrameForTracks - posEnd;

                        // Retain carrier if actively in possession or within closing retraction window
                        const maxClosingFrames = isMoving ? 3 : 10;
                        if (framesSinceStart >= 0 && framesSinceEnd <= maxClosingFrames) {
                          list.push({
                            spotId,
                            posStart,
                            posEnd,
                            framesSinceStart,
                            framesSinceEnd,
                            isClosing: framesSinceEnd > 0,
                          });
                        }
                      });

                      return list;
                    })();

                    if (candidateCarriers.length === 0) return null;

                    const baseSpotlightRingOpacity = isPhase1
                      ? Math.min(1.0, frame / 10.0)
                      : isPhase2
                      ? 1.0
                      : isPhase3
                      ? (phase3Progress < effectivePostRoll ? seq3RingFade * 0.90 : 0.0)
                      : 0.0;

                    if (baseSpotlightRingOpacity <= 0) return null;

                    return candidateCarriers.map((cand) => {
                      const { spotId, framesSinceStart, framesSinceEnd, isClosing } = cand;

                      // Retrieve rock-solid smoothed turf coordinates (with velocity extrapolation if track dropped)
                      let x = 0;
                      let y = 0;
                      let y_pct = 40;
                      if (tracksMap) {
                        const sm = getSmoothedTurfPosition(tracksMap, spotId, activeFrameForTracks, width, height, isMoving);
                        if (!sm.isPresent && !isPhase2) return null;
                        if (sm.isPresent) {
                          x = sm.x;
                          y = sm.y;
                          y_pct = (y / height) * 100;
                        } else if (isPhase2) {
                          const cp = playerMap.get(spotId);
                          if (!cp) return null;
                          x = cp.x;
                          y = cp.y;
                          y_pct = cp.p.y;
                        }
                      } else {
                        const cp = playerMap.get(spotId);
                        if (!cp) return null;
                        x = cp.x;
                        y = cp.y;
                        y_pct = cp.p.y;
                      }

                      if (x === 0 && y === 0) return null;

                      const depthFactor = Math.min(1.2, Math.max(0.55, 0.40 + 0.60 * (Math.min(100, Math.max(0, y_pct)) / 75)));
                      const rx = 24 * depthFactor;
                      const ry = 8 * depthFactor;

                      // 1. Opening Animation (elastic spring scale 0 -> 1.15 -> 1.0, radar acquisition ripple)
                      const OPEN_DURATION = 12;
                      const carrierOpenRatio = isPhase2
                        ? Math.min(1.0, Math.max(0.0, freezeProgress / 14))
                        : Math.min(1.0, Math.max(0.0, framesSinceStart / OPEN_DURATION));

                      const carrierOpenScale = carrierOpenRatio >= 1.0
                        ? 1.0
                        : (carrierOpenRatio <= 0.0 ? 0.0 : Math.sin(carrierOpenRatio * Math.PI * 0.5) * (1.0 + Math.sin(carrierOpenRatio * Math.PI) * 0.18));
                      const carrierOpenRippleScale = 0.45 + carrierOpenRatio * 0.95;
                      const carrierOpenRippleOpacity = (1.0 - carrierOpenRatio) * 0.70;

                      // 2. Closing Animation (smooth cubic ease retraction and dissolve upon losing possession)
                      const CLOSE_DURATION = isMoving ? 3 : 10;
                      const carrierCloseRatio = isPhase2
                        ? Math.min(1.0, Math.max(0.0, (freezeDuration - freezeProgress) / 14))
                        : isClosing
                        ? Math.min(1.0, Math.max(0.0, (CLOSE_DURATION - framesSinceEnd) / CLOSE_DURATION))
                        : 1.0;

                      const carrierCloseScale = carrierCloseRatio < 0.5
                        ? 4 * carrierCloseRatio * carrierCloseRatio * carrierCloseRatio
                        : 1 - Math.pow(-2 * carrierCloseRatio + 2, 3) / 2;
                      const carrierCloseOpacity = carrierCloseRatio;

                      // Combined Scale & Opacity Envelope
                      const carrierScale = carrierOpenScale * carrierCloseScale;
                      const carrierOpacity = Math.min(1.0, carrierOpenRatio * 1.5) * carrierCloseOpacity * baseSpotlightRingOpacity;

                      if (carrierScale <= 0.01 || carrierOpacity <= 0.01) return null;

                      // 3. Hover Breathing (subtle turf bloom pulse during freeze)
                      const carrierHoverPulse = isPhase2 ? 1.0 + Math.sin(freezeProgress * 0.12) * 0.035 : 1.0;
                      const carrierHoverGlow = isPhase2 ? 0.85 + 0.15 * Math.sin(freezeProgress * 0.12) : 1.0;
                      const spotColor = '#00e5ff'; // Broadcast tactical cyan/turquoise matching visual reference

                      return (
                        <g
                          key={`spotlight-carrier-${spotId}`}
                          transform={`translate(${x}, ${y}) scale(${carrierScale})`}
                          style={{ opacity: carrierOpacity }}
                        >
                          {/* Volumetric Beacon with Opening, Closing, and Hover Animations strictly during still image */}
                          {isPhase2 && renderVolumetricBeacon(`carrier-${spotId}`, rx, ry, spotColor, freezeProgress, freezeDuration)}

                          {/* 1. Perspective turf drop-shadow */}
                          <ellipse
                            cx={0}
                            cy={2.5}
                            rx={(rx + 1.5) * carrierHoverPulse}
                            ry={(ry + 0.8) * carrierHoverPulse}
                            fill="rgba(0, 0, 0, 0.35)"
                          />

                          {/* 2. Outer Soft Neon Turf Glow / Bloom (subtle broadcast lawn reflection) */}
                          <ellipse
                            cx={0}
                            cy={0}
                            rx={rx * 1.30 * carrierHoverPulse}
                            ry={ry * 1.30 * carrierHoverPulse}
                            fill={spotColor}
                            fillOpacity={0.06 * carrierHoverGlow * carrierCloseOpacity}
                            filter="url(#tacticalGlow)"
                          />

                          {/* 3. Opening Radar Acquisition Ripple */}
                          {carrierOpenRatio < 1.0 && carrierOpenRippleOpacity > 0.02 && !isClosing && (
                            <ellipse
                              cx={0}
                              cy={0}
                              rx={rx * carrierOpenRippleScale}
                              ry={ry * carrierOpenRippleScale}
                              fill="none"
                              stroke={spotColor}
                              strokeWidth={1.2}
                              strokeOpacity={carrierOpenRippleOpacity}
                            />
                          )}

                          {/* 4. Broadcast Ground Ring */}
                          <ellipse
                            cx={0}
                            cy={0}
                            rx={rx * carrierHoverPulse}
                            ry={ry * carrierHoverPulse}
                            fill="rgba(0, 229, 255, 0.05)"
                            stroke={spotColor}
                            strokeWidth={1.8}
                            filter="url(#tacticalGlow)"
                          />

                          {/* 5. Concentric Inner Ground Ring (flat on pitch, non-rotating) */}
                          <ellipse
                            cx={0}
                            cy={0}
                            rx={rx * 0.76}
                            ry={ry * 0.76}
                            fill="none"
                            stroke="#ffffff"
                            strokeWidth={1.0}
                            strokeOpacity={0.65 * carrierCloseOpacity}
                            strokeDasharray="5 3"
                          />

                          {/* 6. Center Ground Pip */}
                          <circle
                            cx={0}
                            cy={0}
                            r={2.5}
                            fill="#ffffff"
                            filter="url(#tacticalGlow)"
                          />
                        </g>
                      );
                    });
                  })()}

                  {/* Dynamic Player Spotlights triggered by AI visual cue tags [highlight:PlayerName] */}
                  {(() => {
                    const highlightTags = visualCueTags.filter((t) => t.type === 'highlight');
                    const activeHighlights = highlightTags.filter(
                      (t) => frame >= (t.start_frame ?? 0) && frame <= (t.end_frame ?? ((t.start_frame ?? 0) + (t.duration_frames ?? 135)))
                    );
                    if (activeHighlights.length === 0) return null;

                    return activeHighlights.map((hlTag, hlIdx) => {
                      const targetId = matchPlayerForHighlight(hlTag.target || (hlTag as any).player_name);
                      if (!targetId) return null;
                      const hp = playerMap.get(targetId);
                      if (!hp) return null;

                      let x = hp.x;
                      let y = hp.y;
                      if (tracksMap) {
                        const sm = getSmoothedTurfPosition(tracksMap, targetId, activeFrameForTracks, width, height, isMoving);
                        if (sm.isPresent) {
                          x = sm.x;
                          y = sm.y;
                        } else if (isMoving) {
                          return null;
                        }
                      }

                      if (x === 0 && y === 0) return null;

                      const rx = 28;
                      const ry = 10;
                      const localHlFrame = frame - (hlTag.start_frame ?? 0);
                      const hlDuration = hlTag.duration_frames ?? 135;

                      // Opening and closing spring scale for dynamic highlight
                      const hlOpenRatio = Math.min(1.0, Math.max(0.0, localHlFrame / 12));
                      const hlOpenScale = hlOpenRatio >= 1.0
                        ? 1.0
                        : (hlOpenRatio <= 0.0 ? 0.0 : Math.sin(hlOpenRatio * Math.PI * 0.5) * (1.0 + Math.sin(hlOpenRatio * Math.PI) * 0.18));
                      const hlCloseRatio = Math.min(1.0, Math.max(0.0, (hlDuration - localHlFrame) / 12));
                      const hlCloseScale = hlCloseRatio < 0.5
                        ? 4 * hlCloseRatio * hlCloseRatio * hlCloseRatio
                        : 1 - Math.pow(-2 * hlCloseRatio + 2, 3) / 2;
                      const hlScale = hlOpenScale * hlCloseScale;

                      const hlPulse = 1.0 + Math.sin(localHlFrame / 5) * 0.08;
                      const hlExit = localHlFrame > hlDuration - 12
                        ? interpolate(localHlFrame, [hlDuration - 12, hlDuration], [1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })
                        : Math.min(1.0, hlOpenRatio * 1.5);

                      if (hlScale <= 0.01 || hlExit <= 0.01) return null;

                      return (
                        <g
                          key={`dynamic-highlight-spotlight-${targetId}-${hlIdx}`}
                          transform={`translate(${x}, ${y}) scale(${hlScale})`}
                          style={{ opacity: hlExit }}
                        >
                          {/* Volumetric Beacon for highlighted player during freeze */}
                          {isPhase2 && renderVolumetricBeacon(`hl-${targetId}-${hlIdx}`, rx, ry, '#00e5ff', localHlFrame, hlDuration)}

                          {/* Perspective turf drop shadow */}
                          <ellipse
                            cx={0}
                            cy={2.5}
                            rx={(rx + 1.5) * hlPulse}
                            ry={(ry + 0.8) * hlPulse}
                            fill="rgba(0, 0, 0, 0.35)"
                          />
                          {/* Outer Soft Neon Turf Glow */}
                          <ellipse
                            cx={0}
                            cy={0}
                            rx={rx * 1.30 * hlPulse}
                            ry={ry * 1.30 * hlPulse}
                            fill="#00e5ff"
                            fillOpacity={0.06}
                            filter="url(#tacticalGlow)"
                          />
                          {/* Clean Minimalist Highlight Ground Ring */}
                          <ellipse
                            cx={0}
                            cy={0}
                            rx={rx * hlPulse}
                            ry={ry * hlPulse}
                            fill="rgba(0, 229, 255, 0.05)"
                            stroke="#00e5ff"
                            strokeWidth={1.8}
                            filter="url(#tacticalGlow)"
                          />
                          <ellipse
                            cx={0}
                            cy={0}
                            rx={rx * 0.76 * hlPulse}
                            ry={ry * 0.76 * hlPulse}
                            fill="none"
                            stroke="#ffffff"
                            strokeWidth={1.0}
                            strokeOpacity={0.65}
                            strokeDasharray="5 3"
                          />
                          <circle
                            cx={0}
                            cy={0}
                            r={2.5}
                            fill="#ffffff"
                            filter="url(#tacticalGlow)"
                          />
                        </g>
                      );
                    });
                  })()}
                </g>
              );
            })()}

            {/* Ground 3D Rings for Runners (strictly during freeze Phase 2 only - disabled during live play) */}
            {(() => {
              const runnerRingOpacity = isPhase2
                ? (freezeRatio >= 0.50 ? Math.min(1, Math.max(0, (freezeRatio - 0.50) / 0.08)) : 0.0)
                : 0.0;

              if (runnerRingOpacity <= 0) return null;

              const runnerIds = new Set<string>();
              resolvedVectors.forEach((v) => {
                if (v.type === 'PLAYER_RUN_DASHED' && v.from_track_id !== undefined) {
                  runnerIds.add(String(v.from_track_id));
                }
              });
              if (causeTel?.runner_track_id !== undefined && String(causeTel.runner_track_id) !== catalystId) {
                runnerIds.add(String(causeTel.runner_track_id));
              }

              return Array.from(runnerIds).map((rid, i) => {
                if (isMoving) {
                  const rTrNow = currentTracks.find((t) => String(t.track_id) === rid);
                  if (!rTrNow) return null;
                }
                const rp = playerMap.get(rid);
                if (!rp || rid === catalystId) return null;
                const { x, y } = rp;
                const y_pct = Math.min(100, Math.max(0, rp.p.y <= 100 ? rp.p.y : (rp.p.y / height) * 100));
                const rx = 26;
                const ry = 9 * (0.55 + 0.45 * (y_pct / 100));

                const runnerTagLocalFrame = isPhase2 && freezeDuration > 0
                  ? Math.max(0, frame - (T_preroll + Math.round(freezeDuration * 0.50)) - i * 3.5)
                  : 0;

                const tagRatio = Math.min(1.0, Math.max(0, runnerTagLocalFrame / 12));
                const tagScale = tagRatio >= 1.0
                  ? 1.0
                  : (tagRatio <= 0.0 ? 0.0 : Math.sin(tagRatio * Math.PI * 0.5) * (1.0 + Math.sin(tagRatio * Math.PI) * 0.18));
                const rippleScale = 0.5 + tagRatio * 0.85;
                const rippleOpacity = (1.0 - tagRatio) * 0.55;

                return (
                  <g key={`runner-3d-ring-${rid}-${i}`} transform={`translate(${x}, ${y}) scale(${tagScale})`} style={{ opacity: runnerRingOpacity }}>
                    {/* Perspective turf drop shadow */}
                    <ellipse
                      cx={0}
                      cy={2.5}
                      rx={rx + 1.5}
                      ry={ry + 0.8}
                      fill="rgba(0, 0, 0, 0.40)"
                    />
                    {/* Broadcast Runner Ground Ring */}
                    <ellipse
                      cx={0}
                      cy={0}
                      rx={rx}
                      ry={ry}
                      fill={attJerseyColor === '#ffffff' ? 'rgba(255, 255, 255, 0.18)' : 'rgba(225, 29, 72, 0.20)'}
                      stroke={attJerseyColor}
                      strokeWidth={2.2}
                      filter="url(#tacticalGlow)"
                    />
                    {/* Opening Radar Acquisition Ripple */}
                    {tagRatio < 1.0 && rippleOpacity > 0.02 && (
                      <ellipse
                        cx={0}
                        cy={0}
                        rx={rx * rippleScale}
                        ry={ry * rippleScale}
                        fill="none"
                        stroke={attJerseyColor}
                        strokeWidth={1.2}
                        strokeOpacity={rippleOpacity}
                        filter="url(#tacticalGlow)"
                      />
                    )}
                    {/* Subtle non-rotating inner dashed ring */}
                    <ellipse
                      cx={0}
                      cy={0}
                      rx={rx * 0.76}
                      ry={ry * 0.76}
                      fill="none"
                      stroke={attJerseyColor}
                      strokeWidth={1.2}
                      strokeDasharray="5 3"
                    />
                    <circle cx={0} cy={0} r={3.2 * pulse} fill={attJerseyColor} stroke="#ffffff" strokeWidth={1.5} filter="url(#tacticalGlow)" />
                  </g>
                );
              });
            })()}

            {/* Non-choreographed fallback spotlights */}
            {!isChoreographed &&
              focalPlayers.map((p) => {
                const pt = playerMap.get(String(p.id));
                if (!pt || !p.showRing) return null;
                let ringX = pt.x;
                let ringY = pt.y;
                let ringYPct = pt.p?.y ?? 50;
                if (tracksMap) {
                  const sm = getSmoothedTurfPosition(tracksMap, String(p.id), activeFrameForTracks, width, height, isMoving);
                  if (sm.isPresent) {
                    ringX = sm.x;
                    ringY = sm.y;
                    ringYPct = (sm.y / height) * 100;
                  } else if (isMoving) {
                    return null;
                  }
                }
                const depthFactor = Math.min(1.2, Math.max(0.55, 0.40 + 0.60 * (Math.min(100, Math.max(0, ringYPct)) / 75)));
                const rx = 20 * depthFactor;
                const ry = 6.5 * depthFactor;
                return (
                  <g key={`fallback-ring-${p.id}`} transform={`translate(${ringX}, ${ringY})`}>
                    <ellipse
                      cx={0}
                      cy={0}
                      rx={rx}
                      ry={ry}
                      fill="rgba(0, 229, 255, 0.15)"
                      stroke="#00e5ff"
                      strokeWidth={1.5}
                      filter="url(#tacticalGlow)"
                    />
                  </g>
                );
              })}
          </svg>
        )}

        {/* Subtle Frameless Overhead Player Name Text (@TacticDive Style) during Freeze-Frame & Post-Freeze Motion - Suppressed on non-tactical shots */}
        {!renderTurfOnly && effectiveIsTactical && (
          <AbsoluteFill style={{ pointerEvents: 'none' }}>
            {((isPhase2 && freezeExitFade > 0) || (isPhase3 && seq3RingFade > 0) || visualCueTags.some((t) => t.type === 'highlight' && frame >= (t.start_frame ?? 0) && frame <= (t.end_frame ?? ((t.start_frame ?? 0) + (t.duration_frames ?? 135))))) && (() => {
              const activeHighlightTags = visualCueTags.filter(
                (t) => t.type === 'highlight' && frame >= (t.start_frame ?? 0) && frame <= (t.end_frame ?? ((t.start_frame ?? 0) + (t.duration_frames ?? 135)))
              );
              const activeHighlightIds = new Set<string>();
              activeHighlightTags.forEach((t) => {
                const mid = matchPlayerForHighlight(t.target);
                if (mid) activeHighlightIds.add(mid);
              });
              // Seamless synchronization with multi-player broadcast camera slide:
              highlightTargets.forEach((ht) => {
                if (frame >= ht.startFrame && frame <= ht.endFrame) {
                  activeHighlightIds.add(ht.playerId);
                }
              });

              // Intelligent Tag-Driven Staging:
              // Only display overhead badges for players who are actively relevant at the current frame!
              const activeOptionReceiverIds = new Set<string>();
              if (optOpacity > 0.01) {
                let candidateCount = 0;
                resolvedVectors.forEach((v) => {
                  if ((v.type === 'PASS_OPTION_DASHED' || (v.type === 'GROUND_PASS_SOLID' && v.was_taken === false)) && v.to_track_id !== undefined) {
                    const fromP = v.from_track_id
                      ? playerMap.get(String(v.from_track_id))
                      : (catalystId ? playerMap.get(catalystId) : null);
                    if (fromP) {
                      const toP = playerMap.get(String(v.to_track_id));
                      const x1 = fromP.x;
                      const y1 = fromP.y;
                      const targetX = toP
                        ? toP.x
                        : (v.to_pct ? ((Array.isArray(v.to_pct) ? v.to_pct[0] : v.to_pct.x) / 100) * width : x1 + 0.15 * width);
                      const targetY = toP
                        ? toP.y
                        : (v.to_pct ? ((Array.isArray(v.to_pct) ? v.to_pct[1] : v.to_pct.y) / 100) * height : y1);
                      const fullX2 = Math.min(0.95 * width, Math.max(0.05 * width, targetX));
                      const fullY2 = Math.min(0.85 * height, Math.max(0.20 * height, targetY));
                      const carrierTeam = fromP.p?.team || (attTeam as string) || 'home';
                      const safety = isPassLaneBlockedOrCovered(x1, y1, fullX2, fullY2, carrierTeam);
                      if (!safety.isBlocked && candidateCount < 2) {
                        activeOptionReceiverIds.add(String(v.to_track_id));
                        candidateCount++;
                      }
                    }
                  }
                });
              }

              const activeExecutedReceiverIds = new Set<string>();
              if (executedOpacity > 0.01) {
                resolvedVectors.forEach((v) => {
                  if ((v.type === 'GROUND_PASS_SOLID' || v.type === 'AERIAL_BALL_ARC') && v.was_taken !== false) {
                    if (v.to_track_id !== undefined) {
                      activeExecutedReceiverIds.add(String(v.to_track_id));
                    } else if (v.to_pct || v.landing_target_pct) {
                      const tgt = v.to_pct || v.landing_target_pct;
                      const tx = Array.isArray(tgt) ? tgt[0] : (tgt as any).x;
                      const ty = Array.isArray(tgt) ? tgt[1] : (tgt as any).y;
                      let bestId: string | null = null;
                      let bestDist = 14.0;
                      for (const [pId, pPt] of playerMap.entries()) {
                        if (pId === catalystId) continue;
                        const pxPct = (pPt.x / width) * 100;
                        const pyPct = (pPt.y / height) * 100;
                        const d = Math.hypot(pxPct - tx, pyPct - ty);
                        if (d < bestDist) {
                          bestDist = d;
                          bestId = pId;
                        }
                      }
                      if (bestId) activeExecutedReceiverIds.add(bestId);
                    }
                  }
                });
                if (activeExecutedReceiverIds.size === 0 && primaryRunnerId) {
                  activeExecutedReceiverIds.add(primaryRunnerId);
                }
              }

              const activePresserIds = new Set<string>();
              const runVectorTags = visualCueTags.filter(
                (t) => t.type === 'run_vectors' || t.type === 'player_runs' || t.type === 'vectors'
              );
              const earliestRunStart = runVectorTags.length > 0
                ? Math.min(...runVectorTags.map((t) => t.start_frame ?? graphicsStart))
                : graphicsStart;
              const latestRunEnd = runVectorTags.length > 0
                ? Math.max(...runVectorTags.map((t) => t.end_frame ?? (earliestRunStart + 60)))
                : (graphicsStart + Math.round(graphicsDuration * 0.44));

              if (isGraphicsActive && frame >= earliestRunStart && frame <= latestRunEnd && freezeExitFade > 0) {
                resolvedVectors.forEach((v) => {
                  if ((v.type === 'PRESSING_RUN' || v.type === 'PRESSING_ARROW') && v.from_track_id !== undefined) {
                    activePresserIds.add(String(v.from_track_id));
                  }
                });
              }

              const targetPlayerIds = new Set<string>();
              if (isPhase2 && freezeExitFade > 0) {
                if (catalystId) targetPlayerIds.add(catalystId);
                resolvedInitialPlayers.forEach((p) => targetPlayerIds.add(String(p.id)));
                const causeP = cause_telestration?.players || (causeTelestration as any)?.players;
                if (Array.isArray(causeP)) {
                  causeP.forEach((cp: any) => targetPlayerIds.add(String(cp.id ?? cp.track_id)));
                }
              }
              // Dynamically track active ball carrier post-freeze throughout Phase 3
              if (isPhase3 && seq3RingFade > 0 && activeSpotlightId) {
                targetPlayerIds.add(activeSpotlightId);
              }
              activeOptionReceiverIds.forEach((id) => targetPlayerIds.add(id));
              activeExecutedReceiverIds.forEach((id) => targetPlayerIds.add(id));
              activePresserIds.forEach((id) => targetPlayerIds.add(id));
              activeHighlightIds.forEach((id) => targetPlayerIds.add(id));

              // SofaScore Score Color Resolver (Emerald >= 9, Green >= 8, Blue >= 7.3, Light Green >= 7.0, Amber >= 6.6, Yellow >= 6.0, Red < 6.0)
              const getSofaScoreRatingColor = (score: number): string => {
                if (score >= 9.0) return '#00c067'; // Pure vibrant emerald (9.0+)
                if (score >= 8.0) return '#10b981'; // Green (8.0 - 8.9)
                if (score >= 7.3) return '#2563eb'; // Royal Blue (7.3 - 7.9) e.g. Vinicius 7.7
                if (score >= 7.0) return '#22c55e'; // Light Green (7.0 - 7.2) e.g. Raphinha 7.1
                if (score >= 6.6) return '#ea580c'; // Amber / Orange (6.6 - 6.9) e.g. Marquinhos 6.9, Paqueta 6.8
                if (score >= 6.0) return '#eab308'; // Warm Yellow (6.0 - 6.5) e.g. Casemiro 6.6, Ibanez 6.0
                return '#ef4444'; // Red (< 6.0)
              };

              // Dynamic Team Kit & Border Style Resolver (Adapts to jerseys, striped patterns, e.g. Barca Red/Blue, Real Madrid White/Red)
              interface PlayerKitStyle {
                bg: string;
                textColor: string;
                borderColor: string;
                roleBg: string;
                roleTextColor: string;
                isStriped: boolean;
                stripeColor1?: string;
                stripeColor2?: string;
              }

              const resolvePlayerJerseyKit = (
                teamVal?: string,
                isHomePlayer?: boolean,
                tColors?: any,
                cTel?: any
              ): PlayerKitStyle => {
                const normTeam = (teamVal || '').toLowerCase();
                const hName = String(cTel?.home_team || cTel?.homeTeam || '').toLowerCase();
                const aName = String(cTel?.away_team || cTel?.awayTeam || '').toLowerCase();
                const targetName = isHomePlayer ? hName : (normTeam.includes('away') ? aName : normTeam);

                // Priority 1: Dynamically adapted team_kits from scraper / fixture payloads
                const teamKey = isHomePlayer ? 'home' : 'away';
                const kitData = cTel?.team_kits?.[teamKey] || cTel?.team_kits?.[normTeam];
                if (kitData) {
                  const pCol = kitData.primary || (isHomePlayer ? (tColors?.home_hex || cTel?.home_hex) : (tColors?.away_hex || cTel?.away_hex)) || '#ffffff';
                  const sCol = kitData.secondary || (isHomePlayer ? (tColors?.home_secondary || cTel?.home_secondary) : (tColors?.away_secondary || cTel?.away_secondary)) || '#e11d48';
                  const isStr = Boolean(kitData.is_striped ?? kitData.isStriped);
                  if (isStr) {
                    return {
                      bg: 'linear-gradient(135deg, rgba(20, 8, 18, 0.95) 0%, rgba(8, 16, 32, 0.96) 100%)',
                      textColor: kitData.text || '#ffffff',
                      borderColor: sCol,
                      roleBg: 'rgba(255, 255, 255, 0.20)',
                      roleTextColor: '#ffffff',
                      isStriped: true,
                      stripeColor1: pCol,
                      stripeColor2: sCol,
                    };
                  }
                  const cleanHex = pCol.replace('#', '');
                  const r = parseInt(cleanHex.substring(0, 2) || '80', 16);
                  const g = parseInt(cleanHex.substring(2, 4) || '80', 16);
                  const b = parseInt(cleanHex.substring(4, 6) || '80', 16);
                  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                  const isLight = lum > 0.65;
                  return {
                    bg: isLight ? 'rgba(255, 255, 255, 0.96)' : `rgba(${r}, ${g}, ${b}, 0.94)`,
                    textColor: kitData.text || (isLight ? '#0c1b33' : '#ffffff'),
                    borderColor: sCol || (isLight ? '#e11d48' : '#ffffff'),
                    roleBg: isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.18)',
                    roleTextColor: isLight ? '#0c1b33' : '#ffffff',
                    isStriped: false,
                    stripeColor1: pCol,
                    stripeColor2: sCol,
                  };
                }

                // Priority 2: Authoritative presets for known clubs
                // 1. Barcelona Blaugrana (Alternating Red and Blue segmented border)
                if (targetName.includes('barca') || targetName.includes('barcelona') || normTeam.includes('barca')) {
                  return {
                    bg: 'linear-gradient(135deg, rgba(20, 8, 18, 0.95) 0%, rgba(8, 16, 32, 0.96) 100%)',
                    textColor: '#ffffff',
                    borderColor: '#db0030',
                    roleBg: 'rgba(219, 0, 48, 0.28)',
                    roleTextColor: '#38bdf8',
                    isStriped: true,
                    stripeColor1: '#db0030',
                    stripeColor2: '#004d98',
                  };
                }

                // 2. Real Madrid (White background, dark text, red outline)
                if (targetName.includes('real madrid') || targetName.includes('madrid') || normTeam.includes('madrid')) {
                  return {
                    bg: 'rgba(255, 255, 255, 0.96)',
                    textColor: '#0c1b33',
                    borderColor: '#e11d48',
                    roleBg: 'rgba(225, 29, 72, 0.14)',
                    roleTextColor: '#e11d48',
                    isStriped: false,
                  };
                }

                // 3. AS Roma (Carmine maroon background, gold outline)
                if (targetName.includes('roma') || normTeam.includes('roma')) {
                  return {
                    bg: 'linear-gradient(135deg, rgba(138, 21, 41, 0.95) 0%, rgba(95, 12, 28, 0.97) 100%)',
                    textColor: '#ffffff',
                    borderColor: '#f1b434',
                    roleBg: 'rgba(241, 180, 52, 0.24)',
                    roleTextColor: '#fcd34d',
                    isStriped: false,
                  };
                }

                // 4. Striped kit teams: Atletico, Juventus, Inter, AC Milan
                if (targetName.includes('atletico')) {
                  return {
                    bg: 'linear-gradient(135deg, rgba(30, 10, 10, 0.95) 0%, rgba(15, 15, 25, 0.96) 100%)',
                    textColor: '#ffffff',
                    borderColor: '#cb3524',
                    roleBg: 'rgba(203, 53, 36, 0.25)',
                    roleTextColor: '#ffffff',
                    isStriped: true,
                    stripeColor1: '#cb3524',
                    stripeColor2: '#ffffff',
                  };
                }
                if (targetName.includes('juventus')) {
                  return {
                    bg: 'rgba(15, 15, 20, 0.96)',
                    textColor: '#ffffff',
                    borderColor: '#ffffff',
                    roleBg: 'rgba(255, 255, 255, 0.2)',
                    roleTextColor: '#ffffff',
                    isStriped: true,
                    stripeColor1: '#000000',
                    stripeColor2: '#ffffff',
                  };
                }
                if (targetName.includes('inter')) {
                  return {
                    bg: 'linear-gradient(135deg, rgba(0, 30, 60, 0.95) 0%, rgba(5, 10, 20, 0.96) 100%)',
                    textColor: '#ffffff',
                    borderColor: '#0066b2',
                    roleBg: 'rgba(0, 102, 178, 0.25)',
                    roleTextColor: '#38bdf8',
                    isStriped: true,
                    stripeColor1: '#0066b2',
                    stripeColor2: '#000000',
                  };
                }
                if (targetName.includes('milan')) {
                  return {
                    bg: 'linear-gradient(135deg, rgba(40, 8, 8, 0.95) 0%, rgba(10, 10, 12, 0.96) 100%)',
                    textColor: '#ffffff',
                    borderColor: '#fb090b',
                    roleBg: 'rgba(251, 9, 11, 0.25)',
                    roleTextColor: '#ffffff',
                    isStriped: true,
                    stripeColor1: '#fb090b',
                    stripeColor2: '#000000',
                  };
                }
                if (targetName.includes('dortmund')) {
                  return {
                    bg: 'rgba(253, 225, 0, 0.96)',
                    textColor: '#000000',
                    borderColor: '#000000',
                    roleBg: 'rgba(0, 0, 0, 0.15)',
                    roleTextColor: '#000000',
                    isStriped: false,
                  };
                }

                // 5. Dynamic Kit Fallback derived from scraped match data
                const hHex = tColors?.home_hex || cTel?.home_hex || '#8a1529';
                const aHex = tColors?.away_hex || cTel?.away_hex || '#ffffff';
                const hSec = tColors?.home_secondary || cTel?.home_secondary || '#f1b434';
                const aSec = tColors?.away_secondary || cTel?.away_secondary || '#e11d48';

                const pColor = isHomePlayer ? hHex : aHex;
                const sColor = isHomePlayer ? hSec : aSec;

                // Check luminance of primary color
                const cleanHex = pColor.replace('#', '');
                const r = parseInt(cleanHex.substring(0, 2) || '80', 16);
                const g = parseInt(cleanHex.substring(2, 4) || '80', 16);
                const b = parseInt(cleanHex.substring(4, 6) || '80', 16);
                const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

                const isLight = lum > 0.65;
                const kitStriped = Boolean(cTel?.team_kits?.[teamKey]?.is_striped);

                return {
                  bg: isLight ? 'rgba(255, 255, 255, 0.96)' : `rgba(${r}, ${g}, ${b}, 0.94)`,
                  textColor: isLight ? '#0c1b33' : '#ffffff',
                  borderColor: sColor || (isLight ? '#e11d48' : '#ffffff'),
                  roleBg: isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.18)',
                  roleTextColor: isLight ? '#0c1b33' : '#ffffff',
                  isStriped: kitStriped,
                  stripeColor1: pColor,
                  stripeColor2: sColor,
                };
              };

              // Dynamic match clock minute
              let startM = 50;
              let startS = 6;
              const clockRaw = (causeTel as any)?.clock;
              if (typeof clockRaw === 'string' && clockRaw.includes(':')) {
                const [m, s] = clockRaw.split(':').map((v: string) => parseInt(v, 10));
                if (!isNaN(m)) startM = m;
                if (!isNaN(s)) startS = s;
              } else if (typeof (causeTel as any)?.match_minute === 'number') {
                startM = (causeTel as any).match_minute;
                startS = 0;
              }
              const elapsedSec = Math.floor(frame / (fps || 30));
              const totalSec = startM * 60 + startS + elapsedSec;
              const currentMatchMinute = Math.floor(totalSec / 60);

              interface OverheadLabelItem {
                id: string;
                text: string;
                x: number;
                y: number;
                footX: number;
                footY: number;
                priority: number;
                isHighlighted?: boolean;
                portraitUrl?: string;
                team?: 'home' | 'away' | string;
                rating?: number;
                ratingColor?: string;
                visibleGoals: number;
                visibleAssists: number;
                hasYellowCard: boolean;
                hasRedCard: boolean;
                isSubbedOut: boolean;
                isCaptain: boolean;
                kit: PlayerKitStyle;
              }

              const candidateLabels: OverheadLabelItem[] = [];

              targetPlayerIds.forEach((idStr) => {
                const pt = playerMap.get(idStr);
                if (!pt) return;

                const tr = currentTracks.find((t) => String(t.track_id) === idStr);
                const causePlayer = Array.isArray(causeTel?.players)
                  ? causeTel.players.find((cp: any) => String(cp.id || cp.track_id) === idStr)
                  : null;

                const isHighlighted = activeHighlightIds.has(idStr);

                let raw = (causePlayer?.name || tr?.name || pt.p.name || tr?.name_pos || pt.p.label || '').trim();
                if (raw.includes(' // ')) {
                  raw = raw.split(' // ')[0].trim();
                }

                const isPlaceholder = (s: string) =>
                  !s ||
                  s === 'BALL CARRIER' ||
                  s === 'THIRD-MAN RUNNER' ||
                  s === 'COVERING CB' ||
                  s.toLowerCase() === 'support' ||
                  /^(defender|player|runner|carrier)(\s*\d+)?(\s*\(.*\))?$/i.test(s) ||
                  isForbiddenPlayer(s);

                if (isPlaceholder(raw)) {
                  if (idStr === catalystId) {
                    const catCand = catalyst?.name || catalyst?.name_pos || causeTel?.catalyst_player?.name || causeTel?.catalyst_player?.name_pos || '';
                    if (!isPlaceholder(catCand)) {
                      raw = catCand;
                    } else {
                      // Find active tag matching this catalyst
                      const focalTag = activeHighlightTags.find((t) => t.player_name || t.target) || 
                                       (visualCueTags || []).find((t: any) => t.player_name || t.target);
                      if (focalTag) {
                        raw = (focalTag.player_name || focalTag.player_stats?.playerName || focalTag.target || '').toUpperCase();
                      }
                    }
                  } else if (idStr === primaryRunnerId) {
                    const runCand = causeTel?.runner_player?.name || causeTel?.runner_player?.name_pos || '';
                    if (!isPlaceholder(runCand)) {
                      raw = runCand;
                    }
                  } else if (causePlayer?.name && !isPlaceholder(causePlayer.name)) {
                    raw = causePlayer.name;
                  }
                }
                if (isPlaceholder(raw) && isHighlighted) {
                  const hTag = activeHighlightTags.find((t) => matchPlayerForHighlight(t.target) === idStr);
                  const resolvedName = hTag?.player_name || hTag?.player_stats?.playerName || hTag?.target;
                  if (resolvedName && !isPlaceholder(resolvedName)) {
                    raw = resolvedName.toUpperCase();
                  }
                }
                if (raw.includes(' // ')) {
                  raw = raw.split(' // ')[0].trim();
                }
                if (isPlaceholder(raw)) {
                  return;
                }
                if (raw.toLowerCase().includes('kroos') || isForbiddenPlayer(raw)) {
                  return;
                }

                let headX = pt.x;
                let headY = pt.y - 58;
                if (tr?.bbox && Array.isArray(tr.bbox) && tr.bbox.length === 4) {
                  const [ymin, xmin, ymax, xmax] = tr.bbox;
                  if (ymax > ymin && xmax > xmin) {
                    headY = (ymin / 1080) * height - 12;
                    headX = ((xmin + xmax) / (2 * 1920)) * width;
                  }
                }

                // Strict clamp within [8.0%, 90.0%] screen width so labels are never cut off by borders
                headX = Math.max(0.08 * width, Math.min(0.90 * width, headX));

                // Priority ranking: Highlighted (130) > Catalyst (100) > Runner (90) > Tactical Named (75) > Chain players (50)
                let prio = 50;
                if (isHighlighted) {
                  prio = 130;
                } else if (idStr === catalystId) {
                  prio = 100;
                } else if (idStr === primaryRunnerId) {
                  prio = 90;
                } else if (causePlayer?.name && !isPlaceholder(causePlayer.name)) {
                  prio = 75;
                }

                const resolvedSlug = getPlayerPortraitSlug(raw);
                let portraitUrl: string | undefined = undefined;
                if (resolvedSlug) {
                  try {
                    portraitUrl = staticFile(`assets/players/${resolvedSlug}.png`);
                  } catch (e) {
                    portraitUrl = `/assets/players/${resolvedSlug}.png`;
                  }
                }

                // Resolve player team and kit style
                const playerTeam = tr?.team || causePlayer?.team || pt.p.team || (idStr === catalystId ? (causeTel?.attacking_team || 'home') : undefined);
                const isHomePlayer = playerTeam === 'home' || playerTeam === 'attacking' ? (causeTel?.attacking_team !== 'away') : (playerTeam === 'away' ? false : true);
                const kit = resolvePlayerJerseyKit(playerTeam, isHomePlayer, team_colors || teamColors, causeTel);

                // SofaScore stats extraction & live match timestamp evaluation
                const rawNorm = raw.split(' (')[0].trim().toLowerCase();
                const hTag = activeHighlightTags.find((t) => matchPlayerForHighlight(t.target) === idStr);
                const statsSource =
                  (hTag?.player_stats as any) ||
                  (causeTel?.player_stats as any)?.[rawNorm] ||
                  (causeTel?.player_stats as any)?.[resolvedSlug] ||
                  (causeTel?.player_stats as any)?.[idStr] ||
                  (causePlayer as any)?.stats ||
                  {};

                const allIncidents = Array.isArray(causeTel?.incidents) ? causeTel.incidents : (Array.isArray(causeTel?.match_events) ? causeTel.match_events : []);
                let goalCount = Number(statsSource.goals ?? 0);
                let assistCount = Number(statsSource.assists ?? 0);
                let yellowCardCount = Number(statsSource.yellow_cards ?? statsSource.yellowCards ?? 0);
                let redCardCount = Number(statsSource.red_cards ?? statsSource.redCards ?? 0);
                let isSubbedOut = Boolean(statsSource.is_substituted ?? statsSource.isSubstituted);
                let isCaptain = Boolean(statsSource.is_captain ?? statsSource.captain);

                const goalMins: number[] = Array.isArray(statsSource.event_minutes?.goals) ? [...statsSource.event_minutes.goals] : [];
                const assistMins: number[] = Array.isArray(statsSource.event_minutes?.assists) ? [...statsSource.event_minutes.assists] : [];
                const yellowMins: number[] = Array.isArray(statsSource.event_minutes?.yellow_cards) ? [...statsSource.event_minutes.yellow_cards] : [];
                const redMins: number[] = Array.isArray(statsSource.event_minutes?.red_cards) ? [...statsSource.event_minutes.red_cards] : [];

                allIncidents.forEach((inc: any) => {
                  const m = Number(inc.minute || inc.time || 0);
                  const incType = String(inc.type || inc.incidentType || '').toLowerCase();
                  const pName = (inc.player || inc.playerName || inc.player_sent_off || '').trim().toLowerCase();
                  if (incType.includes('goal')) {
                    if (pName.includes(rawNorm) || rawNorm.includes(pName)) {
                      if (!goalMins.includes(m)) {
                        goalMins.push(m);
                        goalCount = Math.max(goalCount, goalMins.length);
                      }
                    }
                    const aName = String(inc.assist || inc.assist1 || '').trim().toLowerCase();
                    if (aName && (aName.includes(rawNorm) || rawNorm.includes(aName))) {
                      if (!assistMins.includes(m)) {
                        assistMins.push(m);
                        assistCount = Math.max(assistCount, assistMins.length);
                      }
                    }
                  } else if (incType.includes('card')) {
                    const isRed = incType.includes('red') || String(inc.incidentClass || '').toLowerCase().includes('red');
                    if (pName.includes(rawNorm) || rawNorm.includes(pName)) {
                      if (isRed) {
                        if (!redMins.includes(m)) {
                          redMins.push(m);
                          redCardCount = Math.max(redCardCount, redMins.length);
                        }
                      } else {
                        if (!yellowMins.includes(m)) {
                          yellowMins.push(m);
                          yellowCardCount = Math.max(yellowCardCount, yellowMins.length);
                        }
                      }
                    }
                  } else if (incType.includes('sub')) {
                    const outP = (inc.player_out || inc.playerOut || '').trim().toLowerCase();
                    if (outP.includes(rawNorm) || rawNorm.includes(outP)) {
                      isSubbedOut = true;
                    }
                  }
                });

                // Gradual reveal based on current match minute
                const hasTimestamps = goalMins.length > 0 || assistMins.length > 0 || yellowMins.length > 0 || redMins.length > 0;
                const visibleGoals = hasTimestamps ? goalMins.filter((m) => m <= currentMatchMinute).length : goalCount;
                const visibleAssists = hasTimestamps ? assistMins.filter((m) => m <= currentMatchMinute).length : assistCount;
                const hasYellowCard = hasTimestamps ? yellowMins.some((m) => m <= currentMatchMinute) : yellowCardCount > 0;
                const hasRedCard = hasTimestamps ? redMins.some((m) => m <= currentMatchMinute) : redCardCount > 0;

                // Authentic SofaScore rating
                let rating = statsSource.rating !== undefined ? Number(statsSource.rating) : (6.8 + (idStr === catalystId ? 0.4 : 0));
                if (statsSource.rating === undefined) {
                  rating += visibleGoals * 0.9 + visibleAssists * 0.5 - (hasYellowCard ? 0.4 : 0) - (hasRedCard ? 1.8 : 0);
                  rating = Math.max(4.5, Math.min(9.8, Math.round(rating * 10) / 10));
                }
                const ratingColor = getSofaScoreRatingColor(rating);

                candidateLabels.push({
                  id: idStr,
                  text: raw,
                  x: headX,
                  y: headY,
                  footX: pt.x,
                  footY: pt.y,
                  priority: prio,
                  isHighlighted,
                  portraitUrl,
                  team: playerTeam,
                  rating,
                  ratingColor,
                  visibleGoals,
                  visibleAssists,
                  hasYellowCard,
                  hasRedCard,
                  isSubbedOut,
                  isCaptain,
                  kit,
                });
              });

              // 1. Deduplicate by player name (keep higher priority instance)
              candidateLabels.sort((a, b) => b.priority - a.priority);
              const seenNames = new Set<string>();
              const deduplicatedLabels: OverheadLabelItem[] = [];
              for (const item of candidateLabels) {
                const normName = item.text.split(' (')[0].trim().toLowerCase();
                if (!seenNames.has(normName)) {
                  seenNames.add(normName);
                  deduplicatedLabels.push(item);
                }
              }

              // 2. 2D Bounding-Box Anti-Stacking Collision Guard across Unified Array & Player Foot/Body Anchors
              const finalPlacedLabels: OverheadLabelItem[] = [];
              const BADGE_MIN_DX = 125.0;
              const BADGE_MIN_DY = 48.0;

              const overlapsAnyPlacedOrPlayerFoot = (cx: number, cy: number, selfId: string): boolean => {
                for (const placed of finalPlacedLabels) {
                  if (Math.abs(cx - placed.x) < BADGE_MIN_DX && Math.abs(cy - placed.y) < BADGE_MIN_DY) {
                    return true;
                  }
                }
                for (const [otherId, otherPt] of playerMap.entries()) {
                  if (otherId === selfId) continue;
                  // Prevent an overhead label from sitting on top of another player's foot spotlight ring or lower body
                  if (Math.abs(cx - otherPt.x) < 65 && cy >= otherPt.y - 60 && cy <= otherPt.y + 24) {
                    return true;
                  }
                }
                return false;
              };

              const CANDIDATE_OFFSETS: [number, number][] = [
                [0, 0],
                [0, -38],
                [0, 36],
                [-65, -38],
                [65, -38],
                [0, -68],
                [-65, 36],
                [65, 36],
                [0, -95],
              ];

              for (const cand of deduplicatedLabels) {
                let placedSlot: { x: number; y: number } | null = null;
                for (const [ox, oy] of CANDIDATE_OFFSETS) {
                  const testX = Math.min(width - 135, Math.max(135, cand.x + ox));
                  const testY = Math.min(height - 45, Math.max(50, cand.y + oy));
                  if (!overlapsAnyPlacedOrPlayerFoot(testX, testY, cand.id)) {
                    placedSlot = { x: testX, y: testY };
                    break;
                  }
                }

                if (placedSlot) {
                  finalPlacedLabels.push({
                    ...cand,
                    x: placedSlot.x,
                    y: placedSlot.y,
                  });
                }
              }

              return (
                <React.Fragment key="tacticdive-frameless-names">
                  {finalPlacedLabels.slice(0, 6).map((item, idx) => {
                    const isHl = Boolean(item.isHighlighted);
                    const hTag = activeHighlightTags.find((t) => matchPlayerForHighlight(t.target) === item.id);

                    // Intelligent Staged Entry & Exit Per Player Role
                    let labelEntryBase = graphicsStart;
                    let labelExitFade = freezeExitFade;

                    if (isHl && hTag?.start_frame !== undefined) {
                      labelEntryBase = hTag.start_frame;
                      labelExitFade = 1.0;
                    } else if (isPhase3 && item.id === activeSpotlightId) {
                      labelEntryBase = graphicsStart;
                      labelExitFade = seq3RingFade * 0.95;
                    } else if (activeOptionReceiverIds.has(item.id)) {
                      labelEntryBase = earliestOptStart;
                      labelExitFade = optOpacity;
                    } else if (activeExecutedReceiverIds.has(item.id)) {
                      labelEntryBase = executedStart;
                      labelExitFade = executedOpacity;
                    } else if (activePresserIds.has(item.id)) {
                      labelEntryBase = earliestRunStart;
                      labelExitFade = freezeExitFade;
                    } else {
                      labelEntryBase = graphicsStart;
                      labelExitFade = freezeExitFade;
                    }

                    const labelLocalFrame = Math.max(0, frame - labelEntryBase);

                    // 1. Fluid Entry Pop Spring with energetic elastic overshoot
                    const popSpring = spring({
                      frame: labelLocalFrame,
                      fps,
                      config: {
                        damping: 11.5,
                        stiffness: 160,
                        mass: 0.5,
                      },
                    });

                    // Fluid upward drift on entry: rises 10px smoothly into position
                    const entryYOffset = interpolate(popSpring, [0, 1], [10, 0], {
                      extrapolateRight: 'clamp',
                    });

                    // Fluid fade-in ramp
                    const entryAlpha = interpolate(popSpring, [0, 0.35], [0, 1], {
                      extrapolateRight: 'clamp',
                    });

                    // 2. Highlight Pop & Pulse (Triggered when AI commentary targets this player)
                    let hlScaleBonus = 1.0;
                    let hlPulseGlow = 0;
                    if (isHl && hTag) {
                      const hlLocal = Math.max(0, frame - (hTag.start_frame ?? labelEntryBase));
                      const hlSpring = spring({
                        frame: hlLocal,
                        fps,
                        config: {
                          damping: 10,
                          stiffness: 175,
                          mass: 0.45,
                        },
                      });
                      hlScaleBonus = interpolate(hlSpring, [0, 1], [1.0, 1.22], {
                        extrapolateRight: 'clamp',
                      });
                      hlPulseGlow = Math.sin(frame * 0.22) * 0.5 + 0.5;
                    }

                    // 3. Stable AR Turf-Tracking with zero artificial jiggling
                    const organicFloat = 0;

                    // 4. Exit Deflate & Fade
                    const exitScale = 0.88 + 0.12 * freezeExitFade;
                    const totalScale = popSpring * (isHl ? hlScaleBonus : 1.0) * exitScale;
                    const totalOpacity = entryAlpha * labelExitFade;

                    if (totalOpacity <= 0.01 || totalScale <= 0.02) {
                      return null;
                    }

                    // Parse player name vs tactical role abbreviation (e.g. "Mbappé" and "(CF)")
                    const nameParts = item.text.match(/^(.*?)(\s*\([A-Z0-9/+-]+\))$/);
                    const mainName = nameParts ? nameParts[1].trim() : item.text;
                    const roleTag = nameParts ? nameParts[2].trim() : '';

                    return (
                      <div
                        key={`overhead-label-${item.id}`}
                        style={{
                          position: 'absolute',
                          left: item.x,
                          top: item.y,
                          transformOrigin: 'bottom center',
                          transform: `translate(-50%, -100%) translateY(${(entryYOffset + organicFloat).toFixed(2)}px) scale(${totalScale.toFixed(3)})`,
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          pointerEvents: 'none',
                          userSelect: 'none',
                          zIndex: isHl ? 30 : 20,
                          opacity: Math.min(1.0, totalOpacity),
                        }}
                      >
                        {/* Bust Player Cutout - Chest sits BEHIND the rectangle tag */}
                        <div
                          style={{
                            position: 'relative',
                            width: isHl ? 52 : 44,
                            height: item.portraitUrl ? (isHl ? 54 : 46) : 0,
                            marginBottom: item.portraitUrl ? (isHl ? -15 : -13) : 0, // Bottom of bust occluded behind the name box
                            zIndex: 1, // Sits BEHIND the name rectangle (zIndex: 2)
                            display: 'flex',
                            alignItems: 'flex-start',
                            justifyContent: 'center',
                            overflow: 'visible',
                            pointerEvents: 'none',
                          }}
                        >
                          {item.portraitUrl && (
                            <img
                              src={item.portraitUrl}
                              alt={mainName}
                              style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'contain',
                                objectPosition: 'top center',
                                filter: isHl
                                  ? `drop-shadow(0 0 ${6 + 4 * hlPulseGlow}px rgba(0, 229, 255, 0.7)) drop-shadow(0 3px 6px rgba(0,0,0,0.85))`
                                  : 'drop-shadow(0 3px 6px rgba(0,0,0,0.85))',
                              }}
                              onError={(e) => {
                                e.currentTarget.style.display = 'none';
                                if (e.currentTarget.parentElement) {
                                  e.currentTarget.parentElement.style.height = '0px';
                                  e.currentTarget.parentElement.style.marginBottom = '0px';
                                }
                              }}
                            />
                          )}

                          {/* SofaScore Event Badges floating above / beside player's head */}
                          {/* 1. Goals: Stacked Soccer Balls (⚽) */}
                          {item.visibleGoals > 0 && (
                            <div
                              style={{
                                position: 'absolute',
                                top: item.portraitUrl ? -8 : -14,
                                right: -8,
                                display: 'flex',
                                flexDirection: 'column-reverse',
                                alignItems: 'center',
                                zIndex: 4,
                              }}
                            >
                              {Array.from({ length: Math.min(3, item.visibleGoals) }).map((_, bIdx) => (
                                <div
                                  key={`ball-${bIdx}`}
                                  style={{
                                    width: 15,
                                    height: 15,
                                    borderRadius: '50%',
                                    backgroundColor: '#ffffff',
                                    border: '1px solid rgba(0, 0, 0, 0.4)',
                                    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.7)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: 9,
                                    lineHeight: 1,
                                    marginTop: bIdx > 0 ? -4 : 0,
                                  }}
                                >
                                  ⚽
                                </div>
                              ))}
                              {item.visibleGoals > 3 && (
                                <span style={{ fontSize: 9, fontWeight: 900, color: '#ffffff', textShadow: '0 1px 2px #000' }}>
                                  x{item.visibleGoals}
                                </span>
                              )}
                            </div>
                          )}

                          {/* 2. Assists: Football Boot / Shoe Badge (👟) */}
                          {item.visibleAssists > 0 && (
                            <div
                              style={{
                                position: 'absolute',
                                top: item.portraitUrl ? (item.visibleGoals > 0 ? -8 : -6) : -14,
                                left: -8,
                                width: 15,
                                height: 15,
                                borderRadius: '50%',
                                backgroundColor: '#ffffff',
                                border: '1px solid rgba(0, 0, 0, 0.4)',
                                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.7)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: 9,
                                lineHeight: 1,
                                zIndex: 4,
                              }}
                              title={`${item.visibleAssists} Assist${item.visibleAssists > 1 ? 's' : ''}`}
                            >
                              👟
                            </div>
                          )}

                          {/* 3. Cards: Yellow Card or Red Card */}
                          {item.hasRedCard ? (
                            <div
                              style={{
                                position: 'absolute',
                                bottom: item.portraitUrl ? 4 : 2,
                                right: -6,
                                width: 9,
                                height: 13,
                                borderRadius: 2,
                                backgroundColor: '#ef4444',
                                border: '1px solid rgba(0,0,0,0.6)',
                                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.7)',
                                zIndex: 4,
                              }}
                            />
                          ) : item.hasYellowCard ? (
                            <div
                              style={{
                                position: 'absolute',
                                bottom: item.portraitUrl ? 4 : 2,
                                right: -6,
                                width: 9,
                                height: 13,
                                borderRadius: 2,
                                backgroundColor: '#facc15',
                                border: '1px solid rgba(0,0,0,0.6)',
                                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.7)',
                                zIndex: 4,
                              }}
                            />
                          ) : null}

                          {/* 4. Substituted Out (Red down arrow) */}
                          {item.isSubbedOut && (
                            <div
                              style={{
                                position: 'absolute',
                                top: item.portraitUrl ? 0 : -10,
                                left: -6,
                                width: 13,
                                height: 13,
                                borderRadius: '50%',
                                backgroundColor: '#ffffff',
                                border: '1px solid rgba(239, 68, 68, 0.8)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                boxShadow: '0 1px 3px rgba(0,0,0,0.6)',
                                zIndex: 4,
                              }}
                            >
                              <span style={{ color: '#ef4444', fontSize: 9, fontWeight: 900, lineHeight: 1 }}>↓</span>
                            </div>
                          )}

                          {/* 5. Captain Badge (Yellow C) */}
                          {item.isCaptain && (
                            <div
                              style={{
                                position: 'absolute',
                                bottom: item.portraitUrl ? 4 : 2,
                                left: -6,
                                width: 13,
                                height: 13,
                                borderRadius: '50%',
                                backgroundColor: '#facc15',
                                border: '1px solid rgba(0, 0, 0, 0.4)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                boxShadow: '0 1px 3px rgba(0,0,0,0.6)',
                                zIndex: 4,
                              }}
                            >
                              <span style={{ color: '#000000', fontSize: 8, fontWeight: 900, lineHeight: 1 }}>C</span>
                            </div>
                          )}
                        </div>

                        {/* Name & Role Pill Container - Sits in FRONT of bust (zIndex 2) */}
                        <div
                          style={{
                            position: 'relative',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 5,
                            padding: isHl ? '3px 8px 3px 8px' : '2.5px 7px 2.5px 7px',
                            borderRadius: 4,
                            background: item.kit?.bg || (isHl
                              ? 'linear-gradient(180deg, rgba(8, 24, 44, 0.94) 0%, rgba(4, 14, 28, 0.96) 100%)'
                              : 'linear-gradient(180deg, rgba(13, 20, 32, 0.92) 0%, rgba(8, 14, 24, 0.94) 100%)'),
                            backdropFilter: 'blur(8px)',
                            WebkitBackdropFilter: 'blur(8px)',
                            border: item.kit?.isStriped ? '2px solid transparent' : (isHl
                              ? `1.5px solid ${item.kit?.borderColor || 'rgba(0, 229, 255, 0.95)'}`
                              : `1px solid ${item.kit?.borderColor || 'rgba(255, 255, 255, 0.35)'}`),
                            boxShadow: isHl
                              ? `0 0 ${12 + 6 * hlPulseGlow}px rgba(0, 229, 255, ${0.55 + 0.25 * hlPulseGlow}), 0 4px 14px rgba(0, 0, 0, 0.85)`
                              : '0 3px 10px rgba(0, 0, 0, 0.70)',
                            fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                            WebkitFontSmoothing: 'antialiased',
                            MozOsxFontSmoothing: 'grayscale',
                            fontSize: isHl ? 13 : 12,
                            fontWeight: isHl ? 800 : 700,
                            letterSpacing: '0.04em',
                            lineHeight: 1.25,
                            color: item.kit?.textColor || '#ffffff',
                            textShadow: (item.kit?.textColor === '#ffffff' || !item.kit?.textColor) ? '0 1px 3px rgba(0, 0, 0, 0.95)' : 'none',
                            whiteSpace: 'nowrap',
                            zIndex: 2, // Sits in FRONT of the chest!
                          }}
                        >
                          {/* SVG Alternating Segmented Border when kit is striped (e.g. Barcelona Red/Blue) */}
                          {item.kit?.isStriped && (
                            <svg
                              style={{
                                position: 'absolute',
                                inset: 0,
                                width: '100%',
                                height: '100%',
                                pointerEvents: 'none',
                                overflow: 'visible',
                                borderRadius: 4,
                              }}
                            >
                              <rect
                                x="1"
                                y="1"
                                rx="4"
                                ry="4"
                                fill="none"
                                stroke={item.kit.stripeColor1 || '#db0030'}
                                strokeWidth="2"
                                strokeDasharray="10 10"
                                strokeDashoffset="0"
                                style={{
                                  width: 'calc(100% - 2px)',
                                  height: 'calc(100% - 2px)',
                                }}
                              />
                              <rect
                                x="1"
                                y="1"
                                rx="4"
                                ry="4"
                                fill="none"
                                stroke={item.kit.stripeColor2 || '#004d98'}
                                strokeWidth="2"
                                strokeDasharray="10 10"
                                strokeDashoffset="10"
                                style={{
                                  width: 'calc(100% - 2px)',
                                  height: 'calc(100% - 2px)',
                                }}
                              />
                            </svg>
                          )}

                          {/* Player Name */}
                          <span style={{ textTransform: 'uppercase' }}>{mainName}</span>

                          {/* Role Tag */}
                          {roleTag && (
                            <span
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                padding: '1px 4.5px',
                                borderRadius: 3,
                                background: item.kit?.roleBg || (isHl ? 'rgba(0, 229, 255, 0.18)' : 'rgba(255, 255, 255, 0.12)'),
                                border: item.kit?.isStriped ? 'none' : (isHl ? '1px solid rgba(0, 229, 255, 0.50)' : '1px solid rgba(255, 255, 255, 0.22)'),
                                color: item.kit?.roleTextColor || (isHl ? '#38bdf8' : '#cbd5e1'),
                                fontSize: '0.82em',
                                fontWeight: 750,
                                letterSpacing: '0.04em',
                              }}
                            >
                              {roleTag.replace(/[()]/g, '')}
                            </span>
                          )}

                          {/* Live SofaScore Rating Pill */}
                          {item.rating !== undefined && (
                            <span
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                padding: '1px 5px',
                                borderRadius: 3,
                                backgroundColor: item.ratingColor || getSofaScoreRatingColor(Number(item.rating)),
                                color: '#ffffff',
                                fontSize: '0.85em',
                                fontWeight: 900,
                                letterSpacing: '0.02em',
                                boxShadow: '0 1px 3px rgba(0, 0, 0, 0.5)',
                                marginLeft: 2,
                                textShadow: '0 1px 2px rgba(0, 0, 0, 0.6)',
                              }}
                            >
                              {Number(item.rating).toFixed(1)}
                            </span>
                          )}
                        </div>

                        {/* Precision downward needle pointer connecting badge to player */}
                        <div
                          style={{
                            position: 'absolute',
                            left: '50%',
                            top: '100%',
                            transform: 'translateX(-50%)',
                            width: 1,
                            height: 7,
                            background: isHl ? 'linear-gradient(to bottom, #00e5ff, rgba(0, 229, 255, 0.35))' : 'linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.2))',
                            boxShadow: isHl ? '0 0 3px #00e5ff' : 'none',
                          }}
                        >
                          <div
                            style={{
                              position: 'absolute',
                              bottom: -2,
                              left: -1.5,
                              width: 4,
                              height: 4,
                              borderRadius: '50%',
                              backgroundColor: isHl ? '#00e5ff' : '#ffffff',
                              boxShadow: isHl ? '0 0 5px #00e5ff' : '0 1px 3px rgba(0, 0, 0, 0.6)',
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </React.Fragment>
              );
            })()}
          </AbsoluteFill>
        )}
      </SplitCanvasCameraRig>

      {/* 4. Human-Readable Tactical Phase Pill (Top-Left HUD) - Suppressed on close-ups / non-tactical shots */}
      {!renderTurfOnly && effectiveIsTactical && !isCloseUp && (() => {
        // Resolve clean, human-readable tactical phase title (strictly removes debug strings like CLIP_10, etc.)
        const resolvePhaseTitle = (): string | null => {
          const raw = String(title || '').trim();
          const upper = raw.toUpperCase();

          // If already clean human-readable and doesn't contain debug tokens
          if (raw && !upper.includes('CLIP_') && !upper.includes('UNKNOWN') && !upper.includes('SHOT_')) {
            return upper;
          }

          // Map context, play_type, and tactical zone into authentic television broadcast phase titles
          const pType = String(causeTel?.play_type || '').toUpperCase();
          const thesis = String(causeTel?.thesis || '').toLowerCase();
          const zone = String(causeTel?.tactical_zone || '').toLowerCase();

          if (pType.includes('CORNER') || pType.includes('SET_PIECE') || thesis.includes('corner')) {
            return 'PHASE: SET PIECE DELIVERY';
          }
          if (pType.includes('BUILDUP') || thesis.includes('buildup') || thesis.includes('build-up')) {
            return 'PHASE: DEEP BUILD-UP CIRCULATION';
          }
          if (isActualGoal || upper.includes('GOAL') || thesis.includes('goal')) {
            return 'PHASE: FINAL THIRD CREATION';
          }
          if (upper.includes('TACTICAL_SUB') || thesis.includes('sub')) {
            return 'PHASE: TACTICAL REORGANIZATION';
          }
          if (upper.includes('YELLOW_CARD') || thesis.includes('card') || thesis.includes('foul')) {
            return 'PHASE: MIDFIELD DUEL & CONTEST';
          }
          if (thesis.includes('transition') || thesis.includes('counter')) {
            return 'PHASE: ATTACKING TRANSITION';
          }
          if (thesis.includes('press')) {
            return 'PHASE: HIGH PRESSING TRAP';
          }
          if (zone.includes('flank') || zone.includes('wing') || thesis.includes('flank') || thesis.includes('wing')) {
            return 'PHASE: WIDE FLANK OVERLOAD';
          }
          if (zone.includes('defensive')) {
            return 'PHASE: LOW BLOCK DEFENSIVE SHAPE';
          }

          return 'PHASE: MIDFIELD PROGRESSION';
        };

        const cleanPhase = resolvePhaseTitle();
        if (!cleanPhase) return null;

        const cleanSubtitle = subtitle && !subtitle.toUpperCase().includes('CLIP_') && !subtitle.toUpperCase().includes('UNKNOWN')
          ? subtitle.toUpperCase()
          : null;

        return (
          <div
            style={{
              position: 'absolute',
              top: 165,
              left: 50,
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              gap: 8,
              background: 'rgba(10, 15, 26, 0.70)',
              borderLeft: '3px solid #00e676',
              borderTop: '1px solid rgba(255, 255, 255, 0.12)',
              borderRight: '1px solid rgba(255, 255, 255, 0.12)',
              borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
              padding: '4px 12px',
              borderRadius: '0 4px 4px 0',
              boxShadow: '0 4px 14px rgba(0,0,0,0.45)',
              backdropFilter: 'blur(6px)',
              zIndex: 90,
            }}
          >
            <div style={{ fontSize: 11, fontWeight: 800, color: '#ffffff', letterSpacing: '0.06em' }}>
              {cleanPhase}
            </div>
            {cleanSubtitle && (
              <div style={{ fontSize: 10, fontWeight: 600, color: '#00e676', letterSpacing: '0.04em' }}>
                // {cleanSubtitle}
              </div>
            )}
          </div>
        );
      })()}

      {/* 5. Dynamic Player SofaScore HUD & Match Stats Card or Full Heatmap Modal triggered by AI Commentary */}
      {!renderTurfOnly && (() => {
        const activeStatsTag = statsTags.find((t) => {
          const s = Math.max(t.start_frame ?? K_video, K_video);
          const d = t.duration_frames ?? 150;
          // Stat card must dismiss by the end of narration beat 3, never linger in dead silence
          const e = Math.min(b3End, Math.max(t.end_frame ?? 0, s + d));
          return frame >= s && frame <= e;
        });
        if (!activeStatsTag) return null;

        const pStats: PlayerStatsData = activeStatsTag.player_stats || {
          playerName: activeStatsTag.player_name || activeStatsTag.target || 'Player Analysis',
          shortName: (activeStatsTag.player_name || activeStatsTag.target || 'Player').split(' ').slice(-1)[0],
          number: activeStatsTag.player_number || 10,
          role: activeStatsTag.player_role || 'MID',
          team: 'away',
          clubName: 'MATCH TELEMETRY',
          rating: 7.6,
          ratingColor: '#10b981',
          stats: {
            'Passing': '91%',
            'Duels Won': '75%',
            'Top Speed': '33.5 km/h',
          },
          heatmap: {
            avg_x: 52,
            avg_y: 50,
            zone: 'Central Channel',
          },
        };

        const isModalHeatmap =
          activeStatsTag.type === 'heatmap' ||
          activeStatsTag.type === 'heatmap_comparison' ||
          activeStatsTag.type === 'comparison_heatmap' ||
          activeStatsTag.type === 'average_position' ||
          activeStatsTag.type === 'data_card' ||
          activeStatsTag.type === 'match_heatmap';

        if (isModalHeatmap) {
          const targetStr = activeStatsTag.target || '';
          const primaryName = targetStr.includes(':') ? targetStr.split(':')[0].trim() : (pStats.playerName || targetStr || catalyst_player?.name || 'Player Analysis');
          const secondaryName = targetStr.includes(':') ? targetStr.split(':')[1].trim() : undefined;

          // Universal Country Code & Flag Resolver
          const resolveCountryFlag = (_name: string, codeHint?: string) => {
            if (codeHint) {
              const code = codeHint.toUpperCase();
              const flagDict: Record<string, string> = {
                FR: '🇫🇷', BR: '🇧🇷', EN: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', CI: '🇨🇮', ES: '🇪🇸',
                IT: '🇮🇹', DE: '🇩🇪', NL: '🇳🇱', PT: '🇵🇹', AR: '🇦🇷',
                UY: '🇺🇾', HR: '🇭🇷', BE: '🇧🇪', TR: '🇹🇷', NO: '🇳🇴',
                US: '🇺🇸', JP: '🇯🇵', KR: '🇰🇷', MA: '🇲🇦', SN: '🇸🇳',
                DK: '🇩🇰', SE: '🇸🇪', PL: '🇵🇱', CH: '🇨🇭', AT: '🇦🇹',
                NG: '🇳🇬', GH: '🇬🇭', EG: '🇪🇬', DZ: '🇩🇿', CM: '🇨🇲',
                CO: '🇨🇴', CL: '🇨🇱', MX: '🇲🇽', CA: '🇨🇦', RS: '🇷🇸',
              };
              if (flagDict[code]) return { code, flag: flagDict[code] };
            }
            return { code: 'INT', flag: '🌐' };
          };

          const pCountry = resolveCountryFlag(primaryName, (activeStatsTag as any).country_code || (pStats as any).countryCode);

          const pProfile: PlayerHeatmapProfile = {
            playerName: primaryName,
            shortName: (primaryName).split(' ').slice(-1)[0],
            number: pStats.number || 10,
            role: pStats.role || pStats.position || 'MID',
            position: pStats.position || 'MID',
            team: pStats.team || 'away',
            clubName: pStats.clubName || (causeTel as any)?.away_team || 'MATCH TELEMETRY',
            countryCode: pCountry.code,
            countryFlagEmoji: (activeStatsTag as any).country_flag || pCountry.flag,
            avatarUrl: pStats.portraitUrl || (activeStatsTag as any).avatar_url || (activeStatsTag as any).portrait_url,
            seasonOrMatch: (activeStatsTag as any).season_or_match || 'Current Season',
            rating: pStats.rating || 7.5,
            ratingColor: pStats.ratingColor || '#22c55e',
            avgPosition: (pStats as any).avgPosition || (pStats.heatmap ? { x: pStats.heatmap.avg_x ?? 50, y: pStats.heatmap.avg_y ?? 50 } : undefined),
            heatmapZone: pStats.heatmap?.zone || 'Attacking Third / Operating Half-Space',
            tacticalNote: (activeStatsTag as any).tactical_note || (pStats as any).tacticalNote,
            heatPoints: (activeStatsTag as any).heat_points || (pStats as any).heatPoints || (pStats.heatmap as any)?.heat_points,
          };

          const compProfile: PlayerHeatmapProfile | undefined = (activeStatsTag as any).comparison_player ? {
            playerName: (activeStatsTag as any).comparison_player.playerName || 'Comparison Player',
            shortName: ((activeStatsTag as any).comparison_player.playerName || 'Player').split(' ').slice(-1)[0],
            number: (activeStatsTag as any).comparison_player.number || 11,
            role: (activeStatsTag as any).comparison_player.role || 'ATT',
            position: (activeStatsTag as any).comparison_player.position || 'ATT',
            clubName: (activeStatsTag as any).comparison_player.clubName || pStats.clubName || 'MATCH TELEMETRY',
            countryCode: (activeStatsTag as any).comparison_player.countryCode || 'INT',
            countryFlagEmoji: (activeStatsTag as any).comparison_player.countryFlagEmoji || '🌐',
            avatarUrl: (activeStatsTag as any).comparison_player.avatarUrl || (activeStatsTag as any).comparison_player.portraitUrl,
            seasonOrMatch: (activeStatsTag as any).comparison_player.seasonOrMatch || 'Current Season',
            rating: (activeStatsTag as any).comparison_player.rating || 7.5,
            heatPoints: (activeStatsTag as any).comparison_player.heatPoints || (activeStatsTag as any).comparison_player.heat_points,
          } : (secondaryName ? {
            playerName: secondaryName,
            shortName: secondaryName.split(' ').slice(-1)[0],
            number: 11,
            role: 'ATT',
            position: 'ATT',
            clubName: pStats.clubName || 'MATCH TELEMETRY',
            countryCode: resolveCountryFlag(secondaryName).code,
            countryFlagEmoji: resolveCountryFlag(secondaryName).flag,
            seasonOrMatch: 'Current Season',
            rating: 7.5,
          } : undefined);

          const cardStart = Math.max(activeStatsTag.start_frame ?? K_video, K_video);
          const rawDuration = activeStatsTag.duration_frames || (activeStatsTag.end_frame ? activeStatsTag.end_frame - cardStart : 150);
          const maxAllowedDuration = Math.max(45, b3End - cardStart);
          const cardDuration = Math.min(rawDuration, maxAllowedDuration);

          return (
            <PlayerHeatmapModal
              player={pProfile}
              comparisonPlayer={compProfile}
              startFrame={cardStart}
              durationFrames={cardDuration}
              title={(activeStatsTag as any).title || (activeStatsTag.type === 'average_position' ? 'Average Position Map' : (activeStatsTag.type === 'match_heatmap' || activeStatsTag.type === 'stat_card' || activeStatsTag.type === 'sofascore' ? 'Match Telemetry & Heatmap' : 'Season heatmap'))}
              metricType={activeStatsTag.type === 'average_position' ? 'avg_position' : 'season_heatmap'}
              placement="bottom-right"
            />
          );
        }

        const isHeatmapFocus = activeStatsTag.type === 'heatmap' || activeStatsTag.type === 'average_position';

        const sCardStart = Math.max(activeStatsTag.start_frame ?? K_video, K_video);
        const sCardRawDuration = activeStatsTag.duration_frames || (activeStatsTag.end_frame ? activeStatsTag.end_frame - sCardStart : 150);
        const sCardDuration = Math.min(sCardRawDuration, Math.max(45, b3End - sCardStart));

        return (
          <PlayerSofaScoreCard
            playerStats={pStats}
            startFrame={sCardStart}
            durationFrames={sCardDuration}
            bottom={50}
            left={60}
            highlightHeatmap={isHeatmapFocus}
          />
        );
      })()}

      {/* 6. Broadcast Tactical Mistake Red Screen Light-Up Overlay */}
      {currentMistakeFlashIntensity > 0.005 && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            zIndex: 88,
            background: `radial-gradient(ellipse at center, rgba(239, 68, 68, ${(currentMistakeFlashIntensity * 0.55).toFixed(3)}) 0%, rgba(225, 29, 72, ${(currentMistakeFlashIntensity * 0.85).toFixed(3)}) 60%, rgba(159, 18, 57, ${(currentMistakeFlashIntensity * 1.20).toFixed(3)}) 100%)`,
            boxShadow: `inset 0 0 ${Math.round(110 * (currentMistakeFlashIntensity / 0.48))}px rgba(225, 29, 72, ${(currentMistakeFlashIntensity * 1.5).toFixed(3)})`,
          }}
        />
      )}
    </AbsoluteFill>
  );
};

export const Tactical3PhaseTimeline: React.FC<TacticalTelestrationProps> = (props) => {
  return <TacticalTelestrationOverlay {...props} />;
};
