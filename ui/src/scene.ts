// Scenes as exported by eval/export_viewer.py, and their geometry: colored
// line segments and points, ready for the renderer. No DOM or WebGL here.

import { bounds, hex, ramp, type RGB, type Vec3 } from "./math.js";

export interface Track {
  name: string;
  role: "truth" | "estimate" | "reference";
  t: number[];
  xyz: number[]; // flat x, y, z
  error?: (number | null)[];
}

export interface Box {
  min: number[];
  max: number[];
}

export interface Scene {
  name: string;
  kind: "euroc" | "mission";
  metrics: Record<string, unknown>;
  tracks: Track[];
  loops: { a: number[]; b: number[]; good: boolean; t?: number }[]; // t: when the loop was closed
  cloud?: number[];
  paths?: number[][];
  room?: number[];
  boxes?: Box[];
}

export interface Layers {
  truth: boolean;
  estimate: boolean;
  reference: boolean;
  loops: boolean;
  cloud: boolean;
  world: boolean;
  colorByError: boolean;
}

export const COLORS = {
  truth: hex("#a1a7b3"),
  estimate: hex("#4c9ee8"),
  reference: hex("#c9a227"),
  path: hex("#e07b39"),
  loopGood: hex("#3fb37f"),
  loopBad: hex("#e5484d"),
  world: hex("#6e7681"),
  box: hex("#9b7fd4"),
  cursor: hex("#ffffff"),
};

/** One hue, dark to light: brighter is a larger error on the dark canvas. */
export const ERROR_RAMP: RGB[] = [hex("#1e3a5f"), hex("#2f6db0"), hex("#5aa5f0"), hex("#a9d4ff"), hex("#f2f8ff")];

export interface Geometry {
  lines: Float32Array; // per vertex: x y z r g b
  points: Float32Array; // per vertex: x y z r g b; the time cursors last
  cursors: number;
  min: Vec3;
  max: Vec3;
}

/** Number of samples of a track with time <= t (binary search). */
export function samplesUntil(track: Track, t: number): number {
  let lo = 0, hi = track.t.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (track.t[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function duration(scene: Scene): number {
  return Math.max(0, ...scene.tracks.map((tr) => (tr.t.length ? tr.t[tr.t.length - 1] : 0)));
}

/** The 12 edges of an axis-aligned box as 24 endpoints. */
export function boxEdges(min: number[], max: number[]): Vec3[] {
  const c = (i: number): Vec3 => [i & 1 ? max[0] : min[0], i & 2 ? max[1] : min[1], i & 4 ? max[2] : min[2]];
  const out: Vec3[] = [];
  for (let i = 0; i < 8; i++)
    for (const b of [1, 2, 4])
      if (!(i & b)) out.push(c(i), c(i | b));
  return out;
}

/** Error statistics of a track (meters): RMS and maximum over the known samples. */
export function errorStats(track: Track): { rms: number; max: number; n: number } | null {
  if (!track.error) return null;
  let s = 0, m = 0, n = 0;
  for (const e of track.error)
    if (e !== null && isFinite(e)) {
      s += e * e;
      m = Math.max(m, e);
      n++;
    }
  return n ? { rms: Math.sqrt(s / n), max: m, n } : null;
}

class Builder {
  lines: number[] = [];
  points: number[] = [];
  seg(a: ArrayLike<number>, b: ArrayLike<number>, ca: RGB, cb: RGB = ca) {
    this.lines.push(a[0], a[1], a[2], ca[0], ca[1], ca[2], b[0], b[1], b[2], cb[0], cb[1], cb[2]);
  }
  point(p: ArrayLike<number>, c: RGB) {
    this.points.push(p[0], p[1], p[2], c[0], c[1], c[2]);
  }
}

/**
 * Build the geometry of a scene up to time `t` (seconds from the start; the
 * whole trajectories when t is Infinity). `errorScale`: the error (m) shown
 * at the bright end of the ramp.
 */
export function buildGeometry(scene: Scene, layers: Layers, t = Infinity, errorScale = 0.2): Geometry {
  const g = new Builder();
  const all: number[][] = [];
  const cursors: number[][] = [];
  for (const tr of scene.tracks) {
    if (!layers[tr.role]) continue;
    all.push(tr.xyz);
    const n = Math.min(samplesUntil(tr, t), tr.xyz.length / 3);
    const color = (i: number): RGB => {
      if (tr.role === "estimate" && layers.colorByError && tr.error) {
        const e = tr.error[i];
        return e === null ? COLORS.estimate : ramp(ERROR_RAMP, e / errorScale);
      }
      return COLORS[tr.role];
    };
    for (let i = 1; i < n; i++) g.seg(tr.xyz.slice(3 * i - 3, 3 * i), tr.xyz.slice(3 * i, 3 * i + 3), color(i - 1), color(i));
    if (n > 0 && isFinite(t)) cursors.push(tr.xyz.slice(3 * n - 3, 3 * n));
  }
  if (layers.loops)
    for (const l of scene.loops) {
      if (l.t !== undefined && l.t > t) continue;
      const c = l.good ? COLORS.loopGood : COLORS.loopBad;
      g.seg(l.a, l.b, c);
    }
  if (layers.world) {
    for (const p of scene.paths ?? [])
      for (let i = 3; i + 2 < p.length; i += 3) g.seg(p.slice(i - 3, i), p.slice(i, i + 3), COLORS.path);
    if (scene.room) for (const [a, b] of pairs(boxEdges([0, 0, 0], scene.room))) g.seg(a, b, COLORS.world);
    for (const box of scene.boxes ?? []) for (const [a, b] of pairs(boxEdges(box.min, box.max))) g.seg(a, b, COLORS.box);
    if (scene.room) all.push([0, 0, 0, ...scene.room]);
  }
  if (layers.cloud && scene.cloud) {
    const c = scene.cloud;
    const { min, max } = bounds(c);
    const h = max[2] - min[2] || 1;
    for (let i = 0; i + 2 < c.length; i += 3) {
      const z = (c[i + 2] - min[2]) / h;
      const v = 0.25 + 0.35 * z; // a dim gray, lighter with height
      g.point([c[i], c[i + 1], c[i + 2]], [v, v, v * 1.08]);
    }
  }
  for (const c of cursors) g.point(c, COLORS.cursor);
  const b = bounds(...(all.length ? all : scene.tracks.map((tr) => tr.xyz)));
  return { lines: new Float32Array(g.lines), points: new Float32Array(g.points), cursors: cursors.length, min: b.min, max: b.max };
}

function* pairs(v: Vec3[]): Generator<[Vec3, Vec3]> {
  for (let i = 0; i + 1 < v.length; i += 2) yield [v[i], v[i + 1]];
}
