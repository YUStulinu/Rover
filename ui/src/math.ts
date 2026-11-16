// Small linear algebra for the viewer: column-major 4x4 matrices (as WebGL
// expects), an orbit camera, bounds and color ramps.

export type Vec3 = [number, number, number];
export type Mat4 = Float32Array;

export function identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** a * b (column-major). */
export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  return out;
}

export function transformPoint(m: Mat4, p: Vec3): Vec3 {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
  const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
  const z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  return [x / w, y / w, z / w];
}

export function perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovy / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export function normalize(a: Vec3): Vec3 {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}

export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const f = normalize(sub(target, eye));
  const s = normalize(cross(f, up));
  const u = cross(s, f);
  const m = identity();
  m[0] = s[0]; m[4] = s[1]; m[8] = s[2];
  m[1] = u[0]; m[5] = u[1]; m[9] = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[12] = -dot(s, eye);
  m[13] = -dot(u, eye);
  m[14] = dot(f, eye);
  return m;
}

/** A camera orbiting a target point; the world's z axis is up. */
export class OrbitCamera {
  target: Vec3 = [0, 0, 0];
  distance = 10;
  yaw = -Math.PI / 4; // around z
  pitch = 0.6; // above the xy plane
  fovy = Math.PI / 4;

  eye(): Vec3 {
    const c = Math.cos(this.pitch);
    return [
      this.target[0] + this.distance * c * Math.cos(this.yaw),
      this.target[1] + this.distance * c * Math.sin(this.yaw),
      this.target[2] + this.distance * Math.sin(this.pitch),
    ];
  }

  rotate(dyaw: number, dpitch: number) {
    this.yaw += dyaw;
    this.pitch = Math.min(1.55, Math.max(-1.55, this.pitch + dpitch));
  }

  zoom(factor: number) {
    this.distance = Math.min(500, Math.max(0.2, this.distance * factor));
  }

  /** Move the target in the view plane by a fraction of the distance. */
  pan(dx: number, dy: number) {
    const e = this.eye();
    const f = normalize(sub(this.target, e));
    const s = normalize(cross(f, [0, 0, 1]));
    const u = cross(s, f);
    const k = this.distance;
    for (let i = 0; i < 3; i++) this.target[i] += k * (-dx * s[i] + dy * u[i]);
  }

  /** Frame a bounding box. */
  fit(min: Vec3, max: Vec3) {
    this.target = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const r = Math.max(0.5, Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2);
    this.distance = r / Math.sin(this.fovy / 2);
  }

  matrix(aspect: number): Mat4 {
    const near = Math.max(0.01, this.distance / 200);
    return multiply(perspective(this.fovy, aspect, near, this.distance * 20 + 50), lookAt(this.eye(), this.target, [0, 0, 1]));
  }
}

/** Bounds of flat xyz arrays. */
export function bounds(...arrays: ArrayLike<number>[]): { min: Vec3; max: Vec3 } {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const a of arrays)
    for (let i = 0; i + 2 < a.length; i += 3)
      for (let k = 0; k < 3; k++) {
        min[k] = Math.min(min[k], a[i + k]);
        max[k] = Math.max(max[k], a[i + k]);
      }
  if (!isFinite(min[0])) return { min: [-1, -1, -1], max: [1, 1, 1] };
  return { min, max };
}

export type RGB = [number, number, number];

export function hex(s: string): RGB {
  const v = parseInt(s.replace("#", ""), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

/** Piecewise-linear color ramp through `stops` for t in [0, 1] (clamped). */
export function ramp(stops: RGB[], t: number): RGB {
  if (!(t > 0)) return stops[0];
  if (t >= 1) return stops[stops.length - 1];
  const x = t * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
