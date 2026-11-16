import { test } from "node:test";
import assert from "node:assert/strict";

import { bounds, identity, lookAt, multiply, OrbitCamera, perspective, ramp, transformPoint, type Vec3 } from "../src/math.js";
import { boxEdges, buildGeometry, duration, errorStats, samplesUntil, type Layers, type Scene } from "../src/scene.js";

const close = (a: number, b: number, eps = 1e-5) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test("identity is neutral for multiply", () => {
  const m = perspective(1, 1.5, 0.1, 100);
  assert.deepEqual([...multiply(identity(), m)], [...m]);
  assert.deepEqual([...multiply(m, identity())], [...m]);
});

test("lookAt maps the target onto the negative z axis", () => {
  const eye: Vec3 = [3, -2, 5], target: Vec3 = [1, 1, 0];
  const p = transformPoint(lookAt(eye, target, [0, 0, 1]), target);
  close(p[0], 0);
  close(p[1], 0);
  close(p[2], -Math.hypot(2, 3, 5));
});

test("projection puts points between near and far into [-1, 1] depth", () => {
  const P = perspective(Math.PI / 3, 1, 1, 10);
  close(transformPoint(P, [0, 0, -1])[2], -1);
  close(transformPoint(P, [0, 0, -10])[2], 1);
});

test("the orbit camera looks at its target and fit frames a box", () => {
  const c = new OrbitCamera();
  c.fit([0, 0, 0], [4, 2, 2]);
  assert.deepEqual(c.target, [2, 1, 1]);
  const p = transformPoint(c.matrix(1.5), c.target);
  close(p[0], 0);
  close(p[1], 0);
  for (const corner of [[0, 0, 0], [4, 2, 2], [4, 0, 2]] as Vec3[]) {
    const q = transformPoint(c.matrix(1), corner);
    assert.ok(Math.abs(q[0]) <= 1 && Math.abs(q[1]) <= 1, `corner ${corner} outside the view`);
  }
  c.rotate(0, 10);
  assert.ok(c.pitch < Math.PI / 2);
});

test("bounds and ramp", () => {
  assert.deepEqual(bounds([1, 2, 3, -1, 5, 0]), { min: [-1, 2, 0], max: [1, 5, 3] });
  assert.deepEqual(ramp([[0, 0, 0], [1, 1, 1]], 0.25), [0.25, 0.25, 0.25]);
  assert.deepEqual(ramp([[0, 0, 0], [1, 1, 1]], 7), [1, 1, 1]);
  assert.deepEqual(ramp([[0, 0, 0], [1, 1, 1]], NaN), [0, 0, 0]);
});

test("a box has 12 edges of the right lengths", () => {
  const e = boxEdges([0, 0, 0], [1, 2, 3]);
  assert.equal(e.length, 24);
  const lengths = [];
  for (let i = 0; i < 24; i += 2) lengths.push(Math.hypot(e[i][0] - e[i + 1][0], e[i][1] - e[i + 1][1], e[i][2] - e[i + 1][2]));
  assert.deepEqual(lengths.sort(), [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
});

const scene: Scene = {
  name: "test",
  kind: "euroc",
  metrics: {},
  tracks: [
    { name: "gt", role: "truth", t: [0, 1, 2, 3], xyz: [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0] },
    { name: "est", role: "estimate", t: [0, 1, 2], xyz: [0, 0, 0, 1, 0.1, 0, 2, 0.2, 0], error: [0, 0.1, null] },
  ],
  loops: [{ a: [0, 0, 0], b: [2, 0, 0], good: true, t: 2 }],
  cloud: [0, 0, 0, 1, 1, 1],
};
const all: Layers = { truth: true, estimate: true, reference: true, loops: true, cloud: true, world: true, colorByError: true };

test("time cursor and duration", () => {
  assert.equal(samplesUntil(scene.tracks[0], 1.5), 2);
  assert.equal(samplesUntil(scene.tracks[0], -1), 0);
  assert.equal(samplesUntil(scene.tracks[0], 3), 4);
  assert.equal(duration(scene), 3);
});

test("geometry: segments, loops, cloud and cursors", () => {
  const g = buildGeometry(scene, all);
  assert.equal(g.lines.length / 6, 2 * (3 + 2 + 1)); // 3 + 2 track segments, 1 loop
  assert.equal(g.points.length / 6, 2);
  assert.equal(g.cursors, 0);
  const g1 = buildGeometry(scene, all, 1.0);
  assert.equal(g1.lines.length / 6, 2 * (1 + 1)); // the loop closes at t = 2: not yet shown
  assert.equal(g1.cursors, 2);
  // the cursors are the last points: at x = 1 on both tracks
  const pts = g1.points;
  assert.equal(pts[pts.length - 6], 1);
  assert.equal(pts[pts.length - 12], 1);
  const off = buildGeometry(scene, { ...all, truth: false, loops: false, cloud: false });
  assert.equal(off.lines.length / 6, 2 * 2);
  assert.equal(off.points.length, 0);
});

test("error statistics skip unknown samples", () => {
  const s = errorStats(scene.tracks[1])!;
  assert.equal(s.n, 2);
  close(s.rms, Math.sqrt(0.01 / 2));
  close(s.max, 0.1);
  assert.equal(errorStats(scene.tracks[0]), null);
});
