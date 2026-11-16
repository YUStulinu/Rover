// The viewer page: pick a scene, orbit around it, play the trajectories in
// time, switch layers on and off.

import { OrbitCamera } from "./math.js";
import { Renderer } from "./render.js";
import { buildGeometry, COLORS, duration, ERROR_RAMP, errorStats, type Layers, type Scene } from "./scene.js";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("view");
const camera = new OrbitCamera();
let renderer: Renderer;
let scene: Scene | null = null;
let t = Infinity;
let playing = false;
let lastFrame = 0;
const layers: Layers = { truth: true, estimate: true, reference: true, loops: true, cloud: true, world: true, colorByError: false };

const css = (rgb: number[]) => `rgb(${rgb.map((v) => Math.round(v * 255)).join(",")})`;

function background(): [number, number, number] {
  const v = getComputedStyle(document.documentElement).getPropertyValue("--canvas").trim() || "#0f1115";
  const n = parseInt(v.replace("#", ""), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function rebuild() {
  if (!scene) return;
  renderer.upload(buildGeometry(scene, layers, t, Number($<HTMLSelectElement>("errscale").value)));
}

function fmt(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) < 1 ? 3 : 1);
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}

function metricsTable(m: Record<string, unknown>): string {
  const rows = (o: Record<string, unknown>) =>
    Object.entries(o)
      .filter(([, v]) => typeof v !== "object" || v === null)
      .map(([k, v]) => `<tr><th>${k.replace(/_/g, " ")}</th><td>${fmt(v)}</td></tr>`)
      .join("");
  const nested = Object.entries(m).filter(([, v]) => v && typeof v === "object");
  if (nested.length)
    return nested.map(([k, v]) => `<h3>${k}</h3><table>${rows(v as Record<string, unknown>)}</table>`).join("");
  return `<table>${rows(m)}</table>`;
}

function legend(s: Scene) {
  const items: [string, number[], string][] = [];
  for (const tr of s.tracks) {
    const st = errorStats(tr);
    items.push([tr.role, COLORS[tr.role], tr.name + (st ? ` (RMS ${(st.rms * 100).toFixed(1)} cm)` : "")]);
  }
  if (s.loops.length) {
    const good = s.loops.filter((l) => l.good).length;
    items.push(["loops", COLORS.loopGood, `loop closures, correct (${good})`]);
    if (good < s.loops.length) items.push(["loops", COLORS.loopBad, `loop closures, wrong (${s.loops.length - good})`]);
  }
  if (s.cloud) items.push(["cloud", [0.5, 0.5, 0.54], "Leica laser scan of the room"]);
  if (s.paths) items.push(["world", COLORS.path, "planned paths (A*)"]);
  if (s.boxes) items.push(["world", COLORS.box, "obstacles"]);
  let html = items.map(([, c, label]) => `<li><span class="swatch" style="background:${css(c)}"></span>${label}</li>`).join("");
  if (layers.colorByError && s.tracks.some((tr) => tr.error)) {
    const scale = Number($<HTMLSelectElement>("errscale").value) * 100;
    const grad = ERROR_RAMP.map(css).join(",");
    html += `<li class="ramp"><span>0</span><span class="bar" style="background:linear-gradient(90deg,${grad})"></span><span>${scale} cm error</span></li>`;
  }
  $("legend").innerHTML = html;
}

async function load(file: string) {
  const r = await fetch(`data/${file}`);
  scene = (await r.json()) as Scene;
  t = Infinity;
  playing = false;
  $<HTMLButtonElement>("play").textContent = "Play";
  const d = duration(scene);
  const slider = $<HTMLInputElement>("time");
  slider.max = String(d);
  slider.value = String(d);
  $("clock").textContent = `${d.toFixed(1)} s`;
  const g = buildGeometry(scene, layers);
  camera.fit(g.min, g.max);
  if (scene.kind === "mission") camera.pitch = 1.0;
  $("metrics").innerHTML = metricsTable(scene.metrics);
  legend(scene);
  for (const id of ["cloud", "loops", "world"] as const) {
    const has = id === "cloud" ? !!scene.cloud : id === "loops" ? scene.loops.length > 0 : !!(scene.boxes || scene.paths);
    $<HTMLInputElement>(id).closest("label")!.classList.toggle("hidden", !has);
  }
  $<HTMLInputElement>("reference").closest("label")!.classList.toggle("hidden", !scene.tracks.some((tr) => tr.role === "reference"));
  rebuild();
}

function frame(now: number) {
  if (playing && scene) {
    const dt = lastFrame ? (now - lastFrame) / 1000 : 0;
    const speed = Number($<HTMLSelectElement>("speed").value);
    const d = duration(scene);
    t = Math.min(d, (isFinite(t) ? t : 0) + dt * speed);
    $<HTMLInputElement>("time").value = String(t);
    $("clock").textContent = `${t.toFixed(1)} s`;
    if (t >= d) {
      playing = false;
      $<HTMLButtonElement>("play").textContent = "Play";
    }
    rebuild();
  }
  lastFrame = now;
  const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
  renderer.draw(camera.matrix(aspect), background(), scene?.cloud ? 1.5 : 3);
  requestAnimationFrame(frame);
}

function controls() {
  let drag: { x: number; y: number; button: number } | null = null;
  const pointers = new Map<number, { x: number; y: number }>();
  let pinch = 0;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    drag = { x: e.clientX, y: e.clientY, button: e.button === 2 || e.shiftKey ? 2 : 0 };
  });
  canvas.addEventListener("pointermove", (e) => {
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch) camera.zoom(pinch / dist);
      pinch = dist;
      return;
    }
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (drag.button === 2) camera.pan(dx / canvas.clientHeight, dy / canvas.clientHeight);
    else camera.rotate(-dx * 0.008, dy * 0.008);
  });
  const end = (e: PointerEvent) => {
    pointers.delete(e.pointerId);
    pinch = 0;
    if (pointers.size === 0) drag = null;
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.zoom(Math.exp(e.deltaY * 0.001));
  }, { passive: false });

  for (const id of ["truth", "estimate", "reference", "loops", "cloud", "world", "colorByError"] as const) {
    const box = $<HTMLInputElement>(id);
    box.checked = layers[id];
    box.addEventListener("change", () => {
      layers[id] = box.checked;
      if (scene) legend(scene);
      rebuild();
    });
  }
  $("errscale").addEventListener("change", () => {
    if (scene) legend(scene);
    rebuild();
  });
  $<HTMLInputElement>("time").addEventListener("input", (e) => {
    t = Number((e.target as HTMLInputElement).value);
    $("clock").textContent = `${t.toFixed(1)} s`;
    rebuild();
  });
  $("play").addEventListener("click", () => {
    if (!scene) return;
    playing = !playing;
    if (playing && (!isFinite(t) || t >= duration(scene))) t = 0;
    $<HTMLButtonElement>("play").textContent = playing ? "Pause" : "Play";
  });
  $("fit").addEventListener("click", () => {
    if (!scene) return;
    const g = buildGeometry(scene, layers);
    camera.fit(g.min, g.max);
  });
}

async function main() {
  try {
    renderer = new Renderer(canvas);
  } catch (err) {
    $("error").textContent = String(err);
    return;
  }
  controls();
  const index = (await (await fetch("data/index.json")).json()) as { file: string; name: string; kind: string }[];
  const select = $<HTMLSelectElement>("scene");
  const groups: Record<string, HTMLOptGroupElement> = {};
  for (const s of index) {
    const label = s.kind === "euroc" ? "EuRoC MAV (real)" : "simulated missions";
    if (!groups[label]) {
      groups[label] = document.createElement("optgroup");
      groups[label].label = label;
      select.appendChild(groups[label]);
    }
    const o = document.createElement("option");
    o.value = s.file;
    o.textContent = s.name;
    groups[label].appendChild(o);
  }
  const wanted = new URLSearchParams(location.search).get("scene");
  if (wanted && index.some((s) => s.file === wanted)) select.value = wanted;
  select.addEventListener("change", () => load(select.value));
  await load(select.value);
  requestAnimationFrame(frame);
}

main();
