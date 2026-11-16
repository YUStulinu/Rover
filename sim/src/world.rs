//! The world: a room with a floor, a ceiling, four walls and box-shaped
//! obstacles, every surface covered with a procedural texture (value noise at
//! several scales plus random patches, so that corners can be tracked), and a
//! ray caster that returns the brightness, the depth and the kind of surface
//! seen along a ray.

use crate::math::{Rng, V3};
#[cfg(test)]
use crate::math::v3;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Surface {
    Floor = 0,
    Wall = 1,
    Ceiling = 2,
    Obstacle = 3,
}

#[derive(Clone, Debug, Serialize)]
pub struct Obstacle {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

pub struct World {
    pub size: [f64; 3], // the room spans [0, size] on each axis
    pub boxes: Vec<Obstacle>,
    seed: u64,
}

pub struct Hit {
    pub t: f64, // distance along the (unit) ray
    pub surface: Surface,
    pub value: f64, // brightness 0..255
}

fn hash(ix: i64, iy: i64, s: u64) -> f64 {
    let mut h = (ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (iy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ s.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn value_noise(u: f64, v: f64, s: u64) -> f64 {
    let (fu, fv) = (u.floor(), v.floor());
    let (iu, iv) = (fu as i64, fv as i64);
    let (a, b) = (u - fu, v - fv);
    let (a, b) = (a * a * (3.0 - 2.0 * a), b * b * (3.0 - 2.0 * b));
    let h00 = hash(iu, iv, s);
    let h10 = hash(iu + 1, iv, s);
    let h01 = hash(iu, iv + 1, s);
    let h11 = hash(iu + 1, iv + 1, s);
    (1.0 - b) * ((1.0 - a) * h00 + a * h10) + b * ((1.0 - a) * h01 + a * h11)
}

/// Brightness of a surface point given its 2D coordinates (meters) on the surface.
pub fn texture(u: f64, v: f64, s: u64) -> f64 {
    // smooth variation at three scales
    let mut x = 0.45 * value_noise(u / 0.9, v / 0.9, s) + 0.3 * value_noise(u / 0.25, v / 0.25, s + 1) + 0.25 * value_noise(u / 0.07, v / 0.07, s + 2);
    // random rectangles on a 0.35 m grid: sharp edges and corners
    let (cu, cv) = ((u / 0.35).floor() as i64, (v / 0.35).floor() as i64);
    let r = hash(cu, cv, s + 3);
    if r < 0.45 {
        let fu = u / 0.35 - cu as f64;
        let fv = v / 0.35 - cv as f64;
        let w = 0.25 + 0.5 * hash(cu, cv, s + 4);
        let h = 0.25 + 0.5 * hash(cu, cv, s + 5);
        if fu > 0.5 - w / 2.0 && fu < 0.5 + w / 2.0 && fv > 0.5 - h / 2.0 && fv < 0.5 + h / 2.0 {
            x = if hash(cu, cv, s + 6) < 0.5 { 0.12 + 0.2 * x } else { 0.75 + 0.2 * x };
        }
    }
    (x * 255.0).clamp(0.0, 255.0)
}

impl World {
    /// A room of 12 x 10 x 3 m with up to `n` boxes placed at random in its middle, at least 1.2 m apart.
    pub fn random(seed: u64, n: usize) -> World {
        let mut rng = Rng(seed.wrapping_mul(2654435761) | 1);
        let size = [12.0, 10.0, 3.0];
        let mut boxes: Vec<Obstacle> = Vec::new();
        let mut tries = 0;
        while boxes.len() < n && tries < 1000 {
            tries += 1;
            let w = 0.5 + rng.uniform() * 1.0;
            let d = 0.5 + rng.uniform() * 1.0;
            let h = 0.5 + rng.uniform() * 1.5;
            // inside the inner part of the room: a lane 2.6 m wide stays free along the walls
            let x = 2.6 + rng.uniform() * (size[0] - 5.2 - w);
            let y = 2.6 + rng.uniform() * (size[1] - 5.2 - d);
            if boxes.iter().any(|b| x < b.max[0] + 1.2 && x + w > b.min[0] - 1.2 && y < b.max[1] + 1.2 && y + d > b.min[1] - 1.2) {
                continue;
            }
            boxes.push(Obstacle { min: [x, y, 0.0], max: [x + w, y + d, h] });
        }
        World { size, boxes, seed }
    }

    /// The first surface along a ray (dir must be a unit vector).
    pub fn cast(&self, o: V3, d: V3) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        let mut consider = |t: f64, surface: Surface, u: f64, v: f64, s: u64, shade: f64| {
            if t > 1e-6 && best.as_ref().map_or(true, |b| t < b.t) {
                best = Some(Hit { t, surface, value: texture(u, v, self.seed.wrapping_add(s)) * shade });
            }
        };
        // the room: the inside of a box
        for axis in 0..3 {
            let (oa, da) = (o.get(axis), d.get(axis));
            if da.abs() < 1e-12 {
                continue;
            }
            for (plane, idx) in [(0.0, 0), (self.size[axis], 1)] {
                let t = (plane - oa) / da;
                if t <= 0.0 {
                    continue;
                }
                let p = o + d * t;
                let inside = (0..3).all(|k| k == axis || (p.get(k) >= -1e-9 && p.get(k) <= self.size[k] + 1e-9));
                if !inside {
                    continue;
                }
                let (surface, u, v, shade) = match axis {
                    2 if idx == 0 => (Surface::Floor, p.x, p.y, 0.85),
                    2 => (Surface::Ceiling, p.x, p.y, 0.6),
                    0 => (Surface::Wall, p.y, p.z, 1.0),
                    _ => (Surface::Wall, p.x, p.z, 0.95),
                };
                consider(t, surface, u, v, (axis * 2 + idx) as u64 * 101, shade);
            }
        }
        // obstacles: slab test, then the face that was entered
        for (bi, b) in self.boxes.iter().enumerate() {
            let mut t0 = 0.0f64;
            let mut t1 = f64::INFINITY;
            let mut face = 0;
            let mut ok = true;
            for axis in 0..3 {
                let (oa, da) = (o.get(axis), d.get(axis));
                if da.abs() < 1e-12 {
                    if oa < b.min[axis] || oa > b.max[axis] {
                        ok = false;
                        break;
                    }
                    continue;
                }
                let (mut ta, mut tb) = ((b.min[axis] - oa) / da, (b.max[axis] - oa) / da);
                if ta > tb {
                    std::mem::swap(&mut ta, &mut tb);
                }
                if ta > t0 {
                    t0 = ta;
                    face = axis;
                }
                t1 = t1.min(tb);
                if t0 > t1 {
                    ok = false;
                    break;
                }
            }
            if !ok || t0 <= 1e-6 {
                continue;
            }
            let p = o + d * t0;
            let (u, v, shade) = match face {
                0 => (p.y, p.z, 0.9),
                1 => (p.x, p.z, 0.8),
                _ => (p.x, p.y, 1.0),
            };
            consider(t0, Surface::Obstacle, u, v, 1000 + bi as u64 * 37 + face as u64, shade);
        }
        best
    }

    /// Whether a disc of radius r at (x, y) on the floor overlaps an obstacle or a wall.
    pub fn collides(&self, x: f64, y: f64, r: f64) -> bool {
        if x < r || y < r || x > self.size[0] - r || y > self.size[1] - r {
            return true;
        }
        self.boxes.iter().any(|b| {
            let cx = x.clamp(b.min[0], b.max[0]);
            let cy = y.clamp(b.min[1], b.max[1]);
            (x - cx).powi(2) + (y - cy).powi(2) < r * r
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays() {
        let w = World { size: [12.0, 10.0, 3.0], boxes: vec![Obstacle { min: [5.0, 4.0, 0.0], max: [6.0, 5.0, 1.0] }], seed: 1 };
        // straight down from 1 m: the floor at 1 m
        let h = w.cast(v3(2.0, 2.0, 1.0), v3(0.0, 0.0, -1.0)).unwrap();
        assert_eq!(h.surface, Surface::Floor);
        assert!((h.t - 1.0).abs() < 1e-9);
        // toward the box along x at 0.5 m high: its face at x = 5
        let h = w.cast(v3(2.0, 4.5, 0.5), v3(1.0, 0.0, 0.0)).unwrap();
        assert_eq!(h.surface, Surface::Obstacle);
        assert!((h.t - 3.0).abs() < 1e-9);
        // above the box: the far wall at x = 12
        let h = w.cast(v3(2.0, 4.5, 1.5), v3(1.0, 0.0, 0.0)).unwrap();
        assert_eq!(h.surface, Surface::Wall);
        assert!((h.t - 10.0).abs() < 1e-9);
        assert!(w.collides(5.5, 4.5, 0.2) && !w.collides(2.0, 2.0, 0.2) && w.collides(0.1, 5.0, 0.2));
        let t = texture(1.234, 2.345, 9);
        assert!((0.0..=255.0).contains(&t));
        // a random world keeps the start free
        let r = World::random(3, 8);
        assert!(!r.collides(1.5, 1.5, 0.4));
    }
}
