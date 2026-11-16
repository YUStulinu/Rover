//! The rover: unicycle dynamics with actuator lag and limits, small body
//! vibrations from the ground, a stereo camera rig and an IMU with the noise
//! model of EuRoC's ADIS16448 (white noise and bias random walk).

use crate::math::{v3, Rng, M3, V3};
use crate::world::World;

pub const GRAVITY: V3 = v3(0.0, 0.0, -9.81);

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub w: usize,
    pub h: usize,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub r_bc: M3, // camera to body
    pub t_bc: V3,
}

pub struct ImuNoise {
    pub gyro_noise: f64,
    pub gyro_walk: f64,
    pub acc_noise: f64,
    pub acc_walk: f64,
}

pub const NOISE: ImuNoise = ImuNoise { gyro_noise: 1.6968e-4, gyro_walk: 1.9393e-5, acc_noise: 2.0e-3, acc_walk: 3.0e-3 };

/// The true state of the body (the IMU frame): x forward, y left, z up.
#[derive(Clone, Copy, Debug)]
pub struct State {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
    pub v: f64, // forward speed
    pub w: f64, // yaw rate
}

pub struct Rover {
    pub s: State,
    pub radius: f64,
    pub height: f64,
    pub cams: [Camera; 2],
    pub max_v: f64,
    pub max_w: f64,
    pub max_acc: f64,
    pub tau: f64,
    pub collided: bool,
    rng: Rng,
    bump_phase: [f64; 4],
    pub bg: V3,
    pub ba: V3,
}

pub struct ImuSample {
    pub t: f64,
    pub gyro: V3,
    pub acc: V3,
}

impl Rover {
    pub fn new(x: f64, y: f64, yaw: f64, seed: u64) -> Rover {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9) | 1);
        // cameras 0.2 m ahead, 0.35 m up, 11 cm apart, tilted 10 degrees down
        let tilt = 10f64.to_radians();
        let r0 = M3([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]);
        let r_bc = M3::from_rpy(0.0, tilt, 0.0).mul(&r0);
        let cam = |dy: f64| Camera { w: 376, h: 240, fx: 230.0, fy: 230.0, cx: 188.0, cy: 120.0, r_bc, t_bc: v3(0.2, dy, 0.35) };
        let bump_phase = [rng.uniform() * 6.28, rng.uniform() * 6.28, rng.uniform() * 6.28, rng.uniform() * 6.28];
        Rover {
            s: State { t: 0.0, x, y, yaw, v: 0.0, w: 0.0 },
            radius: 0.3,
            height: 0.1,
            cams: [cam(0.055), cam(-0.055)],
            max_v: 1.0,
            max_w: 1.5,
            max_acc: 1.5,
            tau: 0.15,
            collided: false,
            bg: v3(0.003 * rng.gauss(), 0.003 * rng.gauss(), 0.003 * rng.gauss()),
            ba: v3(0.05 * rng.gauss(), 0.05 * rng.gauss(), 0.05 * rng.gauss()),
            rng,
            bump_phase,
        }
    }

    /// The full 3D pose of the body at state s: the ground makes the body
    /// roll, pitch and bounce a little, more when it drives faster.
    pub fn pose(&self, s: &State) -> (M3, V3) {
        let a = 0.3 + 0.7 * (s.v.abs() / self.max_v);
        let p = &self.bump_phase;
        let roll = a * (0.012 * (2.0 * std::f64::consts::PI * 1.7 * s.t + p[0]).sin() + 0.006 * (2.0 * std::f64::consts::PI * 4.1 * s.t + p[1]).sin());
        let pitch = a * (0.010 * (2.0 * std::f64::consts::PI * 1.3 * s.t + p[2]).sin() + 0.005 * (2.0 * std::f64::consts::PI * 3.7 * s.t + p[3]).sin());
        let z = self.height + a * 0.004 * (2.0 * std::f64::consts::PI * 2.3 * s.t + p[1]).sin();
        (M3::from_rpy(roll, pitch, s.yaw), v3(s.x, s.y, z))
    }

    /// Advance by h seconds toward the commanded speed and yaw rate.
    pub fn integrate(&mut self, h: f64, v_cmd: f64, w_cmd: f64, world: &World) {
        let v_cmd = v_cmd.clamp(-self.max_v, self.max_v);
        let w_cmd = w_cmd.clamp(-self.max_w, self.max_w);
        let dv = ((v_cmd - self.s.v) / self.tau).clamp(-self.max_acc, self.max_acc);
        let dw = (w_cmd - self.s.w) / self.tau;
        let mut n = self.s;
        n.v += dv * h;
        n.w += dw * h;
        n.yaw += n.w * h;
        n.x += n.v * n.yaw.cos() * h;
        n.y += n.v * n.yaw.sin() * h;
        n.t += h;
        if world.collides(n.x, n.y, self.radius) {
            // stop at the obstacle
            n.x = self.s.x;
            n.y = self.s.y;
            n.v = 0.0;
            self.collided = true;
        }
        self.s = n;
    }

    /// An IMU reading at state b, from the poses at a, b and c (equally spaced).
    pub fn imu(&mut self, a: &State, b: &State, c: &State, dt_imu: f64) -> ImuSample {
        // angular velocity in the body frame and the acceleration in the world by finite differences
        let (r0, p0) = self.pose(a);
        let (r1, p1) = self.pose(b);
        let (r2, p2) = self.pose(c);
        let h = c.t - b.t;
        let gyro = (r1.t().mul(&r2).log() + r0.t().mul(&r1).log()) * (0.5 / h);
        let acc_w = (p2 - p1 * 2.0 + p0) * (1.0 / (h * h));
        let f = r1.t().mul_v(acc_w - GRAVITY);
        // noise: white noise for the sampling interval, biases walking
        let gn = NOISE.gyro_noise / dt_imu.sqrt();
        let an = NOISE.acc_noise / dt_imu.sqrt();
        let gw = NOISE.gyro_walk * dt_imu.sqrt();
        let aw = NOISE.acc_walk * dt_imu.sqrt();
        let r = &mut self.rng;
        self.bg = self.bg + v3(gw * r.gauss(), gw * r.gauss(), gw * r.gauss());
        self.ba = self.ba + v3(aw * r.gauss(), aw * r.gauss(), aw * r.gauss());
        ImuSample {
            t: b.t,
            gyro: gyro + self.bg + v3(gn * r.gauss(), gn * r.gauss(), gn * r.gauss()),
            acc: f + self.ba + v3(an * r.gauss(), an * r.gauss(), an * r.gauss()),
        }
    }
}

/// One image of a camera at a body pose: brightness (8-bit), depth (mm,
/// 16-bit, 0 where nothing is hit) and the surface kind of each pixel.
pub struct Render {
    pub gray: Vec<u8>,
    pub depth_mm: Vec<u16>,
    pub label: Vec<u8>,
}

pub fn render(world: &World, cam: &Camera, r_wb: &M3, p_wb: V3, seed: u64) -> Render {
    let (w, h) = (cam.w, cam.h);
    let r_wc = r_wb.mul(&cam.r_bc);
    let o = p_wb + r_wb.mul_v(cam.t_bc);
    let mut gray = vec![0u8; w * h];
    let mut depth_mm = vec![0u16; w * h];
    let mut label = vec![0u8; w * h];
    let threads = 4;
    let rows_per = h.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut g_rest = gray.as_mut_slice();
        let mut d_rest = depth_mm.as_mut_slice();
        let mut l_rest = label.as_mut_slice();
        for k in 0..threads {
            let y0 = k * rows_per;
            let y1 = (y0 + rows_per).min(h);
            if y0 >= y1 {
                break;
            }
            let (g, gr) = g_rest.split_at_mut((y1 - y0) * w);
            let (d, dr) = d_rest.split_at_mut((y1 - y0) * w);
            let (l, lr) = l_rest.split_at_mut((y1 - y0) * w);
            g_rest = gr;
            d_rest = dr;
            l_rest = lr;
            scope.spawn(move || {
                let mut rng = Rng(seed ^ ((y0 as u64 + 1) * 0x9E37_79B9_7F4A_7C15));
                for y in y0..y1 {
                    for x in 0..w {
                        let mut acc = 0.0;
                        // 2 x 2 supersampling against aliasing
                        for (sx, sy) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
                            let dc = v3((x as f64 + sx - cam.cx) / cam.fx, (y as f64 + sy - cam.cy) / cam.fy, 1.0);
                            let dw = r_wc.mul_v(dc);
                            let dw = dw * (1.0 / dw.norm());
                            acc += world.cast(o, dw).map_or(0.0, |hit| hit.value);
                        }
                        let i = (y - y0) * w + x;
                        // pixel noise of the sensor
                        g[i] = (acc / 4.0 + 1.5 * rng.gauss()).round().clamp(0.0, 255.0) as u8;
                        let dc = v3((x as f64 - cam.cx) / cam.fx, (y as f64 - cam.cy) / cam.fy, 1.0);
                        let dw = r_wc.mul_v(dc);
                        let n = dw.norm();
                        if let Some(hit) = world.cast(o, dw * (1.0 / n)) {
                            // depth along the optical axis: distance / |(x, y, 1)|
                            d[i] = ((hit.t / n) * 1000.0).round().min(65535.0) as u16;
                            l[i] = hit.surface as u8;
                        }
                    }
                }
            });
        }
    });
    Render { gray, depth_mm, label }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imu_at_rest_measures_gravity() {
        let world = World::random(1, 0);
        let mut r = Rover::new(2.0, 2.0, 0.3, 7);
        let mut acc = v3(0.0, 0.0, 0.0);
        let n = 400;
        let mut a = r.s;
        r.integrate(0.005, 0.0, 0.0, &world);
        let mut b = r.s;
        for _ in 0..n {
            r.integrate(0.005, 0.0, 0.0, &world);
            let c = r.s;
            let s = r.imu(&a, &b, &c, 0.005);
            acc = acc + s.acc;
            a = b;
            b = c;
        }
        let m = acc * (1.0 / n as f64);
        // about +9.81 on the body z axis (plus the accelerometer bias)
        assert!((m.z - 9.81).abs() < 0.3, "{:?}", m);
    }

    #[test]
    fn drives_and_stops_at_walls() {
        let world = World::random(1, 0);
        let mut r = Rover::new(2.0, 2.0, 0.0, 7);
        for _ in 0..1000 {
            r.integrate(0.01, 1.0, 0.0, &world);
        }
        assert!(r.s.x > 8.0 && r.s.v > 0.9);
        for _ in 0..1000 {
            r.integrate(0.01, 1.0, 0.0, &world);
        }
        assert!(r.collided && r.s.x < 12.0 - 0.29);
    }
}
